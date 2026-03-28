from __future__ import annotations

import json
import logging
import os
import re
import uuid
from collections import Counter
from typing import Any, Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from fastapi import Body, FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from openai import APIError, APITimeoutError, OpenAI
from pydantic import BaseModel, Field, HttpUrl

load_dotenv()

TINYFISH_API_KEY = os.getenv("TINYFISH_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not TINYFISH_API_KEY:
    raise RuntimeError("Missing required environment variable: TINYFISH_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Missing required environment variable: OPENAI_API_KEY")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("web2api")

app = FastAPI(
    title="Web2API",
    description=(
        "Autonomous web extraction for modern product discovery.\n\n"
        "Web2API turns messy storefronts and listing pages into structured JSON by combining "
        "TinyFish browser automation with OpenAI-powered schema inference and semantic filtering."
    ),
    version="1.0.0",
)

openai_client = OpenAI(api_key=OPENAI_API_KEY, timeout=45.0)

TINYFISH_URL = os.getenv("TINYFISH_URL", "https://agent.tinyfish.ai/v1/automation/run")
DEFAULT_MAX_ITEMS = 15
SYSTEM_PROMPT = (
    "You are an expert data extraction agent. Parse the raw website data and return a JSON array "
    "of objects matching the exact requested schema. If a field is missing, use null. Prices must be numbers."
)
SHOWCASE_SYSTEM_PROMPT = (
    "You are a product merchandising strategist. Given extracted products, create a concise storefront "
    "showcase JSON object. Make it polished, commercial, and grounded in the provided data. "
    "Do not invent technical specs."
)
OPENAI_MODEL = "gpt-4.1-mini"
RAW_DATA_CHAR_LIMIT = 120_000
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
)
REQUEST_TIMEOUT = 12
ANALYSIS_STORE: dict[str, dict[str, Any]] = {}
SCHEMA_PRESETS: dict[str, dict[str, Any]] = {
    "product_standard": {
        "label": "Product Standard",
        "description": "Balanced product catalog schema for storefront data and showcase generation.",
        "fields": [
            "product_name",
            "subtitle",
            "current_price",
            "original_price",
            "product_url",
            "image_url",
        ],
    },
    "pricing_audit": {
        "label": "Pricing Audit",
        "description": "Focus on pricing, discount visibility, and basic landing links.",
        "fields": [
            "product_name",
            "category",
            "current_price",
            "original_price",
            "discount_amount",
            "price_status",
            "product_url",
        ],
    },
    "merchandising_cards": {
        "label": "Merchandising Cards",
        "description": "Creative product-card schema for quick demo grids and editorial previews.",
        "fields": [
            "product_name",
            "subtitle",
            "price_label",
            "badge",
            "product_url",
            "image_url",
            "image_description",
        ],
    },
}


class ProductSchema(BaseModel):
    product_name: str
    subtitle: Optional[str] = None
    current_price: float
    original_price: float
    product_url: str
    image_url: str


class ShowcaseRequest(BaseModel):
    source_url: str
    semantic_query: Optional[str] = None
    products: list[ProductSchema] = Field(min_length=1, max_length=15)


class ShowcaseCard(BaseModel):
    product_name: str
    subtitle: Optional[str] = None
    price_label: str
    image_url: str
    product_url: str
    image_description: str
    marketing_copy: str
    badge: str


class ShowcaseResponse(BaseModel):
    collection_title: str
    collection_subtitle: str
    visual_direction: str
    cards: list[ShowcaseCard]


class ExtractionDetailResponse(BaseModel):
    source_url: str
    semantic_query: Optional[str] = None
    max_items: int
    schema_preset: str
    schema_fields: list[str]
    tinyfish_status: Optional[str] = None
    raw_product_count: int
    schema_preview: list[dict[str, Any]]
    normalized_products: list[ProductSchema]
    raw_openai_items: list[dict[str, Any]]
    dropped_reasons: list[str]


class AnalyzeRequest(BaseModel):
    url: HttpUrl
    goal: str


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.replace(",", "")
        match = re.search(r"-?\d+(?:\.\d+)?", cleaned)
        if match:
            try:
                return float(match.group(0))
            except ValueError:
                return None
    return None


def _format_price(current_price: float, original_price: float) -> str:
    current_label = f"${current_price:,.2f}"
    if original_price > current_price:
        return f"{current_label} (was ${original_price:,.2f})"
    return current_label


def _build_tinyfish_goal(max_items: int) -> str:
    return (
        "Extract product listing data from the collection page. Wait for the main product grid to load, "
        "close any popups, and scroll only as much as needed to reveal more product cards and lazy-loaded images. "
        f"Extract up to the first {max_items} product cards that become visible across the loaded grid. "
        "For each product card, explicitly capture product_name, subtitle or category, current_price, "
        "original_price if present, product_url or href, image_url or image src, and any badges or labels. "
        "Prefer absolute URLs for product_url and image_url. Return raw text/JSON."
    )


def _get_schema_preset(schema_preset: str) -> dict[str, Any]:
    preset = SCHEMA_PRESETS.get(schema_preset)
    if not preset:
        raise HTTPException(status_code=422, detail=f"Unknown schema_preset: {schema_preset}")
    return preset


def _build_schema_instructions(schema_preset: str) -> str:
    preset = _get_schema_preset(schema_preset)
    fields = ", ".join(preset["fields"])
    return (
        f"Use the '{schema_preset}' schema preset. "
        f"Return a JSON array of objects with these exact fields in each item: {fields}. "
        "If a field is missing, use null. Prices must be numbers when numeric fields exist."
    )


def _build_user_prompt(raw_data: str, semantic_query: Optional[str], schema_preset: str) -> str:
    prompt = (
        f"{_build_schema_instructions(schema_preset)} "
        f"Here is the raw data: {raw_data}. Extract the items."
    )
    if semantic_query:
        prompt += (
            f" FILTERING REQUIREMENT: The user only wants items that match this description: "
            f"'{semantic_query}'. Filter the list and ONLY return items that semantically match "
            f"this vibe or description based on your reasoning."
        )
    return prompt


def _parse_json_array(text: str) -> list[dict[str, Any]]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(
            r"^```(?:json)?\s*|\s*```$",
            "",
            cleaned,
            flags=re.IGNORECASE | re.DOTALL,
        ).strip()

    parsed: Any
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("[")
        end = cleaned.rfind("]")
        if start == -1 or end == -1 or start >= end:
            raise ValueError("OpenAI response was not valid JSON.")
        parsed = json.loads(cleaned[start : end + 1])

    if isinstance(parsed, dict):
        for key in ("items", "data", "results", "products"):
            if isinstance(parsed.get(key), list):
                parsed = parsed[key]
                break

    if not isinstance(parsed, list):
        raise ValueError("OpenAI did not return a JSON array.")

    return [item for item in parsed if isinstance(item, dict)]


def _parse_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(
            r"^```(?:json)?\s*|\s*```$",
            "",
            cleaned,
            flags=re.IGNORECASE | re.DOTALL,
        ).strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or start >= end:
            raise ValueError("OpenAI response was not valid JSON.")
        parsed = json.loads(cleaned[start : end + 1])

    if not isinstance(parsed, dict):
        raise ValueError("OpenAI did not return a JSON object.")
    return parsed


def normalize_url(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.path.rstrip("/") or "/"
    return f"{parsed.scheme}://{parsed.netloc}{path}"


def fetch_html(url: str) -> tuple[str, requests.Response]:
    response = requests.get(
        url,
        timeout=REQUEST_TIMEOUT,
        headers={"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.9"},
    )
    response.raise_for_status()
    return response.text, response


def clean_text(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", " ", value).strip()


def same_host_links(soup: BeautifulSoup, base_url: str) -> list[str]:
    parsed_base = urlparse(base_url)
    collected: list[str] = []
    for anchor in soup.select("a[href]"):
        href = anchor.get("href", "").strip()
        if not href or href.startswith(("#", "mailto:", "tel:", "javascript:")):
            continue
        absolute = urljoin(base_url, href)
        parsed = urlparse(absolute)
        if parsed.scheme not in {"http", "https"} or parsed.netloc != parsed_base.netloc:
            continue
        normalized = normalize_url(absolute)
        if normalized != normalize_url(base_url):
            collected.append(normalized)
    return list(dict.fromkeys(collected))[:20]


def detect_entity(goal: str, title: str, url: str) -> str:
    text = f"{goal} {title} {url}".lower()
    mapping = {
        "product": [
            "product", "price", "store", "shop", "cart", "buy", "shoe", "shoes", "sneaker",
            "sneakers", "apparel", "collection", "men", "women", "kids", "tops", "bottoms",
        ],
        "job_listing": ["job", "career", "hiring", "role", "position"],
        "property_listing": ["property", "listing", "rent", "sale", "home", "condo", "apartment"],
        "event": ["event", "conference", "meetup", "workshop", "register"],
    }
    for entity, keywords in mapping.items():
        if any(keyword in text for keyword in keywords):
            return entity
    return "article"


def extract_price_text(value: str) -> str:
    match = re.search(r"((?:sgd|usd|eur|gbp|aud|cad|s\$|\$|€|£)\s?\d[\d,]*(?:\.\d{1,2})?)", value, re.IGNORECASE)
    if match:
        return clean_text(match.group(1))
    return ""


def product_candidates_from_containers(soup: BeautifulSoup, base_url: str) -> list[dict[str, Any]]:
    base_host = urlparse(base_url).netloc
    seen: set[str] = set()
    candidates: list[dict[str, Any]] = []

    for container in soup.select("article, div, li"):
        anchor = container.select_one("a[href]")
        image_node = container.select_one("img")
        if not anchor:
            continue
        href = urljoin(base_url, anchor.get("href", "").strip())
        parsed = urlparse(href)
        if parsed.scheme not in {"http", "https"} or parsed.netloc != base_host:
            continue
        normalized = normalize_url(href)
        if normalized in seen or normalized == normalize_url(base_url):
            continue
        if not looks_like_product_url(normalized, base_host):
            continue

        title = ""
        heading = container.select_one("h1, h2, h3, h4")
        if heading:
            title = clean_text(heading.get_text(" ", strip=True))
        if not title:
            title = clean_text(anchor.get_text(" ", strip=True))
        if len(title) < 4:
            continue
        if "/w/" in normalized and re.search(r"\(\d+\)", title):
            continue

        text_blob = clean_text(container.get_text(" ", strip=True))
        price_text = extract_price_text(text_blob)
        if not heading and not image_node and not price_text:
            continue

        image_url = ""
        if image_node and image_node.get("src"):
            image_url = urljoin(base_url, image_node.get("src"))
        summary_node = container.select_one("p, span")
        summary = clean_text(summary_node.get_text(" ", strip=True)) if summary_node else ""

        candidate: dict[str, Any] = {
            "title": title,
            "url": normalized,
            "summary": summary,
        }
        if price_text:
            candidate["price_text"] = price_text
        if image_url:
            candidate["image_url"] = image_url
        candidates.append(candidate)
        seen.add(normalized)

    return candidates[:12]


def looks_like_article_url(url: str, base_host: str) -> bool:
    parsed = urlparse(url)
    path = parsed.path.lower().strip("/")
    if parsed.netloc != base_host or not path:
        return False
    disallowed = {
        "about", "contact", "privacy", "terms", "advertise", "latest", "tag", "category", "topics",
        "newsletter", "events", "podcasts", "news", "newest", "front", "newcomments", "ask", "show",
        "jobs", "submit", "login", "best", "item", "user",
    }
    first = path.split("/")[0]
    if first in disallowed:
        return False
    return len(path.split("/")) >= 2 or re.search(r"\d{4}", path) is not None


def looks_like_product_url(url: str, base_host: str) -> bool:
    parsed = urlparse(url)
    if parsed.netloc != base_host:
        return False
    path = parsed.path.lower().strip("/")
    if not path:
        return False
    disallowed = {
        "help", "retail", "orders", "member", "members", "login", "join", "cart",
        "favorites", "wishlist", "privacy", "terms", "about", "contact",
    }
    segments = [segment for segment in path.split("/") if segment]
    if any(segment in disallowed for segment in segments):
        return False
    return len(segments) >= 2


def extract_hacker_news_records(soup: BeautifulSoup) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, row in enumerate(soup.select("tr.athing"), start=1):
        title_link = row.select_one(".titleline > a, .title a")
        if not title_link:
            continue
        story_url = clean_text(title_link.get("href"))
        if story_url.startswith("item?id="):
            story_url = urljoin("https://news.ycombinator.com/", story_url)
        title = clean_text(title_link.get_text(" ", strip=True))
        if not title:
            continue

        subtext_row = row.find_next_sibling("tr")
        score = author = age = comments = ""
        if subtext_row:
            score_node = subtext_row.select_one(".score")
            author_node = subtext_row.select_one(".hnuser")
            age_node = subtext_row.select_one(".age")
            comment_links = subtext_row.select("a")
            score = clean_text(score_node.get_text(" ", strip=True)) if score_node else ""
            author = clean_text(author_node.get_text(" ", strip=True)) if author_node else ""
            age = clean_text(age_node.get_text(" ", strip=True)) if age_node else ""
            for link in reversed(comment_links):
                text = clean_text(link.get_text(" ", strip=True))
                if "comment" in text.lower() or text.lower() == "discuss":
                    comments = text
                    break

        record = {"id": index, "title": title, "url": story_url}
        if author:
            record["author"] = author
        if age:
            record["published_at"] = age
        if score:
            record["score"] = score
        if comments:
            record["comments"] = comments
        records.append(record)
    return records[:12]


def article_candidates_from_containers(soup: BeautifulSoup, base_url: str) -> list[dict[str, Any]]:
    base_host = urlparse(base_url).netloc
    seen: set[str] = set()
    candidates: list[dict[str, Any]] = []
    for container in soup.select("article, div, li"):
        headings = container.select("h1 a[href], h2 a[href], h3 a[href], h4 a[href]")
        if not headings:
            continue
        anchor = headings[0]
        href = urljoin(base_url, anchor.get("href", "").strip())
        normalized = normalize_url(href)
        if not looks_like_article_url(normalized, base_host) or normalized in seen:
            continue
        if container.find_parent(["header", "nav"]):
            continue
        title = clean_text(anchor.get_text(" ", strip=True))
        if len(title) < 20:
            continue
        summary_node = container.select_one("p")
        time_node = container.select_one("time")
        author_node = container.select_one('[rel="author"], .author, [class*="author"], [data-testid*="author"]')
        image_node = container.select_one("img")
        candidates.append(
            {
                "title": title,
                "url": normalized,
                "summary": clean_text(summary_node.get_text(" ", strip=True)) if summary_node else "",
                "published_at": clean_text(time_node.get("datetime") or time_node.get_text(" ", strip=True)) if time_node else "",
                "author": clean_text(author_node.get_text(" ", strip=True)) if author_node else "",
                "image_url": urljoin(base_url, image_node.get("src")) if image_node and image_node.get("src") else "",
            }
        )
        seen.add(normalized)
    return candidates[:12]


def article_candidates_from_links(soup: BeautifulSoup, base_url: str) -> list[dict[str, Any]]:
    base_host = urlparse(base_url).netloc
    seen: set[str] = set()
    candidates: list[dict[str, Any]] = []
    for anchor in soup.select("a[href]"):
        href = urljoin(base_url, anchor.get("href", "").strip())
        normalized = normalize_url(href)
        title = clean_text(anchor.get_text(" ", strip=True))
        if normalized in seen or len(title) < 28:
            continue
        if not looks_like_article_url(normalized, base_host):
            continue
        seen.add(normalized)
        candidates.append({"title": title, "url": normalized, "summary": "", "published_at": "", "author": "", "image_url": ""})
    return candidates[:12]


def extract_article_records(soup: BeautifulSoup, base_url: str) -> list[dict[str, Any]]:
    if urlparse(base_url).netloc == "news.ycombinator.com":
        records = extract_hacker_news_records(soup)
        if records:
            return records
    candidates = article_candidates_from_containers(soup, base_url)
    if len(candidates) < 3:
        candidates = article_candidates_from_links(soup, base_url)
    records: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates[:8], start=1):
        record = {"id": index, "title": candidate["title"], "url": candidate["url"]}
        for key in ("summary", "published_at", "author", "image_url"):
            if candidate.get(key):
                record[key] = candidate[key]
        records.append(record)
    return records


def make_sample_records(
    entity: str,
    landing_title: str,
    landing_url: str,
    internal_links: list[str],
    soup: BeautifulSoup,
) -> list[dict[str, Any]]:
    if entity == "product":
        product_candidates = product_candidates_from_containers(soup, landing_url)
        if product_candidates:
            records: list[dict[str, Any]] = []
            for index, candidate in enumerate(product_candidates[:8], start=1):
                record: dict[str, Any] = {
                    "id": index,
                    "title": candidate["title"],
                    "url": candidate["url"],
                }
                if candidate.get("summary"):
                    record["summary"] = candidate["summary"]
                if candidate.get("price_text"):
                    record["price_text"] = candidate["price_text"]
                if candidate.get("image_url"):
                    record["image_url"] = candidate["image_url"]
                records.append(record)
            return records

    if entity == "article":
        article_records = extract_article_records(soup, landing_url)
        if article_records:
            return article_records

    description = ""
    meta_description = soup.select_one('meta[name="description"]')
    if meta_description:
        description = clean_text(meta_description.get("content"))

    if not internal_links:
        internal_links = [landing_url]

    records: list[dict[str, Any]] = []
    for index, link in enumerate(internal_links[:6], start=1):
        link_title = link.rstrip("/").split("/")[-1].replace("-", " ").replace("_", " ").strip().title()
        if not link_title:
            link_title = landing_title
        record: dict[str, Any] = {"id": index, "title": link_title or f"{entity.title()} {index}", "url": link}
        if entity == "product":
            record["summary"] = description or f"Detected product-like page from {landing_title}."
            record["price_text"] = "Price detected on page"
        elif entity == "job_listing":
            record["summary"] = description or f"Detected hiring-related content from {landing_title}."
            record["team"] = "Unknown team"
            record["location"] = "Unknown location"
        elif entity == "property_listing":
            record["summary"] = description or f"Detected property-related content from {landing_title}."
            record["location"] = "Unknown location"
            record["price_text"] = "Price detected on page"
        elif entity == "event":
            record["summary"] = description or f"Detected event-related content from {landing_title}."
            record["date_text"] = "Date not parsed"
            record["location"] = "Location not parsed"
        else:
            record["summary"] = description or f"Extracted from {landing_title}."
            record["published_hint"] = "No explicit publish date parsed"
        records.append(record)
    return records


def infer_schema_from_samples(samples: list[dict[str, Any]]) -> list[dict[str, str]]:
    if not samples:
        return [{"name": "title", "type": "string"}, {"name": "url", "type": "string"}]
    field_types: dict[str, str] = {}
    for key in samples[0].keys():
        value = next((sample.get(key) for sample in samples if sample.get(key) not in (None, "")), "")
        field_types[key] = "number" if isinstance(value, int) else "string"
    ordered: list[dict[str, str]] = []
    preferred_order = ["id", "title", "summary", "author", "published_at", "score", "comments", "price_text", "location", "url", "image_url"]
    for field in preferred_order:
        if field in field_types:
            ordered.append({"name": field, "type": field_types[field]})
    for field, field_type in field_types.items():
        if field not in {item["name"] for item in ordered}:
            ordered.append({"name": field, "type": field_type})
    return ordered


def summarize_link_patterns(links: list[str], canonical_url: str, entity: str) -> list[dict[str, str]]:
    if not links:
        return [{"page_type": "landing", "pattern": urlparse(canonical_url).path or "/", "notes": "Only the root page was accessible."}]
    parsed_paths = [urlparse(link).path or "/" for link in links]
    segment_counter = Counter(path.strip("/").split("/")[0] if path.strip("/") else "root" for path in parsed_paths)
    common_segment, _ = segment_counter.most_common(1)[0]
    listing_pattern = urlparse(canonical_url).path or "/"
    detail_pattern = f"/{common_segment}/:slug" if common_segment != "root" else "/:slug"
    detail_note = "Detail pages are inferred from deeper internal links."
    if entity == "article":
        detail_note = "Article detail pages are inferred from repeated editorial links."
    return [
        {"page_type": "landing", "pattern": "/", "notes": "Fetched the entry page successfully."},
        {"page_type": "listing", "pattern": listing_pattern, "notes": "This page behaves like a collection or category view."},
        {"page_type": "detail", "pattern": detail_pattern, "notes": detail_note},
    ]


def build_generated_api(analysis_id: str, entity: str, origin: str) -> dict[str, Any]:
    base_url = f"{origin.rstrip('/')}/api/generated"
    return {
        "base_url": base_url,
        "endpoints": [
            f"GET /items?analysis_id={analysis_id}",
            f"GET /items/{{id}}?analysis_id={analysis_id}",
            f"GET /schema?analysis_id={analysis_id}",
        ],
        "description": f"Working API endpoints for the stored '{entity}' analysis.",
    }


def build_quickstart(analysis_id: str, origin: str) -> dict[str, str]:
    base_url = f"{origin.rstrip('/')}/api/generated"
    return {
        "curl": f"curl '{base_url}/items?analysis_id={analysis_id}' \\\n  -H 'Accept: application/json'",
        "python": (
            "import requests\n\n"
            f"response = requests.get('{base_url}/items', params={{'analysis_id': '{analysis_id}'}})\n"
            "response.raise_for_status()\nprint(response.json())"
        ),
        "javascript": (
            f"const response = await fetch('{base_url}/items?analysis_id={analysis_id}');\n"
            "const data = await response.json();\nconsole.log(data);"
        ),
    }


def analyze_website(url: str, goal: str, origin: str) -> dict[str, Any]:
    html, response = fetch_html(url)
    soup = BeautifulSoup(html, "html.parser")
    title = clean_text(soup.title.string if soup.title and soup.title.string else "") or urlparse(str(response.url)).netloc
    hostname = urlparse(str(response.url)).netloc.replace("www.", "") or "unknown-site"
    entity = detect_entity(goal, title, str(response.url))
    internal_links = same_host_links(soup, str(response.url))
    samples = make_sample_records(entity, title, str(response.url), internal_links, soup)
    fields = infer_schema_from_samples(samples)
    site_map = summarize_link_patterns(internal_links, str(response.url), entity)
    analysis_id = uuid.uuid4().hex[:12]

    confidence = 0.64
    if len(internal_links) >= 5:
        confidence += 0.12
    if len(fields) >= 4:
        confidence += 0.08
    if samples:
        confidence += 0.06

    result = {
        "status": "success",
        "mode": "live",
        "analysis_id": analysis_id,
        "site": {
            "url": str(response.url),
            "hostname": hostname,
            "entity_guess": entity,
            "confidence": round(min(confidence, 0.96), 2),
            "title": title,
        },
        "site_map": site_map,
        "schema": {"entity": entity, "fields": fields},
        "generated_api": build_generated_api(analysis_id, entity, origin),
        "samples": samples,
        "agent_notes": [
            f"Fetched the live page for {hostname} and parsed its visible HTML structure.",
            f"Discovered {len(internal_links)} internal links and prioritized likely content pages over generic navigation.",
            f"Inferred a '{entity}' schema from the actual extracted records rather than guessing from navigation labels.",
            "Stored the analysis in memory and exposed working generated endpoints for the extracted result.",
        ],
        "quickstart": build_quickstart(analysis_id, origin),
        "product_summary": {
            "headline": "Turn a website into a structured data product.",
            "description": (
                "This backend performs a real live fetch of the target website, infers a schema from the returned HTML, "
                "and creates working API endpoints for the generated records."
            ),
        },
        "goal": goal,
    }
    ANALYSIS_STORE[analysis_id] = result
    return result


def get_saved_analysis(analysis_id: str) -> dict[str, Any]:
    analysis = ANALYSIS_STORE.get(analysis_id)
    if not analysis:
        raise HTTPException(status_code=404, detail="Unknown analysis_id. Run /api/analyze first.")
    return analysis


def _call_tinyfish(target_url: str, max_items: int) -> str:
    headers = {
        "X-API-Key": TINYFISH_API_KEY,
        "Content-Type": "application/json",
    }
    payload = {
        "url": target_url,
        "goal": _build_tinyfish_goal(max_items),
        "browser_profile": "lite",
        "proxy_config": {"enabled": False},
        "api_integration": "web2api",
    }

    try:
        response = requests.post(
            TINYFISH_URL,
            headers=headers,
            json=payload,
            timeout=180,
        )
        response.raise_for_status()
    except requests.Timeout as exc:
        logger.exception("TinyFish request timed out")
        raise HTTPException(status_code=500, detail="TinyFish request timed out.") from exc
    except requests.ConnectionError as exc:
        logger.exception("TinyFish host connection failed")
        raise HTTPException(
            status_code=500,
            detail=(
                f"TinyFish host connection failed for {TINYFISH_URL}. "
                "Check the TinyFish endpoint, DNS, or your network."
            ),
        ) from exc
    except requests.RequestException as exc:
        logger.exception("TinyFish request failed")
        raise HTTPException(status_code=500, detail=f"TinyFish request failed: {exc}") from exc

    try:
        return json.dumps(response.json(), ensure_ascii=False)
    except ValueError:
        return response.text


def _parse_tinyfish_payload(raw_data: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw_data)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _call_openai(raw_data: str, semantic_query: Optional[str], schema_preset: str) -> list[dict[str, Any]]:
    truncated_raw_data = raw_data[:RAW_DATA_CHAR_LIMIT]
    user_prompt = _build_user_prompt(truncated_raw_data, semantic_query, schema_preset)

    try:
        completion = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
    except APITimeoutError as exc:
        logger.exception("OpenAI request timed out")
        raise HTTPException(status_code=500, detail="OpenAI request timed out.") from exc
    except APIError as exc:
        logger.exception("OpenAI request failed")
        raise HTTPException(status_code=500, detail=f"OpenAI request failed: {exc}") from exc
    except Exception as exc:
        logger.exception("Unexpected OpenAI error")
        raise HTTPException(status_code=500, detail=f"Unexpected OpenAI error: {exc}") from exc

    content = ""
    if completion.choices:
        content = (completion.choices[0].message.content or "").strip()
    if not content:
        raise HTTPException(status_code=500, detail="OpenAI returned an empty response.")

    try:
        return _parse_json_array(content)
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=f"Failed to parse OpenAI JSON output: {exc}") from exc


def _normalize_products(items: list[dict[str, Any]]) -> list[ProductSchema]:
    normalized: list[ProductSchema] = []
    for item in items:
        candidate = {
            "product_name": str(item.get("product_name") or "").strip(),
            "subtitle": str(item["subtitle"]).strip() if item.get("subtitle") not in (None, "") else None,
            "current_price": _to_float(item.get("current_price")),
            "original_price": _to_float(item.get("original_price")),
            "product_url": str(item.get("product_url") or "").strip(),
            "image_url": str(item.get("image_url") or "").strip(),
        }

        if (
            not candidate["product_name"]
            or candidate["current_price"] is None
            or candidate["original_price"] is None
            or not candidate["product_url"]
            or not candidate["image_url"]
        ):
            continue

        try:
            normalized.append(ProductSchema(**candidate))
        except Exception:
            logger.warning("Skipping invalid product payload: %s", item)

    return normalized


def _build_schema_candidate(item: dict[str, Any], schema_preset: str) -> dict[str, Any]:
    subtitle = item.get("subtitle")
    if subtitle in (None, ""):
        subtitle = item.get("category")

    current_price = _to_float(item.get("current_price"))
    if current_price is None:
        current_price = _to_float(item.get("price"))

    original_price = _to_float(item.get("original_price"))
    if original_price is None:
        original_price = current_price

    product_url = item.get("product_url") or item.get("url")
    image_url = item.get("image_url") or item.get("image")

    base_name = str(item.get("product_name") or item.get("name") or "").strip() or None
    base_subtitle = str(subtitle).strip() if subtitle not in (None, "") else None
    base_product_url = str(product_url).strip() if product_url else None
    base_image_url = str(image_url).strip() if image_url else None
    category = str(item.get("category") or "").strip() or None
    badge = str(item.get("badge") or "").strip() if item.get("badge") else None
    labels = item.get("labels")
    if not badge and isinstance(labels, list) and labels:
        badge = str(labels[0]).strip() or None

    if schema_preset == "pricing_audit":
        discount_amount = None
        if current_price is not None and original_price is not None:
            discount_amount = max(original_price - current_price, 0.0)
        price_status = "discounted" if discount_amount and discount_amount > 0 else "full_price"
        return {
            "product_name": base_name,
            "category": category,
            "current_price": current_price,
            "original_price": original_price,
            "discount_amount": discount_amount,
            "price_status": price_status,
            "product_url": base_product_url,
        }

    if schema_preset == "merchandising_cards":
        price_label = None
        if current_price is not None:
            price_label = _format_price(current_price, original_price or current_price)
        return {
            "product_name": base_name,
            "subtitle": base_subtitle,
            "price_label": price_label,
            "badge": badge,
            "product_url": base_product_url,
            "image_url": base_image_url,
            "image_description": (
                f"Product image for {base_name}" if base_name else None
            ),
        }

    return {
        "product_name": base_name,
        "subtitle": base_subtitle,
        "current_price": current_price,
        "original_price": original_price,
        "product_url": base_product_url,
        "image_url": base_image_url,
    }


def _extract_raw_product_cards(tinyfish_payload: dict[str, Any]) -> list[dict[str, Any]]:
    result = tinyfish_payload.get("result")
    if not isinstance(result, dict):
        return []

    for key in ("product_cards", "products", "items", "results"):
        value = result.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


def _build_dropped_reasons(schema_preview: list[dict[str, Any]], schema_preset: str) -> list[str]:
    if schema_preset != "product_standard":
        return []
    reasons: list[str] = []
    for item in schema_preview:
        missing = []
        if not item.get("product_name"):
            missing.append("product_name")
        if item.get("current_price") is None:
            missing.append("current_price")
        if item.get("original_price") is None:
            missing.append("original_price")
        if not item.get("product_url"):
            missing.append("product_url")
        if not item.get("image_url"):
            missing.append("image_url")
        if missing:
            reasons.append(
                f"{item.get('product_name') or 'Unknown item'} missing required fields: {', '.join(missing)}"
            )
    return reasons


def _extract_details(
    url: str,
    semantic_query: Optional[str],
    max_items: int = DEFAULT_MAX_ITEMS,
    schema_preset: str = "product_standard",
) -> ExtractionDetailResponse:
    if not re.match(r"^https?://", url, flags=re.IGNORECASE):
        raise HTTPException(status_code=422, detail="`url` must start with http:// or https://")
    if max_items < 1 or max_items > 40:
        raise HTTPException(status_code=422, detail="`max_items` must be between 1 and 40.")
    preset = _get_schema_preset(schema_preset)

    raw_data = _call_tinyfish(url, max_items)
    if not raw_data.strip():
        raise HTTPException(status_code=500, detail="TinyFish returned empty data.")

    tinyfish_payload = _parse_tinyfish_payload(raw_data)
    raw_cards = _extract_raw_product_cards(tinyfish_payload)
    openai_items = _call_openai(raw_data, semantic_query, schema_preset)

    preview_source = openai_items or raw_cards
    schema_preview = [_build_schema_candidate(item, schema_preset) for item in preview_source]
    normalized_products = _normalize_products(openai_items) if schema_preset == "product_standard" else []

    return ExtractionDetailResponse(
        source_url=url,
        semantic_query=semantic_query,
        max_items=max_items,
        schema_preset=schema_preset,
        schema_fields=list(preset["fields"]),
        tinyfish_status=tinyfish_payload.get("status") if isinstance(tinyfish_payload, dict) else None,
        raw_product_count=len(raw_cards),
        schema_preview=schema_preview,
        normalized_products=normalized_products,
        raw_openai_items=openai_items,
        dropped_reasons=_build_dropped_reasons(schema_preview, schema_preset) if not normalized_products else [],
    )


def _extract_products(url: str, semantic_query: Optional[str]) -> list[ProductSchema]:
    return _extract_details(url, semantic_query, DEFAULT_MAX_ITEMS, "product_standard").normalized_products


def _build_showcase_prompt(payload: ShowcaseRequest) -> str:
    products = [
        {
            "product_name": product.product_name,
            "subtitle": product.subtitle,
            "current_price": product.current_price,
            "original_price": product.original_price,
            "product_url": product.product_url,
            "image_url": product.image_url,
        }
        for product in payload.products
    ]
    return (
        "Create a JSON object with the keys collection_title, collection_subtitle, visual_direction, and cards. "
        "cards must be an array. Each card must include product_name, subtitle, price_label, image_url, "
        "product_url, image_description, marketing_copy, and badge. Keep the tone concise and premium. "
        "Ground everything in the supplied products.\n\n"
        f"Source URL: {payload.source_url}\n"
        f"Semantic query: {payload.semantic_query or 'None'}\n"
        f"Products: {json.dumps(products, ensure_ascii=False)}"
    )


def _generate_showcase(payload: ShowcaseRequest) -> ShowcaseResponse:
    try:
        completion = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0.7,
            messages=[
                {"role": "system", "content": SHOWCASE_SYSTEM_PROMPT},
                {"role": "user", "content": _build_showcase_prompt(payload)},
            ],
        )
    except APITimeoutError as exc:
        logger.exception("OpenAI showcase request timed out")
        raise HTTPException(status_code=500, detail="OpenAI showcase generation timed out.") from exc
    except APIError as exc:
        logger.exception("OpenAI showcase request failed")
        raise HTTPException(status_code=500, detail=f"OpenAI showcase generation failed: {exc}") from exc
    except Exception as exc:
        logger.exception("Unexpected OpenAI showcase error")
        raise HTTPException(status_code=500, detail=f"Unexpected showcase generation error: {exc}") from exc

    content = ""
    if completion.choices:
        content = (completion.choices[0].message.content or "").strip()
    if not content:
        raise HTTPException(status_code=500, detail="OpenAI returned an empty showcase response.")

    try:
        parsed = _parse_json_object(content)
        return ShowcaseResponse(**parsed)
    except Exception as exc:
        logger.warning("Falling back to deterministic showcase: %s", exc)

    cards = [
        ShowcaseCard(
            product_name=product.product_name,
            subtitle=product.subtitle,
            price_label=_format_price(product.current_price, product.original_price),
            image_url=product.image_url,
            product_url=product.product_url,
            image_description=(
                f"Product image for {product.product_name}"
                + (f" with {product.subtitle}" if product.subtitle else "")
            ),
            marketing_copy=(
                f"A clean pick for {payload.semantic_query}."
                if payload.semantic_query
                else "A standout product extracted from the source collection."
            ),
            badge="Featured pick",
        )
        for product in payload.products[:6]
    ]
    return ShowcaseResponse(
        collection_title="Curated Product Spotlight",
        collection_subtitle="AI-generated storefront preview based on the extracted catalog.",
        visual_direction="Clean editorial cards with product-first imagery and clear price hierarchy.",
        cards=cards,
    )


def _get_homepage_html() -> str:
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Web2API</title>
  <style>
    :root {
      --bg: #f5efe3;
      --panel: rgba(255, 252, 246, 0.82);
      --panel-strong: rgba(255, 250, 240, 0.95);
      --ink: #16211d;
      --muted: #5d6a65;
      --accent: #c85d2f;
      --accent-deep: #7d2811;
      --line: rgba(22, 33, 29, 0.12);
      --shadow: 0 22px 60px rgba(76, 47, 21, 0.12);
    }
    * { box-sizing: border-box; }
    html { scroll-behavior: smooth; }
    body {
      margin: 0;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(200, 93, 47, 0.20), transparent 24rem),
        radial-gradient(circle at top right, rgba(55, 108, 93, 0.18), transparent 28rem),
        linear-gradient(180deg, #f8f3eb 0%, #f5efe3 45%, #efe6d8 100%);
      font-family: Georgia, "Times New Roman", serif;
    }
    .shell {
      width: min(1220px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 28px 0 64px;
    }
    .hero, .panel {
      backdrop-filter: blur(18px);
      background: var(--panel);
      border: 1px solid rgba(255, 255, 255, 0.65);
      box-shadow: var(--shadow);
    }
    .hero {
      position: relative;
      overflow: hidden;
      border-radius: 34px;
      padding: 32px;
      min-height: 320px;
      display: grid;
      grid-template-columns: 1.25fr 0.95fr;
      gap: 28px;
    }
    .hero::after {
      content: "";
      position: absolute;
      inset: auto -80px -80px auto;
      width: 280px;
      height: 280px;
      border-radius: 999px;
      background: radial-gradient(circle, rgba(200, 93, 47, 0.28), rgba(200, 93, 47, 0));
      pointer-events: none;
    }
    .eyebrow {
      letter-spacing: 0.12em;
      text-transform: uppercase;
      font-family: "Segoe UI", sans-serif;
      font-size: 0.75rem;
      color: var(--accent-deep);
      margin-bottom: 18px;
    }
    h1, h2, h3 { margin: 0; }
    h1 {
      font-size: clamp(2.6rem, 7vw, 5rem);
      line-height: 0.95;
      font-weight: 700;
      max-width: 9ch;
    }
    .hero-copy p, .stat p, .subtext, .table-note, .empty-state, .error-box, .success-box,
    input, button, .chip, .field label, .field-hint, .card p, table, .section-kicker,
    .visual-direction, .footer-note, .results-summary {
      font-family: "Segoe UI", sans-serif;
    }
    .hero-copy p {
      max-width: 58ch;
      color: var(--muted);
      margin: 18px 0 0;
      font-size: 1rem;
      line-height: 1.65;
    }
    .stats {
      align-self: end;
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }
    .stat {
      border-radius: 18px;
      padding: 18px;
      background: rgba(255, 255, 255, 0.62);
      border: 1px solid rgba(22, 33, 29, 0.08);
    }
    .stat strong {
      display: block;
      font-size: 1.8rem;
      margin-bottom: 6px;
    }
    .stat p {
      margin: 0;
      color: var(--muted);
      font-size: 0.92rem;
      line-height: 1.5;
    }
    .layout {
      display: grid;
      grid-template-columns: 1fr;
      gap: 22px;
      margin-top: 22px;
    }
    .panel {
      border-radius: 28px;
      padding: 24px;
    }
    .section-kicker {
      text-transform: uppercase;
      letter-spacing: 0.12em;
      color: var(--accent-deep);
      font-size: 0.72rem;
      margin-bottom: 10px;
      font-weight: 700;
    }
    .panel h2 {
      font-size: 1.8rem;
      margin-bottom: 8px;
    }
    .subtext {
      color: var(--muted);
      line-height: 1.6;
      margin: 0 0 20px;
    }
    .field {
      margin-bottom: 16px;
    }
    .field label {
      display: block;
      margin-bottom: 8px;
      font-size: 0.92rem;
      font-weight: 600;
    }
    .field-hint {
      display: block;
      color: var(--muted);
      margin-top: 8px;
      font-size: 0.82rem;
      line-height: 1.5;
    }
    input, select, textarea {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 15px 16px;
      background: rgba(255, 255, 255, 0.82);
      color: var(--ink);
      font-size: 0.96rem;
      outline: none;
      transition: border-color 160ms ease, transform 160ms ease, box-shadow 160ms ease;
    }
    input:focus, select:focus, textarea:focus {
      border-color: rgba(200, 93, 47, 0.65);
      box-shadow: 0 0 0 4px rgba(200, 93, 47, 0.12);
      transform: translateY(-1px);
    }
    .action-row {
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      margin-top: 20px;
    }
    button {
      border: 0;
      border-radius: 999px;
      padding: 14px 18px;
      font-weight: 700;
      font-size: 0.95rem;
      cursor: pointer;
      transition: transform 160ms ease, opacity 160ms ease, box-shadow 160ms ease;
    }
    button:hover { transform: translateY(-1px); }
    button:disabled { opacity: 0.55; cursor: not-allowed; transform: none; }
    .primary {
      background: linear-gradient(135deg, var(--accent) 0%, #ef8554 100%);
      color: #fffaf4;
      box-shadow: 0 14px 28px rgba(200, 93, 47, 0.28);
    }
    .secondary {
      background: rgba(22, 33, 29, 0.08);
      color: var(--ink);
    }
    .status-strip {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 18px;
      min-height: 32px;
    }
    .chip {
      padding: 8px 12px;
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.68);
      border: 1px solid rgba(22, 33, 29, 0.08);
      font-size: 0.82rem;
      color: var(--muted);
    }
    .right-stack {
      display: grid;
      gap: 22px;
    }
    .results-head, .showcase-header {
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      gap: 12px;
      align-items: end;
      margin-bottom: 16px;
    }
    .table-wrap {
      overflow: auto;
      border-radius: 18px;
      border: 1px solid rgba(22, 33, 29, 0.08);
      background: rgba(255, 255, 255, 0.68);
    }
    table {
      width: 100%;
      border-collapse: collapse;
      min-width: 900px;
      font-size: 0.92rem;
    }
    thead th {
      text-align: left;
      font-size: 0.76rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
      padding: 14px 16px;
      background: rgba(22, 33, 29, 0.04);
    }
    tbody td {
      padding: 14px 16px;
      border-top: 1px solid rgba(22, 33, 29, 0.08);
      vertical-align: top;
    }
    tbody tr:hover {
      background: rgba(255, 255, 255, 0.78);
    }
    .name-cell strong {
      display: block;
      margin-bottom: 4px;
      font-size: 1rem;
    }
    .tiny {
      color: var(--muted);
      font-size: 0.82rem;
      line-height: 1.45;
      word-break: break-word;
    }
    .price {
      font-weight: 700;
      white-space: nowrap;
    }
    .table-note, .visual-direction, .footer-note {
      color: var(--muted);
      line-height: 1.6;
      margin-top: 12px;
      font-size: 0.9rem;
    }
    .showcase-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
      gap: 18px;
    }
    .card {
      position: relative;
      overflow: hidden;
      border-radius: 22px;
      background: var(--panel-strong);
      border: 1px solid rgba(22, 33, 29, 0.08);
      box-shadow: 0 18px 44px rgba(76, 47, 21, 0.10);
    }
    .card-image {
      aspect-ratio: 4 / 4.4;
      background: linear-gradient(180deg, rgba(22, 33, 29, 0.06), rgba(22, 33, 29, 0.01));
      overflow: hidden;
    }
    .card-image img {
      width: 100%;
      height: 100%;
      object-fit: cover;
      display: block;
      transition: transform 220ms ease;
    }
    .card:hover .card-image img {
      transform: scale(1.04);
    }
    .card-body {
      padding: 16px;
    }
    .badge {
      display: inline-block;
      margin-bottom: 10px;
      padding: 6px 10px;
      border-radius: 999px;
      background: rgba(200, 93, 47, 0.12);
      color: var(--accent-deep);
      font-family: "Segoe UI", sans-serif;
      font-size: 0.76rem;
      font-weight: 700;
    }
    .card-title {
      font-size: 1.2rem;
      margin-bottom: 6px;
    }
    .card-subtitle, .card-copy, .card-alt {
      color: var(--muted);
      font-family: "Segoe UI", sans-serif;
      line-height: 1.55;
      font-size: 0.9rem;
    }
    .card-alt {
      margin-top: 12px;
      padding-top: 12px;
      border-top: 1px solid rgba(22, 33, 29, 0.08);
    }
    .card-footer {
      margin-top: 14px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      font-family: "Segoe UI", sans-serif;
    }
    .card-link {
      color: var(--accent-deep);
      text-decoration: none;
      font-weight: 700;
    }
    .empty-state, .error-box, .success-box {
      border-radius: 16px;
      padding: 16px;
      font-size: 0.92rem;
      line-height: 1.6;
    }
    .empty-state {
      background: rgba(255, 255, 255, 0.58);
      border: 1px dashed rgba(22, 33, 29, 0.18);
      color: var(--muted);
    }
    .error-box {
      background: rgba(125, 40, 17, 0.08);
      border: 1px solid rgba(125, 40, 17, 0.15);
      color: #692313;
      display: none;
    }
    .success-box {
      background: rgba(44, 111, 86, 0.10);
      border: 1px solid rgba(44, 111, 86, 0.16);
      color: #214f3f;
      display: none;
    }
    .hidden { display: none !important; }
    .code-block {
      margin: 12px 0 0;
      padding: 14px;
      border-radius: 16px;
      background: rgba(22, 33, 29, 0.88);
      color: #f9f5ee;
      font-family: Consolas, "Courier New", monospace;
      font-size: 0.82rem;
      line-height: 1.6;
      overflow: auto;
      white-space: pre-wrap;
      word-break: break-word;
    }
    .mini-grid {
      display: grid;
      gap: 12px;
      margin-top: 18px;
    }
    .status-card {
      border-radius: 16px;
      padding: 14px;
      background: rgba(255, 255, 255, 0.62);
      border: 1px solid rgba(22, 33, 29, 0.08);
    }
    .status-card strong {
      display: block;
      margin-bottom: 6px;
      font-family: "Segoe UI", sans-serif;
      font-size: 0.9rem;
    }
    .status-card span {
      color: var(--muted);
      font-family: "Segoe UI", sans-serif;
      font-size: 0.84rem;
      line-height: 1.5;
    }
    .placeholder-image {
      width: 100%;
      height: 100%;
      display: grid;
      place-items: center;
      background: linear-gradient(135deg, rgba(200, 93, 47, 0.18), rgba(22, 33, 29, 0.08));
      color: var(--accent-deep);
      font-family: "Segoe UI", sans-serif;
      font-size: 0.85rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }
    .analysis-grid {
      display: grid;
      grid-template-columns: 1.05fr 0.95fr;
      gap: 22px;
      margin-top: 22px;
    }
    .stack-list {
      display: grid;
      gap: 12px;
    }
    .stack-card, .field-card, .summary-item {
      border-radius: 16px;
      padding: 16px;
      background: rgba(255, 255, 255, 0.64);
      border: 1px solid rgba(22, 33, 29, 0.08);
      font-family: "Segoe UI", sans-serif;
    }
    .schema-grid, .summary-row, .sample-grid {
      display: grid;
      gap: 12px;
    }
    .schema-grid, .summary-row {
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    }
    .sample-grid {
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    }
    .summary-label {
      display: block;
      color: var(--muted);
      font-size: 0.76rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin-bottom: 8px;
    }
    .sample-card {
      border-radius: 18px;
      padding: 16px;
      background: rgba(255, 255, 255, 0.64);
      border: 1px solid rgba(22, 33, 29, 0.08);
      font-family: "Segoe UI", sans-serif;
    }
    .sample-meta {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 10px;
      color: var(--muted);
      font-size: 0.8rem;
    }
    .sample-meta span {
      padding: 6px 9px;
      border-radius: 999px;
      background: rgba(22, 33, 29, 0.06);
    }
    .view-toggle, .code-tabs, .preset-row {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin: 14px 0;
    }
    .view-button, .tab-button, .preset {
      border: 1px solid rgba(22, 33, 29, 0.1);
      border-radius: 999px;
      padding: 10px 14px;
      background: rgba(255, 255, 255, 0.72);
      color: var(--ink);
      font-family: "Segoe UI", sans-serif;
      font-weight: 600;
      cursor: pointer;
    }
    .view-button.active, .tab-button.active {
      background: rgba(200, 93, 47, 0.14);
      color: var(--accent-deep);
      border-color: rgba(200, 93, 47, 0.24);
    }
    .table-shell {
      overflow: auto;
      border: 1px solid rgba(22, 33, 29, 0.08);
      border-radius: 18px;
      background: rgba(255, 255, 255, 0.68);
    }
    .code-example {
      display: none;
      margin: 12px 0 0;
      padding: 14px;
      border-radius: 16px;
      background: rgba(22, 33, 29, 0.88);
      color: #f9f5ee;
      font-family: Consolas, "Courier New", monospace;
      font-size: 0.82rem;
      line-height: 1.6;
      white-space: pre-wrap;
      overflow: auto;
    }
    .code-example.active {
      display: block;
    }
    .raw-panel {
      margin-top: 14px;
      font-family: "Segoe UI", sans-serif;
    }
    .raw-panel summary {
      cursor: pointer;
      font-weight: 700;
    }
    @media (max-width: 1024px) {
      .analysis-grid { grid-template-columns: 1fr; }
    }
    @media (max-width: 1024px) {
      .hero, .layout { grid-template-columns: 1fr; }
      .stats { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
    @media (max-width: 640px) {
      .shell { width: min(100vw - 18px, 1220px); padding-top: 10px; }
      .hero, .panel { padding: 20px; border-radius: 24px; }
      .stats { grid-template-columns: 1fr; }
      .action-row, .showcase-header, .results-head {
        flex-direction: column;
        align-items: stretch;
      }
      button { width: 100%; }
    }
  </style>
</head>
<body>
  <main class="shell">
    <section class="hero">
      <div class="hero-copy">
        <div class="eyebrow">Autonomous Catalog Intelligence</div>
        <h1>Turn storefront chaos into usable product data.</h1>
        <p>
          Web2API extracts live product collections from the web, cleans them into a structured catalog,
          and can instantly turn the result into a presentable showcase concept for demos, internal tools,
          and quick merchandising reviews.
        </p>
      </div>
      <div class="stats">
        <div class="stat"><strong>15</strong><p>Product cards targeted from the visible grid on each extraction run.</p></div>
        <div class="stat"><strong>JSON</strong><p>Structured output shaped for APIs, dashboards, and downstream automation.</p></div>
        <div class="stat"><strong>AI</strong><p>Semantic filtering and showcase writing layered on top of raw browser extraction.</p></div>
        <div class="stat"><strong>2 views</strong><p>Database-style table for inspection and a visual storefront for demonstration.</p></div>
      </div>
    </section>
    <section class="layout">
      <div class="right-stack">
        <section class="panel">
          <div class="results-head">
            <div>
              <div class="section-kicker">Database View</div>
              <h2>Extracted product table</h2>
            </div>
            <div id="resultsSummary" class="results-summary">No extraction has been run yet.</div>
          </div>
          <div id="tableEmpty" class="empty-state">
            Your structured catalog will appear here after extraction. Each row includes product identity,
            pricing, and source links so you can inspect the data before generating a showcase.
          </div>
          <div id="tableWrap" class="table-wrap hidden"></div>
          <div class="table-note">This is your quick review surface before using the products anywhere else.</div>
        </section>
        <section class="panel">
          <div class="showcase-header">
            <div>
              <div class="section-kicker">Showcase View</div>
              <h2>UI-based product preview</h2>
            </div>
          </div>
          <div id="showcaseEmpty" class="empty-state">
            After extraction, generate a showcase to get a lightweight product presentation with image descriptions,
            short marketing copy, and a visual direction you can use in a hackathon demo.
          </div>
          <div id="showcaseRoot" class="hidden"></div>
        </section>
      </div>
    </section>
    <section class="analysis-grid">
      <section class="panel">
        <div class="section-kicker">Analysis Studio</div>
        <h2>Turn a website into a generated API</h2>
        <p class="subtext">
          This workflow comes from the imported zip logic. It live-fetches a site, infers its structure,
          stores an analysis in memory, and exposes generated API endpoints you can call immediately.
        </p>
        <div class="field">
          <label for="analysisUrlInput">Analysis URL</label>
          <input id="analysisUrlInput" type="url" placeholder="https://news.ycombinator.com" />
        </div>
        <div class="field">
          <label for="analysisGoalInput">Analysis goal</label>
          <textarea id="analysisGoalInput" rows="5" placeholder="Extract the top stories and infer a reusable schema for article data."></textarea>
        </div>
        <div class="preset-row">
          <button class="preset" data-url="https://news.ycombinator.com" data-goal="Extract the top stories and infer a reusable schema for article data.">News</button>
          <button class="preset" data-url="https://www.producthunt.com" data-goal="Infer a launch-listing schema and generate reusable endpoints.">Launches</button>
          <button class="preset" data-url="https://www.ycombinator.com/jobs" data-goal="Infer a job listing schema and propose a searchable jobs API.">Jobs</button>
        </div>
        <div class="action-row">
          <button id="extractBtn" class="secondary">Extract Products</button>
          <button id="showcaseBtn" class="secondary" disabled>Generate Showcase</button>
          <button id="analyzeBtn" class="primary">Analyze Website</button>
        </div>
        <div id="storefrontStatus" class="status-strip"><span class="chip">Storefront idle</span></div>
        <div id="errorBox" class="error-box"></div>
        <div id="successBox" class="success-box"></div>
        <div id="analysisStatus" class="status-strip"><span class="chip">Analysis idle</span></div>
        <div class="table-note">This is a separate workflow from TinyFish extraction. It performs direct HTML analysis and generated API design.</div>
      </section>
      <section class="right-stack">
        <section class="panel">
          <div class="results-head">
            <div>
              <div class="section-kicker">Analysis Summary</div>
              <h2>Detected structure</h2>
            </div>
          </div>
          <div id="analysisSummary" class="summary-row">
            <article class="summary-item"><span class="summary-label">Entity</span><strong>Waiting</strong></article>
            <article class="summary-item"><span class="summary-label">Confidence</span><strong>--</strong></article>
            <article class="summary-item"><span class="summary-label">Endpoints</span><strong>--</strong></article>
            <article class="summary-item"><span class="summary-label">Page classes</span><strong>--</strong></article>
          </div>
          <div class="analysis-grid" style="grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 16px;">
            <div>
              <h3>Detected page classes</h3>
              <div id="analysisSiteMap" class="stack-list">
                <div class="stack-card">No page classes yet. Run an analysis.</div>
              </div>
            </div>
            <div>
              <h3>Agent notes</h3>
              <div id="analysisNotes" class="stack-list">
                <div class="stack-card">Run an analysis to generate agent notes.</div>
              </div>
            </div>
          </div>
        </section>
        <section class="panel">
          <div class="results-head">
            <div>
              <div class="section-kicker">Schema Output</div>
              <h2>Inferred schema and samples</h2>
            </div>
          </div>
          <div id="analysisSchemaCards" class="schema-grid">
            <div class="field-card"><strong>Run an analysis</strong><span>Schema fields will appear here.</span></div>
          </div>
          <details class="raw-panel">
            <summary>View raw schema JSON</summary>
            <pre id="analysisSchemaOutput" class="code-block">Run an analysis to generate an inferred schema.</pre>
          </details>
          <div class="view-toggle">
            <button class="view-button active" data-analysis-view="analysisCardsView">Cards</button>
            <button class="view-button" data-analysis-view="analysisTableView">Table</button>
            <button class="view-button" data-analysis-view="analysisRawView">Raw JSON</button>
          </div>
          <div id="analysisCardsView" class="sample-grid">
            <div class="sample-card"><strong>Waiting for results</strong><p>Sample records will appear here in a readable format.</p></div>
          </div>
          <div id="analysisTableView" class="table-shell hidden">
            <table>
              <thead id="analysisTableHead"></thead>
              <tbody id="analysisTableBody"></tbody>
            </table>
          </div>
          <pre id="analysisRawView" class="code-block hidden">Structured results will appear here.</pre>
        </section>
        <section class="panel">
          <div class="results-head">
            <div>
              <div class="section-kicker">Generated API</div>
              <h2>Use the result immediately</h2>
            </div>
          </div>
          <pre id="analysisApiOutput" class="code-block">Generated endpoints will appear here.</pre>
          <div class="code-tabs">
            <button class="tab-button active" data-target="analysisCurlExample">cURL</button>
            <button class="tab-button" data-target="analysisPythonExample">Python</button>
            <button class="tab-button" data-target="analysisJsExample">JavaScript</button>
          </div>
          <pre id="analysisCurlExample" class="code-example active">Run an analysis to generate integration examples.</pre>
          <pre id="analysisPythonExample" class="code-example">Run an analysis to generate integration examples.</pre>
          <pre id="analysisJsExample" class="code-example">Run an analysis to generate integration examples.</pre>
        </section>
      </section>
    </section>
    <p class="footer-note">API docs remain available at <a href="/docs">/docs</a> if you need direct endpoint testing.</p>
  </main>
  <script>
    const state = {
      products: [],
      preview: [],
      sourceUrl: "",
      semanticQuery: "",
      showcase: null
    };
    const resultsSummary = document.getElementById("resultsSummary");
    const tableEmpty = document.getElementById("tableEmpty");
    const tableWrap = document.getElementById("tableWrap");
    const showcaseEmpty = document.getElementById("showcaseEmpty");
    const showcaseRoot = document.getElementById("showcaseRoot");
    const analysisUrlInput = document.getElementById("analysisUrlInput");
    const extractBtn = document.getElementById("extractBtn");
    const showcaseBtn = document.getElementById("showcaseBtn");
    const storefrontStatus = document.getElementById("storefrontStatus");
    const errorBox = document.getElementById("errorBox");
    const successBox = document.getElementById("successBox");
    const analysisGoalInput = document.getElementById("analysisGoalInput");
    const analyzeBtn = document.getElementById("analyzeBtn");
    const analysisStatus = document.getElementById("analysisStatus");
    const analysisSummary = document.getElementById("analysisSummary");
    const analysisSiteMap = document.getElementById("analysisSiteMap");
    const analysisNotes = document.getElementById("analysisNotes");
    const analysisSchemaCards = document.getElementById("analysisSchemaCards");
    const analysisSchemaOutput = document.getElementById("analysisSchemaOutput");
    const analysisCardsView = document.getElementById("analysisCardsView");
    const analysisTableView = document.getElementById("analysisTableView");
    const analysisTableHead = document.getElementById("analysisTableHead");
    const analysisTableBody = document.getElementById("analysisTableBody");
    const analysisRawView = document.getElementById("analysisRawView");
    const analysisApiOutput = document.getElementById("analysisApiOutput");
    const analysisCurlExample = document.getElementById("analysisCurlExample");
    const analysisPythonExample = document.getElementById("analysisPythonExample");
    const analysisJsExample = document.getElementById("analysisJsExample");

    function escapeHtml(value) {
      return String(value ?? "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
    }
    function setStorefrontStatus(items) {
      storefrontStatus.innerHTML = items.map((item) => `<span class="chip">${item}</span>`).join("");
    }
    function setError(message) {
      errorBox.style.display = "block";
      errorBox.textContent = message;
    }
    function clearError() {
      errorBox.style.display = "none";
      errorBox.textContent = "";
    }
    function setSuccess(message) {
      successBox.style.display = "block";
      successBox.textContent = message;
    }
    function clearSuccess() {
      successBox.style.display = "none";
      successBox.textContent = "";
    }
    function formatCell(field, value) {
      if (value == null || value === "") return "null";
      if (field.includes("url")) {
        return `<a href="${escapeHtml(value)}" target="_blank" rel="noreferrer">Open</a>`;
      }
      if (typeof value === "number") {
        return Number(value).toFixed(2);
      }
      return escapeHtml(value);
    }
    function renderTable(products, preview = [], droppedReasons = [], schemaFields = []) {
      const rowsSource = products.length ? products : preview;
      const isPreview = !products.length && preview.length > 0;
      if (!rowsSource.length) {
        tableEmpty.classList.remove("hidden");
        tableWrap.classList.add("hidden");
        tableWrap.innerHTML = "";
        resultsSummary.textContent = "Extraction completed with 0 usable products.";
        return;
      }
      tableEmpty.classList.add("hidden");
      tableWrap.classList.remove("hidden");
      resultsSummary.textContent = isPreview
        ? `TinyFish found ${preview.length} preview rows, but they are still preview data.`
        : `${products.length} normalized products extracted from ${state.sourceUrl}`;
      const activeFields = schemaFields.length ? schemaFields : Object.keys(rowsSource[0] || {});
      const headers = activeFields.map((field) => `<th>${escapeHtml(field)}</th>`).join("");
      const rows = rowsSource.map((product, index) => `
        <tr>
          <td>${index + 1}</td>
          ${activeFields.map((field) => `<td class="tiny">${formatCell(field, product[field])}</td>`).join("")}
        </tr>
      `).join("");
      const reasonBlock = droppedReasons.length
        ? `<div class="table-note"><strong>Why normalization failed:</strong> ${escapeHtml(droppedReasons[0])}</div>`
        : "";
      tableWrap.innerHTML = `
        <table>
          <thead>
            <tr>
              <th>#</th>
              ${headers}
            </tr>
          </thead>
          <tbody>${rows}</tbody>
        </table>
        ${reasonBlock}
      `;
    }
    function renderPreviewShowcase(rows) {
      if (!rows || !rows.length) {
        renderShowcase(null);
        return;
      }
      const cards = rows.slice(0, 8).map((row) => ({
        product_name: row.product_name || row.name || "Untitled product",
        subtitle: row.subtitle || row.category || "Schema preview row",
        price_label: row.price_label || (row.current_price != null ? `$${Number(row.current_price).toFixed(2)}` : "Price unavailable"),
        badge: row.badge || row.price_status || "Preview",
        image_url: row.image_url || null,
        product_url: row.product_url || null,
        image_description: row.image_description || `Preview for ${row.product_name || row.name || "product"}`,
        marketing_copy: "Preview card generated from schema output. Missing fields are surfaced instead of hidden."
      }));
      renderShowcase({
        collection_title: "Preview Showcase",
        collection_subtitle: "Immediate UI preview built from extracted schema rows.",
        visual_direction: "Fast fallback showcase using available extraction data.",
        cards
      });
    }
    function syncStorefrontFromAnalysis(result) {
      const previewRows = (result.samples || []).map((item) => ({
        product_name: item.title || "Untitled",
        subtitle: item.summary || item.author || item.location || result.site?.entity_guess || "Analyzed record",
        current_price: item.price_text ? Number(String(item.price_text).replace(/[^0-9.]/g, "")) || null : null,
        original_price: item.price_text ? Number(String(item.price_text).replace(/[^0-9.]/g, "")) || null : null,
        product_url: item.url || null,
        image_url: item.image_url || null,
      }));
      state.products = [];
      state.preview = previewRows;
      state.sourceUrl = result.site?.url || analysisUrlInput.value.trim();
      state.semanticQuery = "";
      renderTable([], previewRows, [], [
        "product_name",
        "subtitle",
        "current_price",
        "product_url",
        "image_url",
      ]);
      renderPreviewShowcase(previewRows);
      setStorefrontStatus([`${previewRows.length} preview rows`, "Synced from analysis"]);
    }
    function renderShowcase(showcase) {
      if (!showcase || !showcase.cards || !showcase.cards.length) {
        showcaseEmpty.classList.remove("hidden");
        showcaseRoot.classList.add("hidden");
        showcaseRoot.innerHTML = "";
        return;
      }
      showcaseEmpty.classList.add("hidden");
      showcaseRoot.classList.remove("hidden");
      const cards = showcase.cards.map((card) => `
        <article class="card">
          <div class="card-image">
            ${card.image_url
              ? `<img src="${escapeHtml(card.image_url)}" alt="${escapeHtml(card.image_description)}" />`
              : `<div class="placeholder-image">No Image URL</div>`}
          </div>
          <div class="card-body">
            <span class="badge">${escapeHtml(card.badge)}</span>
            <h3 class="card-title">${escapeHtml(card.product_name)}</h3>
            <div class="card-subtitle">${escapeHtml(card.subtitle || "Curated product highlight")}</div>
            <p class="card-copy">${escapeHtml(card.marketing_copy)}</p>
            <div class="card-alt"><strong>Image description:</strong> ${escapeHtml(card.image_description)}</div>
            <div class="card-footer">
              <strong>${escapeHtml(card.price_label)}</strong>
              ${card.product_url
                ? `<a class="card-link" href="${escapeHtml(card.product_url)}" target="_blank" rel="noreferrer">Visit</a>`
                : `<span class="tiny">No product URL</span>`}
            </div>
          </div>
        </article>
      `).join("");
      showcaseRoot.innerHTML = `
        <div>
          <h3>${escapeHtml(showcase.collection_title)}</h3>
          <p class="visual-direction">${escapeHtml(showcase.collection_subtitle)}</p>
          <p class="visual-direction"><strong>Visual direction:</strong> ${escapeHtml(showcase.visual_direction)}</p>
        </div>
        <div class="showcase-grid">${cards}</div>
      `;
    }
    function setAnalysisStatus(items) {
      analysisStatus.innerHTML = items.map((item) => `<span class="chip">${item}</span>`).join("");
    }
    function renderAnalysisSummary(result) {
      analysisSummary.innerHTML = `
        <article class="summary-item"><span class="summary-label">Entity</span><strong>${escapeHtml(result.site.entity_guess)}</strong></article>
        <article class="summary-item"><span class="summary-label">Confidence</span><strong>${Math.round(result.site.confidence * 100)}%</strong></article>
        <article class="summary-item"><span class="summary-label">Endpoints</span><strong>${result.generated_api.endpoints.length}</strong></article>
        <article class="summary-item"><span class="summary-label">Page classes</span><strong>${result.site_map.length}</strong></article>
      `;
    }
    function renderAnalysisSiteMap(result) {
      analysisSiteMap.innerHTML = result.site_map.map((page) => `
        <div class="stack-card">
          <strong>${escapeHtml(page.page_type)}</strong>
          <div>${escapeHtml(page.notes)}</div>
          <div class="sample-meta"><span>${escapeHtml(page.pattern)}</span></div>
        </div>
      `).join("");
    }
    function renderAnalysisSchema(result) {
      analysisSchemaCards.innerHTML = result.schema.fields.map((field) => `
        <div class="field-card">
          <strong>${escapeHtml(field.name)}</strong>
          <span>${escapeHtml(field.type)}</span>
        </div>
      `).join("");
      analysisSchemaOutput.textContent = JSON.stringify(result.schema, null, 2);
    }
    function renderAnalysisSamples(result) {
      analysisCardsView.innerHTML = result.samples.slice(0, 4).map((item) => {
        const meta = [];
        if (item.author) meta.push(`<span>${escapeHtml(item.author)}</span>`);
        if (item.published_at) meta.push(`<span>${escapeHtml(item.published_at)}</span>`);
        if (item.location) meta.push(`<span>${escapeHtml(item.location)}</span>`);
        if (item.price_text) meta.push(`<span>${escapeHtml(item.price_text)}</span>`);
        return `
          <article class="sample-card">
            <strong>${escapeHtml(item.title || "Untitled")}</strong>
            <p>${escapeHtml(item.summary || item.url || "No summary available.")}</p>
            <div class="sample-meta">${meta.join("")}</div>
          </article>
        `;
      }).join("");
      analysisRawView.textContent = JSON.stringify(result.samples, null, 2);
      const columns = [...new Set(result.samples.flatMap((item) => Object.keys(item)))];
      analysisTableHead.innerHTML = `<tr>${columns.map((column) => `<th>${escapeHtml(column)}</th>`).join("")}</tr>`;
      analysisTableBody.innerHTML = result.samples.slice(0, 8).map((item) => `
        <tr>
          ${columns.map((column) => {
            const value = item[column] ?? "";
            if (column === "url" && value) {
              return `<td><a href="${escapeHtml(value)}" target="_blank" rel="noreferrer">Open Link</a></td>`;
            }
            return `<td>${escapeHtml(String(value))}</td>`;
          }).join("")}
        </tr>
      `).join("");
    }
    function renderAnalysisNotes(result) {
      analysisNotes.innerHTML = result.agent_notes.map((note) => `<div class="stack-card">${escapeHtml(note)}</div>`).join("");
    }
    function renderAnalysisQuickstart(result) {
      analysisCurlExample.textContent = result.quickstart.curl;
      analysisPythonExample.textContent = result.quickstart.python;
      analysisJsExample.textContent = result.quickstart.javascript;
      analysisApiOutput.textContent = JSON.stringify(result.generated_api, null, 2);
    }
    function renderAnalysisFailure(message) {
      analysisSummary.innerHTML = `
        <article class="summary-item"><span class="summary-label">Entity</span><strong>Error</strong></article>
        <article class="summary-item"><span class="summary-label">Confidence</span><strong>--</strong></article>
        <article class="summary-item"><span class="summary-label">Endpoints</span><strong>--</strong></article>
        <article class="summary-item"><span class="summary-label">Page classes</span><strong>--</strong></article>
      `;
      analysisSiteMap.innerHTML = `<div class="stack-card">No site map available because the analysis request failed.</div>`;
      analysisNotes.innerHTML = `<div class="stack-card">Unable to complete the reverse-engineering workflow.</div>`;
      analysisSchemaCards.innerHTML = `<div class="field-card"><strong>No schema</strong><span>The analysis did not complete.</span></div>`;
      analysisSchemaOutput.textContent = "The analysis request failed.";
      analysisCardsView.innerHTML = `<div class="sample-card"><strong>No sample data</strong><p>The analysis did not complete.</p></div>`;
      analysisTableHead.innerHTML = "";
      analysisTableBody.innerHTML = "";
      analysisRawView.textContent = message;
      analysisApiOutput.textContent = "Please retry after checking the target website.";
      analysisCurlExample.textContent = "Run an analysis to generate integration examples.";
      analysisPythonExample.textContent = "Run an analysis to generate integration examples.";
      analysisJsExample.textContent = "Run an analysis to generate integration examples.";
    }
    async function extractProducts() {
      const url = analysisUrlInput.value.trim();
      const semanticQuery = null;
      const schemaPreset = "product_standard";
      const maxItems = 15;
      clearError();
      clearSuccess();
      setStorefrontStatus(["Calling TinyFish", "Preset product_standard", `Targeting ${maxItems} items`]);
      extractBtn.disabled = true;
      showcaseBtn.disabled = true;
      try {
        const params = new URLSearchParams({ url });
        if (semanticQuery) params.set("semantic_query", semanticQuery);
        params.set("schema_preset", schemaPreset);
        params.set("max_items", String(maxItems));
        const response = await fetch(`/api/extract-detailed?${params.toString()}`);
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "Extraction failed.");
        state.products = data.normalized_products || [];
        state.preview = data.schema_preview || [];
        state.sourceUrl = url;
        state.semanticQuery = semanticQuery;
        state.showcase = null;
        renderTable(state.products, state.preview, data.dropped_reasons || [], data.schema_fields || []);
        if (state.products.length > 0) {
          renderShowcase(null);
        } else {
          renderPreviewShowcase(state.preview);
        }
        showcaseBtn.disabled = state.products.length === 0;
        if (state.products.length === 0) {
          const previewCount = state.preview.length;
          setSuccess(`Extraction completed. ${previewCount} preview rows are available. Strict normalization failed because some required fields are missing, but a fallback UI preview is rendered below.`);
          setStorefrontStatus(["Extraction complete", `${previewCount} preview rows`, `Limit ${maxItems}`]);
        } else {
          setSuccess(`Extraction completed with ${state.products.length} normalized products.`);
          setStorefrontStatus(["Extraction complete", `${state.products.length} products ready`, `Limit ${maxItems}`]);
        }
      } catch (error) {
        renderTable([], []);
        renderShowcase(null);
        setError(error.message || "Something went wrong during extraction.");
        setStorefrontStatus(["Extraction failed"]);
      } finally {
        extractBtn.disabled = false;
      }
    }
    async function generateShowcase() {
      clearError();
      clearSuccess();
      setStorefrontStatus(["Writing showcase", "Generating product copy"]);
      showcaseBtn.disabled = true;
      try {
        const response = await fetch("/api/showcase", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            source_url: state.sourceUrl,
            semantic_query: state.semanticQuery || null,
            products: state.products,
          }),
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "Showcase generation failed.");
        state.showcase = data;
        renderShowcase(data);
        setSuccess("Showcase generated. Scroll down to review the visual preview.");
        setStorefrontStatus(["Showcase ready", `${data.cards.length} featured cards`]);
      } catch (error) {
        setError(error.message || "Something went wrong during showcase generation.");
        setStorefrontStatus(["Showcase failed"]);
      } finally {
        showcaseBtn.disabled = state.products.length === 0;
      }
    }
    async function analyzeWebsite() {
      const payload = {
        url: analysisUrlInput.value.trim(),
        goal: analysisGoalInput.value.trim(),
      };
      if (!payload.url || !payload.goal) {
        setAnalysisStatus(["Missing input"]);
        return;
      }
      analyzeBtn.disabled = true;
      analyzeBtn.textContent = "Analyzing...";
      setAnalysisStatus(["Fetching page", "Inferring schema", "Generating API"]);
      renderAnalysisFailure("Collecting live analysis...");
      try {
        const response = await fetch("/api/analyze", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        const result = await response.json();
        if (!response.ok) throw new Error(result.detail || "Analysis failed.");
        renderAnalysisSummary(result);
        renderAnalysisSiteMap(result);
        renderAnalysisSchema(result);
        renderAnalysisSamples(result);
        renderAnalysisNotes(result);
        renderAnalysisQuickstart(result);
        syncStorefrontFromAnalysis(result);
        setAnalysisStatus([`${result.mode} mode`, result.site.entity_guess, `${result.generated_api.endpoints.length} endpoints`]);
      } catch (error) {
        renderAnalysisFailure(error.message || "Analysis failed.");
        setAnalysisStatus(["Analysis failed"]);
      } finally {
        analyzeBtn.disabled = false;
        analyzeBtn.textContent = "Analyze Website";
      }
    }
    extractBtn.addEventListener("click", extractProducts);
    showcaseBtn.addEventListener("click", generateShowcase);
    analyzeBtn.addEventListener("click", analyzeWebsite);
    document.querySelectorAll(".preset").forEach((button) => {
      button.addEventListener("click", () => {
        analysisUrlInput.value = button.dataset.url || "";
        analysisGoalInput.value = button.dataset.goal || "";
      });
    });
    document.querySelectorAll(".tab-button").forEach((button) => {
      button.addEventListener("click", () => {
        document.querySelectorAll(".tab-button").forEach((item) => item.classList.remove("active"));
        document.querySelectorAll(".code-example").forEach((item) => item.classList.remove("active"));
        button.classList.add("active");
        document.getElementById(button.dataset.target).classList.add("active");
      });
    });
    document.querySelectorAll("[data-analysis-view]").forEach((button) => {
      button.addEventListener("click", () => {
        document.querySelectorAll("[data-analysis-view]").forEach((item) => item.classList.remove("active"));
        button.classList.add("active");
        analysisCardsView.classList.add("hidden");
        analysisTableView.classList.add("hidden");
        analysisRawView.classList.add("hidden");
        document.getElementById(button.dataset.analysisView).classList.remove("hidden");
      });
    });
    analysisUrlInput.value = "https://news.ycombinator.com";
    analysisGoalInput.value = "Extract the top stories and infer a reusable schema for article data.";
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def home() -> HTMLResponse:
    return HTMLResponse(_get_homepage_html())


@app.get("/api/extract", response_model=list[ProductSchema], tags=["Extraction"])
def extract(
    url: str = Query(..., description="Target webpage URL to extract from"),
    semantic_query: Optional[str] = Query(
        default=None,
        description="Optional semantic filter, for example: white retro shoes",
    ),
) -> list[ProductSchema]:
    return _extract_products(url, semantic_query)


@app.get("/api/extract-detailed", response_model=ExtractionDetailResponse, tags=["Extraction"])
def extract_detailed(
    url: str = Query(..., description="Target webpage URL to extract from"),
    semantic_query: Optional[str] = Query(
        default=None,
        description="Optional semantic filter, for example: white retro shoes",
    ),
    schema_preset: str = Query(
        default="product_standard",
        description="Named schema preset used to shape the extracted output",
    ),
    max_items: int = Query(
        default=DEFAULT_MAX_ITEMS,
        ge=1,
        le=40,
        description="Maximum number of product cards TinyFish should attempt to extract",
    ),
) -> ExtractionDetailResponse:
    return _extract_details(url, semantic_query, max_items, schema_preset)


@app.get("/api/schema-presets", tags=["Schema"])
def list_schema_presets() -> dict[str, Any]:
    return {
        "default": "product_standard",
        "presets": SCHEMA_PRESETS,
    }


@app.post("/api/analyze", tags=["Analysis"])
def analyze(request: Request, payload: AnalyzeRequest = Body(...)) -> dict[str, Any]:
    try:
        return analyze_website(str(payload.url), payload.goal, str(request.base_url).rstrip("/"))
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Failed to fetch target website: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/generated/items", tags=["Analysis"])
def generated_items(
    analysis_id: str = Query(...),
    limit: int = Query(10, ge=1, le=50),
) -> dict[str, Any]:
    analysis = get_saved_analysis(analysis_id)
    return {
        "analysis_id": analysis_id,
        "entity": analysis["schema"]["entity"],
        "count": min(limit, len(analysis["samples"])),
        "items": analysis["samples"][:limit],
    }


@app.get("/api/generated/items/{item_id}", tags=["Analysis"])
def generated_item(item_id: int, analysis_id: str = Query(...)) -> dict[str, Any]:
    analysis = get_saved_analysis(analysis_id)
    for item in analysis["samples"]:
        if item.get("id") == item_id:
            return item
    raise HTTPException(status_code=404, detail="Item not found for this analysis.")


@app.get("/api/generated/schema", tags=["Analysis"])
def generated_schema(analysis_id: str = Query(...)) -> dict[str, Any]:
    analysis = get_saved_analysis(analysis_id)
    return {
        "analysis_id": analysis_id,
        "schema": analysis["schema"],
        "site_map": analysis["site_map"],
    }


@app.post("/api/showcase", response_model=ShowcaseResponse, tags=["Showcase"])
def create_showcase(payload: ShowcaseRequest = Body(...)) -> ShowcaseResponse:
    return _generate_showcase(payload)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
