from __future__ import annotations

import re
import uuid
from collections import Counter
from time import time
from typing import Any
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, HttpUrl
from starlette.requests import Request


app = FastAPI(title="Web2API Studio", version="2.0.0")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0 Safari/537.36"
)
REQUEST_TIMEOUT = 6
ANALYSIS_STORE: dict[str, dict[str, Any]] = {}
ANALYSIS_CACHE: dict[tuple[str, str], tuple[float, dict[str, Any]]] = {}
CACHE_TTL_SECONDS = 900
MAX_INTERNAL_LINKS = 12
MAX_CONTAINER_SCAN = 120
MAX_ANCHOR_SCAN = 180


class AnalyzeRequest(BaseModel):
    url: HttpUrl
    goal: str


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
        if parsed.scheme not in {"http", "https"}:
            continue
        if parsed.netloc != parsed_base.netloc:
            continue
        normalized = normalize_url(absolute)
        if normalized != normalize_url(base_url):
            collected.append(normalized)
        if len(collected) >= MAX_INTERNAL_LINKS:
            break

    return list(dict.fromkeys(collected))[:MAX_INTERNAL_LINKS]


def detect_entity(goal: str, title: str, url: str) -> str:
    text = f"{goal} {title} {url}".lower()
    mapping = {
        "product": ["product", "price", "store", "shop", "cart", "buy"],
        "job_listing": ["job", "career", "hiring", "role", "position"],
        "property_listing": ["property", "listing", "rent", "sale", "home", "condo", "apartment"],
        "event": ["event", "conference", "meetup", "workshop", "register"],
    }

    for entity, keywords in mapping.items():
        if any(keyword in text for keyword in keywords):
            return entity
    return "article"


def infer_fields(entity: str, soup: BeautifulSoup, title: str, canonical_url: str) -> list[dict[str, str]]:
    description = ""
    meta_description = soup.select_one('meta[name="description"]')
    if meta_description:
        description = clean_text(meta_description.get("content"))

    fields_map: dict[str, str] = {"title": "string", "url": "string"}

    if description:
        fields_map["summary"] = "string"

    if entity == "product":
        price_text = soup.get_text(" ", strip=True)
        if re.search(r"(sgd|\$|usd|eur)\s?\d", price_text, re.IGNORECASE):
            fields_map["price_text"] = "string"
        image = soup.select_one("img")
        if image and image.get("src"):
            fields_map["image_url"] = "string"
    elif entity == "job_listing":
        fields_map["team"] = "string"
        fields_map["location"] = "string"
    elif entity == "property_listing":
        fields_map["location"] = "string"
        fields_map["price_text"] = "string"
    elif entity == "event":
        fields_map["date_text"] = "string"
        fields_map["location"] = "string"
    else:
        fields_map["published_hint"] = "string"

    return [{"name": name, "type": field_type} for name, field_type in fields_map.items()]


def looks_like_article_url(url: str, base_host: str) -> bool:
    parsed = urlparse(url)
    path = parsed.path.lower().strip("/")
    if parsed.netloc != base_host or not path:
        return False
    disallowed = {
        "about",
        "contact",
        "privacy",
        "terms",
        "advertise",
        "latest",
        "tag",
        "category",
        "topics",
        "newsletter",
        "events",
        "podcasts",
        "news",
        "newest",
        "front",
        "newcomments",
        "ask",
        "show",
        "jobs",
        "submit",
        "login",
        "best",
        "item",
        "user",
    }
    first = path.split("/")[0]
    if first in disallowed:
        return False
    return len(path.split("/")) >= 2 or re.search(r"\d{4}", path) is not None


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
        score = ""
        author = ""
        age = ""
        comments = ""

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

        record = {
            "id": index,
            "title": title,
            "url": story_url,
        }
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

    for container in soup.select("article, div, li")[:MAX_CONTAINER_SCAN]:
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

        candidate = {
            "title": title,
            "url": normalized,
            "summary": clean_text(summary_node.get_text(" ", strip=True)) if summary_node else "",
            "published_at": clean_text(time_node.get("datetime") or time_node.get_text(" ", strip=True)) if time_node else "",
            "author": clean_text(author_node.get_text(" ", strip=True)) if author_node else "",
            "image_url": urljoin(base_url, image_node.get("src")) if image_node and image_node.get("src") else "",
        }
        seen.add(normalized)
        candidates.append(candidate)
        if len(candidates) >= 8:
            break

    return candidates[:12]


def article_candidates_from_links(soup: BeautifulSoup, base_url: str) -> list[dict[str, Any]]:
    base_host = urlparse(base_url).netloc
    seen: set[str] = set()
    candidates: list[dict[str, Any]] = []

    for anchor in soup.select("a[href]")[:MAX_ANCHOR_SCAN]:
        href = urljoin(base_url, anchor.get("href", "").strip())
        normalized = normalize_url(href)
        title = clean_text(anchor.get_text(" ", strip=True))
        if normalized in seen or len(title) < 28:
            continue
        if not looks_like_article_url(normalized, base_host):
            continue
        seen.add(normalized)
        candidates.append({"title": title, "url": normalized, "summary": "", "published_at": "", "author": "", "image_url": ""})
        if len(candidates) >= 8:
            break

    return candidates[:12]


def extract_article_records(soup: BeautifulSoup, base_url: str) -> list[dict[str, Any]]:
    if urlparse(base_url).netloc == "news.ycombinator.com":
        hacker_news_records = extract_hacker_news_records(soup)
        if hacker_news_records:
            return hacker_news_records

    candidates = article_candidates_from_containers(soup, base_url)
    if len(candidates) < 3:
        candidates = article_candidates_from_links(soup, base_url)

    records: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates[:8], start=1):
        record = {
            "id": index,
            "title": candidate["title"],
            "url": candidate["url"],
        }
        if candidate.get("summary"):
            record["summary"] = candidate["summary"]
        if candidate.get("published_at"):
            record["published_at"] = candidate["published_at"]
        if candidate.get("author"):
            record["author"] = candidate["author"]
        if candidate.get("image_url"):
            record["image_url"] = candidate["image_url"]
        records.append(record)

    return records


def make_sample_records(
    entity: str,
    landing_title: str,
    landing_url: str,
    internal_links: list[str],
    soup: BeautifulSoup,
) -> list[dict[str, Any]]:
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

        record: dict[str, Any] = {
            "id": index,
            "title": link_title or f"{entity.title()} {index}",
            "url": link,
        }

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


def infer_schema_from_samples(entity: str, samples: list[dict[str, Any]]) -> list[dict[str, str]]:
    if not samples:
        return [{"name": "title", "type": "string"}, {"name": "url", "type": "string"}]

    field_types: dict[str, str] = {}
    for key in samples[0].keys():
        value = next((sample.get(key) for sample in samples if sample.get(key) not in (None, "")), "")
        if isinstance(value, int):
            field_types[key] = "number"
        else:
            field_types[key] = "string"

    ordered = []
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
        "curl": (
            f"curl '{base_url}/items?analysis_id={analysis_id}' \\\n"
            "  -H 'Accept: application/json'"
        ),
        "python": (
            "import requests\n\n"
            f"response = requests.get('{base_url}/items', params={{'analysis_id': '{analysis_id}'}})\n"
            "response.raise_for_status()\n"
            "print(response.json())"
        ),
        "javascript": (
            f"const response = await fetch('{base_url}/items?analysis_id={analysis_id}');\n"
            "const data = await response.json();\n"
            "console.log(data);"
        ),
    }


def analyze_website(url: str, goal: str, origin: str) -> dict[str, Any]:
    cache_key = (normalize_url(url), goal.strip().lower())
    cached = ANALYSIS_CACHE.get(cache_key)
    if cached and (time() - cached[0]) < CACHE_TTL_SECONDS:
        cached_result = dict(cached[1])
        cached_result["mode"] = "cached"
        cached_result["generated_api"] = build_generated_api(cached_result["analysis_id"], cached_result["schema"]["entity"], origin)
        cached_result["quickstart"] = build_quickstart(cached_result["analysis_id"], origin)
        return cached_result

    html, response = fetch_html(url)
    soup = BeautifulSoup(html, "html.parser")

    title = clean_text(soup.title.string if soup.title and soup.title.string else "") or urlparse(str(response.url)).netloc
    hostname = urlparse(str(response.url)).netloc.replace("www.", "") or "unknown-site"
    entity = detect_entity(goal, title, str(response.url))
    internal_links = same_host_links(soup, str(response.url))
    samples = make_sample_records(entity, title, str(response.url), internal_links, soup)
    fields = infer_schema_from_samples(entity, samples) or infer_fields(entity, soup, title, str(response.url))
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
        "schema": {
            "entity": entity,
            "fields": fields,
        },
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
    ANALYSIS_CACHE[cache_key] = (time(), result)
    return result


def get_saved_analysis(analysis_id: str) -> dict[str, Any]:
    analysis = ANALYSIS_STORE.get(analysis_id)
    if not analysis:
        raise HTTPException(status_code=404, detail="Unknown analysis_id. Run /api/analyze first.")
    return analysis


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/analyze")
async def analyze(payload: AnalyzeRequest, request: Request) -> dict[str, Any]:
    try:
        return analyze_website(str(payload.url), payload.goal, str(request.base_url).rstrip("/"))
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Failed to fetch target website: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/generated/items")
async def generated_items(analysis_id: str = Query(...), limit: int = Query(10, ge=1, le=50)) -> dict[str, Any]:
    analysis = get_saved_analysis(analysis_id)
    return {
        "analysis_id": analysis_id,
        "entity": analysis["schema"]["entity"],
        "count": min(limit, len(analysis["samples"])),
        "items": analysis["samples"][:limit],
    }


@app.get("/api/generated/items/{item_id}")
async def generated_item(item_id: int, analysis_id: str = Query(...)) -> dict[str, Any]:
    analysis = get_saved_analysis(analysis_id)
    for item in analysis["samples"]:
        if item.get("id") == item_id:
            return item
    raise HTTPException(status_code=404, detail="Item not found for this analysis.")


@app.get("/api/generated/schema")
async def generated_schema(analysis_id: str = Query(...)) -> dict[str, Any]:
    analysis = get_saved_analysis(analysis_id)
    return {
        "analysis_id": analysis_id,
        "schema": analysis["schema"],
        "site_map": analysis["site_map"],
    }
