# TinyFishNUS-ByteMe

Web2API Studio is a hackathon project that turns a public website into a cleaner, API-like data surface.

The product is designed to help a developer or non-technical judge understand a website as structured data instead of raw HTML. A user pastes a URL, describes the extraction goal in plain English, and the app:

- fetches the target page live
- identifies likely content patterns
- infers a schema from extracted records
- shows the output in multiple formats
- generates API-style endpoints for the extracted result

## Official Submission Branch

The official submission branch for this project is:

[`submission-final`](https://github.com/roboticspro1/TinyFishNUS-ByteMe/tree/submission-final)

`main` may not always reflect the intended submission snapshot, so judges and reviewers should use `submission-final`.

## What The Product Does

Web2API Studio presents a top-down workflow:

1. Enter a target website and extraction goal.
2. Let the system analyze the page structure.
3. Review detected page classes and inferred schema.
4. Inspect extracted records as cards, table rows, or raw JSON.
5. View generated API endpoints and example code.

This makes the output accessible for both:

- judges who want a quick product-level explanation
- developers who want structured output and integration examples

## Features

- Live website analysis using `requests` + `BeautifulSoup`
- Hacker News-specific extraction logic for better story parsing
- Generic article/content extraction heuristics for other sites
- Inferred schema cards plus raw schema JSON
- Sample data shown as:
  - Cards
  - Table
  - Raw JSON
- Generated API-style endpoints for the extracted analysis
- Developer quickstart examples in:
  - cURL
  - Python
  - JavaScript

## Tech Stack

- Backend: FastAPI
- Frontend: HTML, CSS, vanilla JavaScript
- Parsing: BeautifulSoup
- Server: Uvicorn

## Project Structure

```text
.
├── main.py
├── requirements.txt
├── templates/
│   └── index.html
├── static/
│   ├── styles.css
│   └── app.js
├── Dockerfile
├── Procfile
└── README.md
```

## How To Run Locally

From the `TinyFish NUS copy 4` folder:

```bash
cd "/Users/atharva/Desktop/TinyFish NUS copy 4"
python3 -m pip install -r requirements.txt
python3 -m uvicorn main:app --host 127.0.0.1 --port 8020
```

Then open:

[http://127.0.0.1:8020](http://127.0.0.1:8020)

If you are already inside another virtual environment, open a fresh terminal first so you do not accidentally use the wrong Python environment.

## Example Input

Target URL:

```text
https://news.ycombinator.com
```

Extraction goal:

```text
Extract the top stories and infer a reusable schema for article data.
```

## Example Output

The app can return:

- extracted records with fields such as `title`, `url`, `author`, `published_at`, `score`, and `comments`
- an inferred schema describing the extracted structure
- generated endpoints like:

```text
GET /items?analysis_id=...
GET /items/{id}?analysis_id=...
GET /schema?analysis_id=...
```

## Why This Is Useful

Many websites contain valuable data but do not expose it in a clean developer-friendly format. This project demonstrates a workflow for turning a messy web interface into:

- a structured data layer
- an understandable schema
- a more usable API contract

That makes it useful for:

- research agents
- market intelligence tools
- monitoring dashboards
- web-to-API internal tools

## Notes

- Best experience is local because temporary public tunnels can be slow or unstable.
- `.env`, `venv`, and cache files are excluded from git.
- This project is a prototype built for a hackathon, so extraction quality depends on the target site structure.
