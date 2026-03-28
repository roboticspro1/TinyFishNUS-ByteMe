# Web2API Studio

A cleaner, demo-friendly frontend for your hackathon idea: paste a website URL, infer a schema, and show generated API endpoints.

## Run locally

```bash
cd "/Users/atharva/Documents/New project/web2api-studio"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000).

## Notes

- The current backend is in demo mode so the site is fully runnable without external keys.
- This is meant to give you a polished pitch surface you can extend with real TinyFish and OpenAI calls later.
