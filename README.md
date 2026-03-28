# TinyFishNUS-ByteMe

Web2API Studio is a hackathon prototype that turns a website into a more usable API-style data surface.

## Run locally

```bash
cd "/Users/atharva/Desktop/TinyFish NUS copy"
python -m pip install -r requirements.txt
python -m uvicorn main:app --host 127.0.0.1 --port 8010
```

Open [http://127.0.0.1:8010](http://127.0.0.1:8010).

## What it does

- analyzes a live website URL
- infers a schema from extracted records
- shows record cards, record table, and raw records JSON
- generates API-style endpoints for the extracted result

## Notes

- Best experience is local; temporary tunnels can be slower.
- `.env`, `venv`, and caches are ignored from git.
