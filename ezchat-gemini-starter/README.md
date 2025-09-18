
# EzChat — Gemini Free-tier Starter (No-DevOps)

This starter lets you upload product catalogs (Excel/CSV) and policies (PDF/TXT/MD), builds a lightweight local index per tenant, and answers customer questions using Gemini API (Free tier).

## Run locally (optional)
- Create venv; `pip install -r requirements.txt`
- Set `GEMINI_API_KEY` in your environment
- Run: `uvicorn main:app --reload --port 8000`
- Open `/static/upload.html`

## Deploy (Render)
- Create a Web Service from this repo
- Add a **Persistent Disk** mounted to `/opt/render/project/src/tenants` (default working dir on Render)
- Set Environment:
  - `GEMINI_API_KEY`
  - `MODEL_NAME=gemini-2.0-flash-lite`
- Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
