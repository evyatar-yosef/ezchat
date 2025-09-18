import os
import io
import json
import pickle
import hashlib
from pathlib import Path
from typing import List

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import numpy as np
import pandas as pd
from pypdf import PdfReader

# ===== Embeddings: FastEmbed (קליל, בלי PyTorch) =====
from fastembed import TextEmbedding
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")  # מודל רב-לשוני, חסכוני בזיכרון
_embedder = None
def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = TextEmbedding(model_name=EMBEDDING_MODEL)
    return _embedder

# ===== Gemini (Free tier) =====
import google.generativeai as genai
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-2.0-flash-lite")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# ===== Paths / App =====
TENANTS_DIR = Path("tenants")
TENANTS_DIR.mkdir(exist_ok=True)

app = FastAPI(title="EzChat — Gemini Starter (FastEmbed)")

# בייצור: החלף ["*"] לדומיינים שלך
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# ===== Helpers =====
def chunk_text(text: str, max_chars: int = 900) -> List[str]:
    text = text.replace("\r", "")
    parts, buf = [], []
    total = 0
    for para in text.split("\n\n"):
        if total + len(para) > max_chars:
            if buf:
                parts.append("\n\n".join(buf))
                buf, total = [], 0
        buf.append(para)
        total += len(para) + 2
    if buf:
        parts.append("\n\n".join(buf))
    return [p.strip() for p in parts if p.strip()]

def embed_texts(texts: List[str]) -> np.ndarray:
    # FastEmbed מחזיר generator של וקטורים; נהפוך למערך וננרמל לקוסינוס
    embs = np.asarray(list(get_embedder().embed(texts)), dtype=np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs = embs / (norms + 1e-10)
    return embs

def load_excel(file_bytes: bytes) -> List[dict]:
    df = pd.read_excel(io.BytesIO(file_bytes))
    docs = []
    for row in df.to_dict(orient="records"):
        sku = str(row.get("sku") or row.get("SKU") or row.get("id") or "")
        name = str(row.get("name") or row.get("Name") or "")
        price = row.get("price") or row.get("Price")
        currency = row.get("currency") or row.get("Currency") or "ILS"
        short_desc = str(row.get("short_desc") or "")
        long_desc = str(row.get("long_desc") or "")
        attrs = str(row.get("attributes") or "")
        meta = f"SKU: {sku} | מחיר: {price} {currency}"
        base = f"{name}\n{short_desc}\n{long_desc}\n{attrs}\n{meta}"
        for i, ch in enumerate(chunk_text(base)):
            docs.append({"id": f"excel:{sku}:{i}", "text": ch, "metadata": row})
    return docs

def load_csv(file_bytes: bytes) -> List[dict]:
    df = pd.read_csv(io.BytesIO(file_bytes))
    xls = io.BytesIO()
    df.to_excel(xls, index=False)
    return load_excel(xls.getvalue())

def load_pdf(file_bytes: bytes) -> List[dict]:
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = []
    for p in reader.pages:
        try:
            pages.append(p.extract_text() or "")
        except Exception:
            pages.append("")
    full_text = "\n\n".join(pages)
    return [{"id": f"pdf:{i}", "text": ch, "metadata": {"source": "pdf"}} for i, ch in enumerate(chunk_text(full_text))]

def load_txt(file_bytes: bytes, source: str = "txt") -> List[dict]:
    text = file_bytes.decode("utf-8", errors="ignore")
    return [{"id": f"{source}:{i}", "text": ch, "metadata": {"source": source}} for i, ch in enumerate(chunk_text(text))]

def tenant_path(tenant_id: str) -> Path:
    safe = hashlib.md5(tenant_id.encode()).hexdigest()
    p = TENANTS_DIR / safe
    p.mkdir(exist_ok=True)
    return p

def save_index(tenant_id: str, records: List[dict]):
    texts = [r["text"] for r in records]
    embs = embed_texts(texts)
    payload = {
        "texts": texts,
        "embeddings": embs.astype(np.float32),
        "metadatas": [r.get("metadata", {}) for r in records],
        "ids": [r.get("id") for r in records],
    }
    with open(tenant_path(tenant_id) / "index.pkl", "wb") as f:
        pickle.dump(payload, f)

def load_index(tenant_id: str):
    fp = tenant_path(tenant_id) / "index.pkl"
    if not fp.exists():
        return None
    with open(fp, "rb") as f:
        return pickle.load(f)

def retrieve(tenant_id: str, query: str, k: int = 5):
    idx = load_index(tenant_id)
    if not idx:
        return []
    q_emb = embed_texts([query])[0]
    embs = idx["embeddings"]  # (n, d)
    sims = embs @ q_emb  # cosine (vectors מנורמלים)
    topk = np.argsort(-sims)[:k]
    results = []
    for i in topk:
        results.append({
            "id": idx["ids"][i],
            "text": idx["texts"][i],
            "metadata": idx["metadatas"][i],
            "score": float(sims[i]),
        })
    return results

# ===== Schemas =====
class ChatIn(BaseModel):
    tenant_id: str
    message: str

class ChatOut(BaseModel):
    answer: str
    sources: list = []

# ===== Endpoints =====
@app.get("/health")
def health():
    return {"ok": True}

@app.post("/api/upload")
async def upload(tenant_id: str = Form(...), files: List[UploadFile] = File(...)):
    records: List[dict] = []
    for uf in files:
        data = await uf.read()
        name = (uf.filename or "").lower()
        try:
            if name.endswith(".xlsx") or name.endswith(".xls"):
                records.extend(load_excel(data))
            elif name.endswith(".csv"):
                records.extend(load_csv(data))
            elif name.endswith(".pdf"):
                records.extend(load_pdf(data))
            elif name.endswith(".md"):
                records.extend(load_txt(data, source="md"))
            elif name.endswith(".txt"):
                records.extend(load_txt(data, source="txt"))
            else:
                records.extend(load_txt(data, source="raw"))  # fallback
        except Exception as e:
            return {"ok": False, "error": f"failed {name}: {e}"}

    if not records:
        return {"ok": False, "error": "no records parsed"}

    save_index(tenant_id, records)
    return {"ok": True, "records": len(records)}

@app.post("/api/chat", response_model=ChatOut)
async def chat(body: ChatIn):
    ctx = retrieve(body.tenant_id, body.message, k=5)
    context_text = "\n\n".join([f"[מקור {i+1}]\n" + c["text"] for i, c in enumerate(ctx)])

    system_text = (
        "אתה עוזר חכם לעסק מקומי. השתמש תחילה במידע שסופק בהקשר (קטלוג/תקנון), "
        "ואם משהו לא מופיע שם, אפשר להסתמך על ידע כללי — אבל אל תמציא פרטי מדיניות/מחיר. "
        "אם מידע חסר או לא ברור, אמור זאת והצע מה ניתן לספק כדי לעזור. השב בעברית תקינה וברורה."
    )

    user_text = (
        f"שאלה: {body.message}\n\n"
        f"הקשר (קטעים רלוונטיים מהמוצרים/תקנון):\n{context_text}\n\n"
        f"ענה תשובה ידידותית, עניינית וקצרה ככל האפשר, ואם יש רלוונטיות למוצרים — הצע 2-3 אפשרויות."
    )

    if not GEMINI_API_KEY:
        return ChatOut(answer="לא הוגדר GEMINI_API_KEY – הוסף משתנה סביבה בשרת.", sources=[])

    try:
        model = genai.GenerativeModel(model_name=MODEL_NAME, system_instruction=system_text)
        resp = model.generate_content(user_text)
        answer = getattr(resp, "text", None) or (resp.candidates[0].content.parts[0].text if resp.candidates else "—")
    except Exception as e:
        answer = f"שגיאה בפנייה למודל: {e}"

    return ChatOut(answer=answer, sources=[{"id": c["id"], "score": c["score"]} for c in ctx])
