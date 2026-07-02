"""Offline Wikipedia search service.

Serves semantic search over the txtai Wikipedia embeddings index
(https://huggingface.co/NeuML/txtai-wikipedia, ~9GB, lead-paragraph abstracts)
so the assistant's wiki tool works with no internet access at runtime.
The index downloads once from Hugging Face on first start, then loads from cache.
"""

import logging
import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("wiki-offline")

app = FastAPI(title="wiki-offline")
embeddings = None


class SearchResult(BaseModel):
    id: str
    text: str
    score: float


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]


@app.on_event("startup")
def load_index():
    global embeddings
    from txtai import Embeddings

    index = os.getenv("WIKI_INDEX", "neuml/txtai-wikipedia")
    logger.info("Loading Wikipedia index %s (first run downloads ~9GB)...", index)
    embeddings = Embeddings()
    embeddings.load(provider="huggingface-hub", container=index)
    logger.info("Wikipedia index ready")


@app.get("/health")
def health():
    if embeddings is None:
        raise HTTPException(status_code=503, detail="index still loading")
    return {"status": "ok"}


@app.get("/search", response_model=SearchResponse)
def search(q: str, n: int = 2):
    if embeddings is None:
        raise HTTPException(status_code=503, detail="index still loading")
    n = max(1, min(n, 10))
    rows = embeddings.search(q, n)
    results = [
        SearchResult(id=str(row.get("id", "")), text=row.get("text", ""), score=float(row.get("score", 0.0)))
        for row in rows
    ]
    return SearchResponse(query=q, results=results)
