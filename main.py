import os
import time
import logging
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_NAME = os.environ.get("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
MAX_DOCUMENTS = 50

model = None
tokenizer = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer
    logger.info("Loading model %s ...", MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
    if torch.backends.mps.is_available():
        model = model.to("mps")
        logger.info("Using MPS (Apple Silicon) backend")
    logger.info("Model loaded")
    yield


app = FastAPI(title="Reranker Service", lifespan=lifespan)


class Document(BaseModel):
    id: str
    text: str


class RerankRequest(BaseModel):
    query: str
    documents: list[Document]
    topK: int | None = Field(default=None, ge=1)


class RerankResult(BaseModel):
    id: str
    score: float
    rank: int


class RerankResponse(BaseModel):
    model: str
    results: list[RerankResult]


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME, "ready": model is not None}


@app.post("/rerank", response_model=RerankResponse)
def rerank(req: RerankRequest):
    if not req.documents:
        raise HTTPException(400, "documents list is empty")
    if len(req.documents) > MAX_DOCUMENTS:
        raise HTTPException(400, f"too many documents, max {MAX_DOCUMENTS}")

    start = time.perf_counter()
    device = next(model.parameters()).device

    pairs = [[req.query, doc.text] for doc in req.documents]

    with torch.no_grad():
        inputs = tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=1024,
            return_tensors="pt",
        ).to(device)
        scores = model(**inputs, return_dict=True).logits.view(-1).float()

    scores = scores.cpu().tolist()

    results = [
        {"id": doc.id, "score": round(score, 6)}
        for doc, score in zip(req.documents, scores)
    ]
    results.sort(key=lambda r: r["score"], reverse=True)

    if req.topK is not None:
        results = results[: req.topK]

    for rank, r in enumerate(results, 1):
        r["rank"] = rank

    latency = time.perf_counter() - start
    logger.info(
        "rerank query_len=%d docs=%d topK=%s latency=%.3fs",
        len(req.query),
        len(req.documents),
        req.topK,
        latency,
    )

    return RerankResponse(model=MODEL_NAME, results=results)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8081)
