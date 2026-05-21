import os
import time
import logging
from contextlib import asynccontextmanager

import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RERANKER_MODEL = os.environ.get("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "intfloat/multilingual-e5-small")
MAX_DOCUMENTS = 50
MAX_TEXTS = 100

reranker = None
reranker_tokenizer = None
embedder = None
embedder_tokenizer = None


def _get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global reranker, reranker_tokenizer, embedder, embedder_tokenizer
    device = _get_device()

    logger.info("Loading reranker %s ...", RERANKER_MODEL)
    reranker_tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL)
    reranker = AutoModelForSequenceClassification.from_pretrained(RERANKER_MODEL)
    reranker.eval().to(device)

    logger.info("Loading embedder %s ...", EMBED_MODEL)
    embedder_tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)
    embedder = AutoModel.from_pretrained(EMBED_MODEL)
    embedder.eval().to(device)

    logger.info("Models loaded on %s", device)
    yield


app = FastAPI(title="Reranker & Embedding Service", lifespan=lifespan)


# --- Models ---

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


class EmbedRequest(BaseModel):
    texts: list[str]


class EmbedResponse(BaseModel):
    model: str
    dimensions: int
    embeddings: list[list[float]]


# --- Endpoints ---

@app.get("/health")
def health():
    return {
        "status": "ok",
        "reranker": RERANKER_MODEL,
        "embedder": EMBED_MODEL,
        "ready": reranker is not None and embedder is not None,
    }


@app.post("/rerank", response_model=RerankResponse)
def rerank(req: RerankRequest):
    if not req.documents:
        raise HTTPException(400, "documents list is empty")
    if len(req.documents) > MAX_DOCUMENTS:
        raise HTTPException(400, f"too many documents, max {MAX_DOCUMENTS}")

    start = time.perf_counter()
    device = next(reranker.parameters()).device

    pairs = [[req.query, doc.text] for doc in req.documents]

    with torch.no_grad():
        inputs = reranker_tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=1024,
            return_tensors="pt",
        ).to(device)
        scores = reranker(**inputs, return_dict=True).logits.view(-1).float()

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

    return RerankResponse(model=RERANKER_MODEL, results=results)


@app.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest):
    if not req.texts:
        raise HTTPException(400, "texts list is empty")
    if len(req.texts) > MAX_TEXTS:
        raise HTTPException(400, f"too many texts, max {MAX_TEXTS}")

    start = time.perf_counter()
    device = next(embedder.parameters()).device

    # multilingual-e5 models expect "query: " or "passage: " prefix
    prefixed = [f"passage: {t}" for t in req.texts]

    with torch.no_grad():
        inputs = embedder_tokenizer(
            prefixed,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)
        output = embedder(**inputs)
        # mean pooling
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        vectors = (output.last_hidden_state * mask).sum(1) / mask.sum(1)
        vectors = F.normalize(vectors, p=2, dim=1)

    embeddings = vectors.cpu().tolist()

    latency = time.perf_counter() - start
    logger.info(
        "embed texts=%d dimensions=%d latency=%.3fs",
        len(req.texts),
        len(embeddings[0]),
        latency,
    )

    return EmbedResponse(
        model=EMBED_MODEL,
        dimensions=len(embeddings[0]),
        embeddings=embeddings,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8081)
