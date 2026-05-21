# Reranker Service

Local cross-encoder reranker using BAAI/bge-reranker-v2-m3. Multilingual (RU/EN).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

Server starts on http://localhost:8081. First launch downloads the model (~2.3 GB).

## API

### POST /rerank

```bash
curl -s http://localhost:8081/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "query": "доступ по подсетям kafka",
    "documents": [
      {"id": "chunk-1", "text": "Для доступа к kafka нужно настроить ACL по подсетям"},
      {"id": "chunk-2", "text": "PostgreSQL поддерживает репликацию"}
    ],
    "topK": 5
  }' | python3 -m json.tool
```

### GET /health

```bash
curl http://localhost:8081/health
```
