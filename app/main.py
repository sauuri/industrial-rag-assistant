from fastapi import FastAPI, HTTPException
from app.schemas import QueryRequest, QueryResponse, IngestResponse
from app.ingest import ingest
import app.rag as rag

app = FastAPI(title="Industrial RAG Assistant", version="0.1.0")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest", response_model=IngestResponse)
def ingest_documents():
    try:
        count = ingest()
        return IngestResponse(message="Ingestion complete", documents_processed=count)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
def query_documents(request: QueryRequest):
    try:
        result = rag.query(request.question, request.top_k)
        return QueryResponse(**result)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Vectorstore not found. Run /ingest first.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
