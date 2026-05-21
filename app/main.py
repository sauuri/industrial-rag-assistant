import asyncio
import os
from functools import partial
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from app.schemas import QueryRequest, QueryResponse, IngestResponse, DocumentInfo, UploadResponse
from app.ingest import ingest, ingest_file
import app.rag as rag
from app.config import settings

app = FastAPI(title="Industrial RAG Assistant", version="0.2.0")

static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

os.makedirs(settings.data_path, exist_ok=True)


async def _run_sync(fn, *args, **kwargs):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, partial(fn, *args, **kwargs))


@app.get("/")
def root():
    return FileResponse(os.path.join(static_dir, "index.html"))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/documents")
def list_documents():
    try:
        files = []
        for fname in sorted(os.listdir(settings.data_path)):
            if fname.lower().endswith(".pdf"):
                path = os.path.join(settings.data_path, fname)
                files.append(DocumentInfo(
                    filename=fname,
                    size_kb=round(os.path.getsize(path) / 1024, 1),
                ))
        return {"documents": [f.model_dump() for f in files]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="PDF 파일만 업로드 가능합니다.")

    save_path = os.path.join(settings.data_path, file.filename)
    try:
        data = await file.read()
        with open(save_path, "wb") as f:
            f.write(data)

        result = await _run_sync(ingest_file, save_path)
        return UploadResponse(
            message="업로드 및 인덱싱 완료",
            filename=file.filename,
            chunks_created=result["chunks"],
        )
    except Exception as e:
        if os.path.exists(save_path):
            os.remove(save_path)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest", response_model=IngestResponse)
async def ingest_documents():
    try:
        result = await _run_sync(ingest)
        return IngestResponse(
            message="Ingestion complete",
            documents_processed=result["documents"],
            chunks_created=result["chunks"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    try:
        result = await _run_sync(rag.query, request.question, request.top_k)
        return QueryResponse(**result)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Vectorstore not found. Upload a document first.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
