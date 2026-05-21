from pydantic import BaseModel
from typing import List, Optional


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5


class SourceInfo(BaseModel):
    file: str
    page: Optional[int] = None
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceInfo]
    latency_ms: int
    retrieved_chunks: int


class IngestResponse(BaseModel):
    message: str
    documents_processed: int
    chunks_created: int


class DocumentInfo(BaseModel):
    filename: str
    size_kb: float


class UploadResponse(BaseModel):
    message: str
    filename: str
    chunks_created: int
