import os
import time
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from app.config import settings

BATCH_SIZE = 10


def load_documents():
    loader = DirectoryLoader(
        settings.data_path,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
    )
    return loader.load()


def ingest() -> dict:
    docs = load_documents()
    if not docs:
        raise ValueError(f"No PDF files found in {settings.data_path}")
    return _build_vectorstore(docs, merge=False)


def ingest_file(file_path: str) -> dict:
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    if not docs:
        raise ValueError(f"No content extracted from {file_path}")
    return _build_vectorstore(docs, merge=True)


def _build_vectorstore(docs: list, merge: bool = False) -> dict:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    chunks = splitter.split_documents(docs)
    chunks = [c for c in chunks if len(c.page_content.strip()) >= 100]

    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model,
        openai_api_key=settings.openai_api_key,
        chunk_size=BATCH_SIZE,
    )

    vs_index = os.path.join(settings.vectorstore_path, "index.faiss")
    if merge and os.path.exists(vs_index):
        vectorstore = FAISS.load_local(
            settings.vectorstore_path,
            embeddings,
            allow_dangerous_deserialization=True,
        )
        for i in range(0, len(chunks), BATCH_SIZE):
            vectorstore.add_documents(chunks[i:i + BATCH_SIZE])
            time.sleep(0.5)
    else:
        vectorstore = None
        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i:i + BATCH_SIZE]
            if vectorstore is None:
                vectorstore = FAISS.from_documents(batch, embeddings)
            else:
                vectorstore.add_documents(batch)
            time.sleep(0.5)

    vectorstore.save_local(settings.vectorstore_path)
    return {"documents": len(set(d.metadata.get("source", "") for d in docs)), "chunks": len(chunks)}
