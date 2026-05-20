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


def ingest() -> int:
    docs = load_documents()
    if not docs:
        raise ValueError(f"No PDF files found in {settings.data_path}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model,
        openai_api_key=settings.openai_api_key,
        chunk_size=BATCH_SIZE,
    )

    vectorstore = None
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        if vectorstore is None:
            vectorstore = FAISS.from_documents(batch, embeddings)
        else:
            vectorstore.add_documents(batch)
        time.sleep(2)

    vectorstore.save_local(settings.vectorstore_path)
    return len(docs)
