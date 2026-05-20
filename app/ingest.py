from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from app.config import settings


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
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(settings.vectorstore_path)

    return len(docs)
