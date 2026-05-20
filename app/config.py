from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    openai_api_key: str = ""
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-4o-mini"
    chunk_size: int = 500
    chunk_overlap: int = 50
    vectorstore_path: str = "vectorstore"
    data_path: str = "data/sample_docs"

    class Config:
        env_file = ".env"


settings = Settings()
