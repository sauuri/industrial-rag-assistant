from pydantic_settings import BaseSettings
from pydantic import ConfigDict


class Settings(BaseSettings):
    model_config = ConfigDict(env_file=".env")

    openai_api_key: str = ""
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-4o-mini"
    chunk_size: int = 500
    chunk_overlap: int = 50
    vectorstore_path: str = "vectorstore"
    data_path: str = "data/sample_docs"


settings = Settings()
