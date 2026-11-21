from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM
    anthropic_api_key: str = ""
    llm_model: str = "claude-sonnet-4-6"
    llm_max_tokens: int = 2048
    llm_temperature: float = 0.0

    # Embeddings
    openai_api_key: str = ""
    embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int = 1536

    # Vector Store
    chroma_persist_dir: str = "data/vector_db"
    chroma_collection_name: str = "transactions"

    # Retrieval
    top_k_results: int = 10
    similarity_threshold: float = 0.7

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"

    # Data Paths
    raw_pdf_dir: str = "data/raw"
    processed_dir: str = "data/processed"

    def ensure_dirs(self) -> None:
        for path in [self.raw_pdf_dir, self.processed_dir, self.chroma_persist_dir]:
            Path(path).mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()
