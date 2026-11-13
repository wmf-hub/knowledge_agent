"""
应用配置 — 通过 .env 文件加载
"""

import dotenv
import os
from pathlib import Path

from pydantic_settings import BaseSettings

dotenv.load_dotenv(Path(__file__).resolve().parents[2] / ".env")

class Settings(BaseSettings):
    # LLM
    llm_api_key: str = os.getenv("LLM_API_KEY")
    llm_base_url: str = os.getenv("LLM_BASE_URL")
    llm_model: str = os.getenv("LLM_MODEL")

    # Embedding
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL")
    embedding_model: str = os.getenv("EMBEDDING_MODEL")
    embedding_dim: int = int(os.getenv("EMBEDDING_DIM")) or None
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 64

    # Milvus
    milvus_uri: str = os.getenv("MILVUS_URI")
    collection_name: str = os.getenv("COLLECTION_NAME")

    # Rerank
    rerank_model: str = os.getenv("RERANK_MODEL")

    # Neo4j
    neo4j_uri: str = os.getenv("NEO4J_URI")
    neo4j_user: str = os.getenv("NEO4J_USER")
    neo4j_password: str = os.getenv("NEO4J_PASSWORD")

    # PGVector
    pgvector_dsn: str = os.getenv("POSTGRES_DSN")

    # API
    api_host: str = os.getenv("API_HOST")
    api_port: int = int(os.getenv("API_PORT"))

settings = Settings()