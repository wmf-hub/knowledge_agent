"""
应用配置 — 通过 .env 文件加载
"""

import dotenv
import os
from pydantic_settings import BaseSettings

dotenv.load_dotenv(".env")

class Settings(BaseSettings):
    # LLM
    llm_api_key: str = os.getenv("LLM_API_KEY")
    llm_base_url: str = os.getenv("LLM_BASE_URL")
    llm_model: str = os.getenv("LLM_MODEL")

    # Embedding
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL")
    embedding_model: str = os.getenv("EMBEDDING_MODEL")
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 64

    # Neo4j
    neo4j_uri: str = os.getenv("NEO4J_URI")
    neo4j_user: str = os.getenv("NEO4J_USER")
    neo4j_password: str = os.getenv("NEO4J_PASSWORD")

    # Vector Store
    vector_store_type: str = os.getenv("VECTOR_STORE_TYPE")  # chroma | pgvector
    chroma_host: str = os.getenv("CHROMA_HOST")
    chroma_port: int = int(os.getenv("CHROMA_PORT"))
    pgvector_dsn: str = os.getenv("PGVECTOR_DSN")

    # Kafka (CDC)
    kafka_bootstrap_servers: str = os.getenv("KAFKA_BOOTSTRAP_SERVERS")
    kafka_topic_doc_changes: str = "doc-changes"
    kafka_topic_kg_updates: str = "kg-updates"

    # API
    api_host: str = os.getenv("API_HOST")
    api_port: int = int(os.getenv("API_PORT"))

    # Document Store
    upload_dir: str = "./uploads"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()