"""向量库服务：切块 → 向量化 → 入库 → 混合检索 → 删除

- 切块复用 chunking.Chunker
- 向量化走本地 Ollama 模型（settings.embedding_model，默认 qwen3-embedding:0.6b）
- 存储用 Milvus，稠密向量 + BM25 稀疏向量双路召回，RRFRanker 融合
- 重排用本地模型（settings.rerank_model，默认 bge-reranker-base）
- 删除按文档块来源（source）清理
"""

from __future__ import annotations

import os
from typing import Any

from src.config.settings import settings
from src.services.document_types import DocType, DocumentChunk

EMBED_BATCH_SIZE = 32    # 向量化批量大小
RERANK_CANDIDATE_MULTIPLIER = 4    # 重排候选数倍数
HYBRID_RRF_K = 60    # 混合检索 K 值，越大名次差异越平滑，常用 60
OUTPUT_FIELDS = ["id", "doc_id", "content", "doc_type", "source", "chunk_index", "metadata"]    # 混合检索输出字段

class VectorStore:
    """文档块的向量库：切块、向量化、入库、混合检索、按来源删除"""

    def __init__(
        self,
        uri: str | None = None,
        collection_name: str | None = None,
        embedding_model: str | None = None,
        rerank_model: str | None = None,
    ) -> None:
        self.uri = uri or settings.milvus_uri
        self.collection_name = collection_name or settings.collection_name
        self.embedding_model = embedding_model or settings.embedding_model
        self.rerank_model = rerank_model or settings.rerank_model
        self._client: Any = None
        self._dim: int | None = None
        self._reranker: Any = None

    # ---------- 切块 \ 向量化 \ 入库 ----------

    def ingest_texts(self, chunks: list[DocumentChunk]) -> int:
        """向量化并写入 Milvus（按 chunk_id upsert，重复入库不会产生副本）"""
        if not chunks:
            return 0

        self._ensure_collection()
        embeddings = self._embed([c.content for c in chunks])

        rows = [
            {
                "id": c.chunk_id,
                "doc_id": c.doc_id,
                "content": c.content,
                "doc_type": c.doc_type.value,
                "source": str(c.metadata.get("source", "")),
                "chunk_index": c.chunk_index,
                "metadata": c.metadata,
                "dense": vec,
            }
            for c, vec in zip(chunks, embeddings)
        ]
        self._client.upsert(self.collection_name, rows)
        # 落盘，确保插入数据立即可被检索到
        self._client.flush(self.collection_name)

        return len(rows)

    # ---------- 检索 ----------

    def search(
        self, query: str, top_k: int = 5, top_n: int = 5, use_rerank: bool = True
    ) -> list[DocumentChunk]:
        """混合检索：稠密向量 + BM25 稀疏向量 → RRFRanker 融合(top_k) → 可选重排(top_n) 注：top_k >= top_n"""
        if not query.strip():
            return []
        self._ensure_collection()
        candidates = max(top_k * RERANK_CANDIDATE_MULTIPLIER, top_k) if use_rerank else top_k
        dense_vec = self._embed([query])[0]

        from pymilvus import AnnSearchRequest, RRFRanker

        requests = [
            AnnSearchRequest(
                data=[dense_vec],
                anns_field="dense",
                param={"metric_type": "COSINE"},
                limit=candidates,
            ),
            AnnSearchRequest(
                data=[query],
                anns_field="sparse",
                param={"metric_type": "BM25"},
                limit=candidates,
            ),
        ]
        results = self._client.hybrid_search(
            collection_name=self.collection_name,
            reqs=requests,
            ranker=RRFRanker(k=HYBRID_RRF_K),
            limit=candidates,
            output_fields=OUTPUT_FIELDS,
        )
        chunks = [self._to_chunk(hit) for hit in (results[0] if results else [])]
        if not use_rerank or not chunks:
            return chunks[:top_k]
        return self._rerank(query, chunks, top_n)

    def _to_chunk(self, hit: dict) -> DocumentChunk:
        """Milvus 命中结果 → DocumentChunk"""
        entity = hit.get("entity", hit)
        metadata = dict(entity.get("metadata") or {})
        metadata["score"] = hit.get("distance", 0.0)
        try:
            doc_type = DocType(entity.get("doc_type"))
        except ValueError:
            doc_type = DocType.UNKNOWN
        return DocumentChunk(
            content=entity.get("content", ""),
            doc_id=entity.get("doc_id", ""),
            chunk_index=int(entity.get("chunk_index", 0)),
            doc_type=doc_type,
            metadata=metadata,
        )

    def _rerank(
        self, query: str, chunks: list[DocumentChunk], top_n: int = 5
    ) -> list[DocumentChunk]:
        """用本地 rerank 模型对候选块重排，取前 top_n"""
        try:
            scores = self._get_reranker().predict(
                [[query, c.content] for c in chunks]
            )
        except Exception:
            print("【本地 rerank 模型重排失败!!!】")
            return chunks[:top_n]

        # 兼容部分输出两列分数的模型（取 relevant 列）
        if hasattr(scores, "ndim") and scores.ndim > 1:
            scores = scores[:, 1]
        if isinstance(scores, (int, float)):
            scores = [scores]
        for chunk, score in zip(chunks, scores):
            chunk.metadata["rerank_score"] = float(score)
        ordered = sorted(chunks, key=lambda c: c.metadata["rerank_score"], reverse=True)
        return ordered[:top_n]

    # ---------- 删除 ----------

    def delete_by_source(self, source: str) -> int:
        """按文档块来源删除，返回删除条数"""
        client = self.client
        if not client.has_collection(self.collection_name):
            return 0
        expr = f'source == "{self._escape(source)}"'
        result = client.delete(self.collection_name, filter=expr)
        if isinstance(result, dict):
            return int(result.get("delete_count", 0))
        return int(result or 0)

    @staticmethod
    def _escape(value: str) -> str:
        """转义 Milvus 表达式中的字符串字面量，防止注入"""
        return value.replace("\\", "\\\\").replace('"', '\\"')

    # ---------- 底层客户端 ----------

    @property
    def client(self):
        if self._client is None:
            from pymilvus import MilvusClient

            self._client = MilvusClient(uri=self.uri)
        return self._client

    def _embed(self, texts: list[str]) -> list[list[float]]:
        """批量调用 Ollama 本地模型向量化"""
        import ollama

        client = ollama.Client(host=settings.ollama_base_url)
        vectors: list[list[float]] = []
        for start in range(0, len(texts), EMBED_BATCH_SIZE):
            batch = texts[start : start + EMBED_BATCH_SIZE]
            resp = client.embed(model=self.embedding_model, input=batch)
            embeddings = resp["embeddings"] if isinstance(resp, dict) else resp.embeddings
            vectors.extend([list(v) for v in embeddings])
        return vectors

    def _embed_dim(self) -> int:
        """探测向量维度（由模型实际输出决定，避免硬编码不一致）"""
        if self._dim is None:
            self._dim = len(self._embed(["维度探测"])[0])
        return self._dim

    def _get_reranker(self):
        """使用 sentence_transformers.CrossEncoder 加载本地 rerank 模型（bge-reranker）"""
        if self._reranker is not None:
            return self._reranker

        from sentence_transformers import CrossEncoder

        path = self.rerank_model if os.path.isabs(self.rerank_model) else os.path.abspath(self.rerank_model)
        self._reranker = CrossEncoder(path)
        return self._reranker

    def _ensure_collection(self) -> None:
        """集合不存在时按当前向量维度创建，并建立稠密 / BM25 稀疏索引"""
        from pymilvus import DataType, Function, FunctionType

        client = self.client
        if client.has_collection(self.collection_name):
            return

        dim = settings.embedding_dim or self._embed_dim()
        schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=512)
        schema.add_field("doc_id", DataType.VARCHAR, max_length=256)
        schema.add_field(
            "content",
            DataType.VARCHAR,
            max_length=65535,
            enable_analyzer=True,
            analyzer_params={"type": "chinese"},
        )
        schema.add_field("doc_type", DataType.VARCHAR, max_length=32)
        schema.add_field("source", DataType.VARCHAR, max_length=2048)
        schema.add_field("chunk_index", DataType.INT64)
        schema.add_field("metadata", DataType.JSON)
        schema.add_field("dense", DataType.FLOAT_VECTOR, dim=dim)
        schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
        schema.add_function(
            Function(
                name="text_bm25",
                function_type=FunctionType.BM25,
                input_field_names=["content"],
                output_field_names=["sparse"],
            )
        )

        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="dense", index_type="AUTOINDEX", metric_type="COSINE"
        )
        index_params.add_index(
            field_name="sparse", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25"
        )
        client.create_collection(
            self.collection_name, schema=schema, index_params=index_params
        )

    def close(self) -> None:
        """释放 Milvus 连接"""
        if self._client is not None:
            self._client.close()
            self._client = None