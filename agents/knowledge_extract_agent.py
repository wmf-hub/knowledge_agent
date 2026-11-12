"""
知识抽取子图 — LangGraph 编排，具体抽取能力全部位于 services/knowledge_extract_services/ 目录

子图流程:
  extract_per_chunk → deduplicate → to_triples → END

组件对应关系（services/knowledge_extract_services/）:
  knowledge_extraction.py  → NER / 关系抽取 / 事件抽取
  deduplication.py         → 跨块去重与实体消解
  knowledge_graph.py       → 三元组生成 / Neo4j 写入
  extraction_types.py      → Entity / Relation / KnowledgeEvent / ExtractionResult

共享类型 services/document_types.py（DocumentChunk）作为上游文档解析的输出
"""

from __future__ import annotations

import asyncio
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from services.document_types import DocumentChunk
from services.knowledge_extract_services.deduplication import deduplicate
from services.knowledge_extract_services.extraction_types import ExtractionResult
from services.knowledge_extract_services.knowledge_extraction import extract_from_text
from services.knowledge_extract_services.knowledge_graph import to_triples


class ExtractionState(TypedDict, total=False):
    """子图状态：文档块 → 抽取结果 → 去重结果 → 三元组"""
    chunks: list[DocumentChunk]
    results: list[ExtractionResult]
    triples: list[tuple[str, str, str]]


MAX_CONCURRENCY = 5


def build_graph():
    """构建并编译知识抽取子图"""

    async def extract_node(state: ExtractionState) -> ExtractionState:
        chunks = state.get("chunks", [])
        if not chunks:
            return {"results": []}

        sem = asyncio.Semaphore(MAX_CONCURRENCY)

        async def _one(chunk: DocumentChunk) -> ExtractionResult:
            async with sem:
                return await extract_from_text(chunk.content, chunk.chunk_id)

        results = await asyncio.gather(*(_one(c) for c in chunks), return_exceptions=True)
        # 单个块抽取失败不影响其他块
        return {"results": [r for r in results if isinstance(r, ExtractionResult)]}

    def deduplicate_node(state: ExtractionState) -> ExtractionState:
        return {"results": deduplicate(state.get("results", []))}

    def triples_node(state: ExtractionState) -> ExtractionState:
        return {"triples": to_triples(state.get("results", []))}

    graph = StateGraph(ExtractionState)
    graph.add_node("extract", extract_node)
    graph.add_node("deduplicate", deduplicate_node)
    graph.add_node("triples", triples_node)

    graph.add_edge(START, "extract")
    graph.add_edge("extract", "deduplicate")
    graph.add_edge("deduplicate", "triples")
    graph.add_edge("triples", END)

    return graph.compile()


class KnowledgeExtractAgent:
    """知识抽取 Agent：只负责调用子图，不做具体抽取"""

    def __init__(self) -> None:
        self.graph = build_graph()

    async def extract(self, chunks: list[DocumentChunk]) -> list[ExtractionResult]:
        """从一组文档块中抽取知识，返回去重后的抽取结果"""
        result = await self.graph.ainvoke({"chunks": chunks})
        return result.get("results", [])

    async def extract_with_triples(
        self, chunks: list[DocumentChunk]
    ) -> tuple[list[ExtractionResult], list[tuple[str, str, str]]]:
        """从文档块中抽取知识，同时返回三元组"""
        result = await self.graph.ainvoke({"chunks": chunks})
        return result.get("results", []), result.get("triples", [])