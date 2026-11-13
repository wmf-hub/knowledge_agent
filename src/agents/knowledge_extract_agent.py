"""
知识抽取子图 — LangGraph 编排，

子图流程:
  extract_per_chunk → deduplicate → to_triples → write_neo4j → END

组件对应关系（services/knowledge_extract_services/）:
  knowledge_extraction.py  → NER / 关系抽取 / 事件抽取
  deduplication.py         → 跨块去重与实体消解
  knowledge_graph_store.py       → 三元组生成 / Neo4j 写入
  extraction_types.py      → Entity / Relation / KnowledgeEvent / ExtractionResult

共享类型 services/document_types.py（DocumentChunk）作为上游文档解析的输出
"""

from __future__ import annotations

import asyncio
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from src.services.document_types import DocumentChunk
from src.services.knowledge_extract_services.deduplication import deduplicate
from src.services.knowledge_extract_services.extraction_types import ExtractionResult
from src.services.knowledge_extract_services.knowledge_extraction import extract_from_text
from src.services.knowledge_extract_services.knowledge_graph_store import to_triples, write_to_neo4j


class ExtractionState(TypedDict, total=False):
    """子图状态：文档块 → 抽取结果 → 去重结果 → 三元组 → 写入图谱"""
    chunks: list[DocumentChunk]
    results: list[ExtractionResult]
    triples: list[tuple[str, str, str]]
    write_to_kg: bool
    written_relations: int
    write_error: str


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

    async def write_neo4j_node(state: ExtractionState) -> ExtractionState:
        results = state.get("results", [])
        if not results:
            return {"written_relations": 0}
        # Neo4j 驱动是同步阻塞的，放到线程池避免卡住事件循环
        try:
            written = await asyncio.to_thread(write_to_neo4j, results)
            return {"written_relations": written}
        except Exception as e:
            # 图谱写入失败不应中断子图，记录原因交由调用方判断
            return {"written_relations": 0, "write_error": str(e)}

    def should_write(state: ExtractionState) -> str:
        """默认写入图谱；调用方传 write_to_kg=False 可跳过（例如仅预览抽取结果）"""
        return "write_neo4j" if state.get("write_to_kg", True) else END

    graph = StateGraph(ExtractionState)
    graph.add_node("extract", extract_node)
    graph.add_node("deduplicate", deduplicate_node)
    graph.add_node("triples", triples_node)
    graph.add_node("write_neo4j", write_neo4j_node)

    graph.add_edge(START, "extract")
    graph.add_edge("extract", "deduplicate")
    graph.add_edge("deduplicate", "triples")
    graph.add_conditional_edges("triples", should_write, ["write_neo4j", END])
    graph.add_edge("write_neo4j", END)

    return graph.compile()


class KnowledgeExtractAgent:
    """文本块知识抽取 Agent"""

    def __init__(self) -> None:
        self.graph = build_graph()

    async def extract(
        self, chunks: list[DocumentChunk], write_to_kg: bool = True
    ) -> list[ExtractionResult]:
        """从一组文档块中抽取知识并写入 Neo4j，返回去重后的抽取结果"""
        result = await self.graph.ainvoke({"chunks": chunks, "write_to_kg": write_to_kg})
        return result.get("results", [])

    async def extract_with_triples(
        self, chunks: list[DocumentChunk], write_to_kg: bool = True
    ) -> tuple[list[ExtractionResult], list[tuple[str, str, str]]]:
        """从文档块中抽取知识，同时返回三元组"""
        result = await self.graph.ainvoke({"chunks": chunks, "write_to_kg": write_to_kg})
        return result.get("results", []), result.get("triples", [])

    async def run(
        self, chunks: list[DocumentChunk], write_to_kg: bool = True
    ) -> dict:
        """完整运行子图，返回包含写入统计与错误信息的完整状态"""
        return await self.graph.ainvoke({"chunks": chunks, "write_to_kg": write_to_kg})