"""
文档解析子图 — LangGraph 编排，具体解析能力全部位于 services/doc_parser_services/ 目录

子图流程:
  classify ─┬─→ parse_pdf   ─┐
            ├─→ parse_image ─┤
            ├─→ parse_table ─┼─→ chunk → END
            └─→ parse_text  ─┘

组件对应关系（services/doc_parser_services/）:
  doc_classification.py  → 文档分类
  PDF_parsing.py         → PDF 多模态解析
  image_parsing.py       → 图片 OCR + 视觉理解
  table_parsing.py       → CSV / Excel 结构化
  text_parsing.py        → 纯文本 / Markdown
  ocr.py                 → 转图 / OCR / LLM 视觉描述（PDF 与图片共用）
  chunking.py            → 分块与元数据

共享类型 services/document_types.py（DocType / DocumentChunk）由各子图共用
"""

from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from src.config.settings import settings
from src.services.doc_parser_services.chunking import Chunker
from src.services.doc_parser_services.vector_store import VectorStore
from src.services.doc_parser_services.doc_classification import classify, make_doc_id
from src.services.doc_parser_services.image_parsing import parse_image
from src.services.doc_parser_services.PDF_parsing import parse_pdf
from src.services.doc_parser_services.table_parsing import parse_table
from src.services.doc_parser_services.text_parsing import parse_text
from src.services.document_types import DocType, DocumentChunk


class ParserState(TypedDict, total=False):
    """子图状态：文件路径 → 分类 → 原始文本 → 文档块"""
    file_path: str
    doc_type: DocType
    doc_id: str
    raw_texts: list[str]
    chunks: list[DocumentChunk]
    num_added: int


def build_graph():
    """构建并编译文档解析子图"""

    def classify_node(state: ParserState) -> ParserState:
        file_path = state["file_path"]
        return {
            "doc_type": classify(file_path),
            "doc_id": make_doc_id(file_path),
        }

    async def parse_pdf_node(state: ParserState) -> ParserState:
        return {"raw_texts": await parse_pdf(state["file_path"])}

    async def parse_image_node(state: ParserState) -> ParserState:
        return {"raw_texts": await parse_image(state["file_path"])}

    async def parse_table_node(state: ParserState) -> ParserState:
        return {"raw_texts": await parse_table(state["file_path"])}

    def parse_text_node(state: ParserState) -> ParserState:
        return {"raw_texts": parse_text(state["file_path"])}

    def chunk_node(state: ParserState) -> ParserState:
        chunker = Chunker(settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
        return {"chunks": chunker.chunk(state["raw_texts"], state["doc_id"], state["doc_type"], state["file_path"])}

    def vector_store_node(state: ParserState) -> ParserState:
        vector_store = VectorStore()
        num_added = vector_store.ingest_texts(state["chunks"])
        return {"num_added": num_added}

    def route_by_type(state: ParserState) -> str:
        """按文档类型路由到对应解析组件"""
        doc_type = state["doc_type"]
        if doc_type == DocType.PDF:
            return "parse_pdf"
        if doc_type == DocType.IMAGE:
            return "parse_image"
        if doc_type == DocType.TABLE:
            return "parse_table"
        return "parse_text"

    graph = StateGraph(ParserState)
    graph.add_node("classify", classify_node)
    graph.add_node("parse_pdf", parse_pdf_node)
    graph.add_node("parse_image", parse_image_node)
    graph.add_node("parse_table", parse_table_node)
    graph.add_node("parse_text", parse_text_node)
    graph.add_node("chunk", chunk_node)
    graph.add_node("vector_store", vector_store_node)

    graph.add_edge(START, "classify")
    graph.add_conditional_edges(
        "classify",
        route_by_type,
        {
            "parse_pdf": "parse_pdf",
            "parse_image": "parse_image",
            "parse_table": "parse_table",
            "parse_text": "parse_text",
        },
    )
    for parse_node in ("parse_pdf", "parse_image", "parse_table", "parse_text"):
        graph.add_edge(parse_node, "chunk")
    graph.add_edge("chunk", "vector_store")
    graph.add_edge("vector_store", END)

    return graph.compile()


class DocParserAgent:
    """文档解析 Agent"""

    def __init__(self) -> None:
        self.graph = build_graph()

    async def parse(self, file_path: str) -> list[DocumentChunk]:
        """解析单个文件并入库"""
        result = await self.graph.ainvoke({"file_path": file_path})
        return result.get("chunks", [])

    async def parse_batch(self, file_paths: list[str]) -> list[list[DocumentChunk]]:
        """批量解析多个文件并入库"""
        result = []
        for fp in file_paths:
            result.append(await self.parse(fp))
        return result
