"""
文档监听子图 — LangGraph 编排，负责文档增量同步：向量库与知识图谱更新

子图流程:
  detect ─┬─→ sync ─→ commit ─→ END
          └─→ commit ─→ END   （无变更时直接落账）

组件对应关系（services/verdict_services/）:
  watcher.py       → 目录扫描、指纹计算、差异比对
  sync_ledger.py   → 已同步指纹台账（跨进程持久化）
  graph_sync.py    → 按来源写入 / 删除知识图谱
  sync_types.py    → FileEvent / ChangeType

复用上游子图:
  agents/doc_parser_agent.py        → 解析文件得到文档块
  agents/knowledge_extract_agent.py → 抽取知识（图谱写入由本子图统一控制）
  services/doc_parser_services/vector_store.py → 向量库入库与删除
"""

from __future__ import annotations

import asyncio
import os
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from src.agents.doc_parser_agent import DocParserAgent
from src.agents.knowledge_extract_agent import KnowledgeExtractAgent
from src.config.settings import settings
from src.services.doc_parser_services.vector_store import VectorStore
from src.services.verdict_services import graph_sync, watcher
from src.services.verdict_services.sync_ledger import SyncLedger
from src.services.verdict_services.sync_types import ChangeType, FileEvent


class VerdictState(TypedDict, total=False):
    """子图状态：监听根目录 → 变更事件 → 同步结果 → 台账快照"""
    root: str
    ledger_path: str
    events: list[FileEvent]
    snapshot: dict[str, str]
    synced: int
    removed: int
    errors: list[str]
    failed: list[str]
    skip_commit: bool


def build_graph():
    """构建并编译文档监听子图"""

    def detect_node(state: VerdictState) -> VerdictState:
        root = state["root"]
        # 目录不存在时不能当作「全部删除」处理，否则会清空向量库与图谱
        if not os.path.isdir(root):
            return {
                "snapshot": {},
                "events": [],
                "errors": [f"{root}: 文档目录不存在，已跳过本轮同步"],
                "skip_commit": True,
            }
        snapshot = watcher.scan(root)
        previous = SyncLedger(state["ledger_path"]).load()
        return {"snapshot": snapshot, "events": watcher.diff(previous, snapshot)}

    async def sync_node(state: VerdictState) -> VerdictState:
        parser = DocParserAgent()
        extractor = KnowledgeExtractAgent()
        store = VectorStore()
        synced = 0
        removed = 0
        errors: list[str] = []
        failed: list[str] = []
        try:
            for event in state.get("events", []):
                try:
                    if event.change == ChangeType.DELETED:
                        removed += await asyncio.to_thread(
                            store.delete_by_source, event.path
                        )
                        await asyncio.to_thread(graph_sync.delete_by_source, event.path)
                        continue

                    # 修改的文件先清理旧数据：块数可能变少，残留块会成为脏数据
                    if event.change == ChangeType.MODIFIED:
                        await asyncio.to_thread(store.delete_by_source, event.path)
                        await asyncio.to_thread(graph_sync.delete_by_source, event.path)

                    # 解析子图内不再入库，由本子图统一写入，避免重复向量化
                    chunks = await parser.parse(event.path, ingest=False)
                    if not chunks:
                        continue
                    synced += await asyncio.to_thread(store.ingest_texts, chunks)
                    # 图谱写入由本子图统一控制，故跳过抽取子图内的写入
                    results = await extractor.extract(chunks, write_to_kg=False)
                    await asyncio.to_thread(
                        graph_sync.upsert_knowledge, results, event.path
                    )
                except Exception as e:
                    # 单个文件失败不阻塞其余文件
                    errors.append(f"{event.path}: {e}")
                    failed.append(event.path)
        finally:
            store.close()
        return {
            "synced": synced,
            "removed": removed,
            "errors": errors,
            "failed": failed,
        }

    def commit_node(state: VerdictState) -> VerdictState:
        # 本轮未成功扫描目录时保留台账原样，避免下次误判为全部删除
        if state.get("skip_commit"):
            return {}
        snapshot = dict(state.get("snapshot", {}))
        # 失败的文件不记账，下次轮询会重新触发同步，避免脏数据被永久跳过
        for path in state.get("failed", []):
            snapshot.pop(path, None)
        SyncLedger(state["ledger_path"]).save(snapshot)
        return {}

    def has_events(state: VerdictState) -> str:
        return "sync" if state.get("events") else "commit"

    graph = StateGraph(VerdictState)
    graph.add_node("detect", detect_node)
    graph.add_node("sync", sync_node)
    graph.add_node("commit", commit_node)

    graph.add_edge(START, "detect")
    graph.add_conditional_edges("detect", has_events, ["sync", "commit"])
    graph.add_edge("sync", "commit")
    graph.add_edge("commit", END)

    return graph.compile()


class VerdictAgent:
    """文档监听 Agent：增量同步向量库与知识图谱"""

    def __init__(self, root: str | None = None, ledger_path: str | None = None) -> None:
        self.root = root or settings.docs_dir
        self.ledger_path = ledger_path or os.path.join(self.root, ".sync_ledger.json")
        self.graph = build_graph()

    async def sync(self) -> dict:
        """执行一次增量同步，返回包含统计与错误的完整状态"""
        return await self.graph.ainvoke(
            {"root": self.root, "ledger_path": self.ledger_path}
        )

    async def watch_forever(self, interval: float = 30.0) -> None:
        """轮询监听，按固定间隔持续同步"""
        while True:
            await self.sync()
            await asyncio.sleep(interval)