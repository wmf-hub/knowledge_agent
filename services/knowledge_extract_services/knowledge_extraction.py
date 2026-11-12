"""知识抽取组件：NER / 关系抽取 / 事件抽取（LLM + 容错 JSON 解析）"""

from __future__ import annotations

import json
from typing import Any

from services.knowledge_extract_services.extraction_types import (
    Entity,
    ExtractionResult,
    KnowledgeEvent,
    Relation,
)

EXTRACTION_SYSTEM_PROMPT = """\
你是一个专业的知识抽取引擎。给定一段文本，请提取其中的：
1. **实体 (entities)**：人名、组织、地点、产品、技术、概念等
2. **关系 (relations)**：实体之间的关系，用三元组 (头实体, 关系, 尾实体) 表示
3. **事件 (events)**：文本中提到的事件，包含触发词和参与者

请严格按照以下 JSON 格式返回：
{
  "entities": [
    {"name": "实体名", "type": "实体类型", "description": "简短描述"}
  ],
  "relations": [
    {"head": "头实体", "relation": "关系类型", "tail": "尾实体", "confidence": 0.95}
  ],
  "events": [
    {"trigger": "触发词", "type": "事件类型", "participants": ["参与者1"]}
  ]
}

注意:
- 实体类型包括: Person, Organization, Location, Product, Technology, Concept, Event, Time
- 关系类型包括: belongs_to, works_at, located_in, developed_by, related_to, part_of, uses, depends_on
- confidence 为 0-1 之间的浮点数
- 只返回 JSON，不要包含其他文字
"""


async def extract_from_text(text: str, source_id: str = "") -> ExtractionResult:
    """从单段文本中抽取实体、关系、事件"""
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_openai import ChatOpenAI

    from config.settings import settings

    llm = ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        temperature=0,
    )
    messages = [
        SystemMessage(content=EXTRACTION_SYSTEM_PROMPT),
        HumanMessage(content=f"请从以下文本中抽取知识：\n\n{text}"),
    ]
    resp = await llm.ainvoke(messages)
    return parse_response(resp.content, source_id)


def parse_response(raw: str, source_id: str = "") -> ExtractionResult:
    """解析 LLM 响应，提取知识"""
    empty = ExtractionResult(entities=[], relations=[], events=[], source_chunk_id=source_id)
    data = _load_json(raw)
    if not isinstance(data, dict):
        return empty

    entities = [
        Entity(
            name=str(e["name"]).strip(),
            type=str(e.get("type") or "Concept"),
            description=str(e.get("description") or ""),
        )
        for e in _as_list(data.get("entities"))
        if isinstance(e, dict) and str(e.get("name") or "").strip()
    ]
    relations = [
        Relation(
            head=str(r["head"]).strip(),
            relation=str(r.get("relation") or "related_to"),
            tail=str(r["tail"]).strip(),
            confidence=_as_confidence(r.get("confidence")),
        )
        for r in _as_list(data.get("relations"))
        if isinstance(r, dict)
        and str(r.get("head") or "").strip()
        and str(r.get("tail") or "").strip()
    ]
    events = [
        KnowledgeEvent(
            trigger=str(ev["trigger"]).strip(),
            type=str(ev.get("type") or ""),
            participants=[str(p) for p in _as_list(ev.get("participants"))],
        )
        for ev in _as_list(data.get("events"))
        if isinstance(ev, dict) and str(ev.get("trigger") or "").strip()
    ]
    return ExtractionResult(
        entities=entities,
        relations=relations,
        events=events,
        source_chunk_id=source_id,
    )


def _load_json(raw: str) -> Any:
    """去掉 Markdown 代码围栏后解析 JSON，失败返回 None"""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1]
        cleaned = cleaned.rsplit("```", 1)[0].strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None


def _as_list(value: Any) -> list:
    """LLM 偶尔返回单个对象而非数组，统一成列表"""
    if isinstance(value, list):
        return value
    return [value] if value else []


def _as_confidence(value: Any) -> float:
    """confidence 归一化到 0-1，非法值回退 0.5"""
    try:
        conf = float(value)
    except (TypeError, ValueError):
        return 0.5
    return max(0.0, min(1.0, conf))