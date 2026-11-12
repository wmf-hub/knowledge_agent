"""去重组件：跨 chunk 合并重复实体、关系、事件"""

from __future__ import annotations

from services.knowledge_extract_services.extraction_types import (
    Entity,
    ExtractionResult,
    KnowledgeEvent,
    Relation,
)


def deduplicate(results: list[ExtractionResult]) -> list[ExtractionResult]:
    """跨 chunk 去重: 同名同类型实体合并，关系、事件去重"""
    seen_entities: dict[str, Entity] = {}
    seen_relations: set[tuple[str, str, str]] = set()
    seen_events: set[tuple[str, str]] = set()
    deduped: list[ExtractionResult] = []

    for result in results:
        unique_entities: list[Entity] = []
        for ent in result.entities:
            key = f"{ent.name}::{ent.type}"
            if key not in seen_entities:
                seen_entities[key] = ent
                unique_entities.append(ent)

        unique_relations: list[Relation] = []
        for rel in result.relations:
            key = (rel.head, rel.relation, rel.tail)
            if key not in seen_relations:
                seen_relations.add(key)
                unique_relations.append(rel)

        unique_events: list[KnowledgeEvent] = []
        for ev in result.events:
            key = (ev.trigger, ev.type)
            if key not in seen_events:
                seen_events.add(key)
                unique_events.append(ev)

        deduped.append(ExtractionResult(
            entities=unique_entities,
            relations=unique_relations,
            events=unique_events,
            source_chunk_id=result.source_chunk_id,
        ))
    return deduped