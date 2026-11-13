"""图谱同步组件：按文档来源写入 / 删除知识

实体节点以「标签 + 名称」MERGE，多个文档可能共享同一实体，
因此额外建 Document 节点并以 MENTIONS 关系挂接：
删除某个来源时只断开它的 MENTIONS，实体仅在所有来源都不再引用时才清除。
"""

from __future__ import annotations

from src.services.knowledge_extract_services.extraction_types import ExtractionResult
from src.services.knowledge_extract_services.knowledge_graph_store import safe_label


def _driver():
    from neo4j import GraphDatabase

    from src.config.settings import settings

    return GraphDatabase.driver(
        settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password)
    )


def upsert_knowledge(results: list[ExtractionResult], source: str) -> int:
    """写入某来源的实体与关系，返回写入的关系条数（MERGE 幂等）"""
    if not results:
        return 0
    driver = _driver()
    written = 0
    try:
        with driver.session() as session:
            session.run(
                "MERGE (d:Document {source: $source})", source=source
            )
            for result in results:
                label_of: dict[str, str] = {}
                for ent in result.entities:
                    label = safe_label(ent.node_label, "Concept")
                    label_of[ent.name] = label
                    session.run(
                        f"MERGE (n:`{label}` {{name: $name}}) "
                        "SET n.description = $description "
                        "WITH n "
                        "MATCH (d:Document {source: $source}) "
                        "MERGE (d)-[:MENTIONS]->(n)",
                        name=ent.name,
                        description=ent.description,
                        source=source,
                    )
                for rel in result.relations:
                    head_label = label_of.get(rel.head)
                    tail_label = label_of.get(rel.tail)
                    if head_label and tail_label:
                        head_clause = f"MATCH (h:`{head_label}` {{name: $head}})"
                        tail_clause = f"MATCH (t:`{tail_label}` {{name: $tail}})"
                    else:
                        head_clause = "MERGE (h:Entity {name: $head})"
                        tail_clause = "MERGE (t:Entity {name: $tail})"
                    rel_type = safe_label(rel.relation, "related_to")
                    session.run(
                        f"{head_clause} {tail_clause} "
                        f"MERGE (h)-[:`{rel_type}`]->(t)",
                        head=rel.head,
                        tail=rel.tail,
                    )
                    written += 1
    finally:
        driver.close()
    return written


def delete_by_source(source: str) -> int:
    """删除某来源的 Document 节点及其独占实体，返回清除的实体数"""
    driver = _driver()
    removed = 0
    try:
        with driver.session() as session:
            # 先记录该来源挂接过的实体，删除后据此判断是否成为孤儿
            rows = session.run(
                "MATCH (d:Document {source: $source})-[:MENTIONS]->(n) "
                "RETURN labels(n) AS labels, n.name AS name",
                source=source,
            )
            targets = [
                (r["labels"], r["name"]) for r in rows if r["labels"] and r["name"]
            ]

            session.run(
                "MATCH (d:Document {source: $source}) DETACH DELETE d", source=source
            )

            for labels, name in targets:
                label = next(
                    (lbl for lbl in labels if lbl != "Document"), None
                )
                if not label:
                    continue
                session.run(
                    f"MATCH (n:`{label}` {{name: $name}}) "
                    "WHERE NOT ()-[:MENTIONS]->(n) "
                    "DETACH DELETE n",
                    name=name,
                )
                removed += 1
    finally:
        driver.close()
    return removed