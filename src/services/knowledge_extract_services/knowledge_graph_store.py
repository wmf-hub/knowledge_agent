"""知识图谱组件：三元组生成与 Neo4j 写入"""

from __future__ import annotations

from src.services.knowledge_extract_services.extraction_types import ExtractionResult


def to_triples(results: list[ExtractionResult]) -> list[tuple[str, str, str]]:
    """把抽取结果拍平成 (头实体, 关系, 尾实体) 三元组"""
    return [
        (rel.head, rel.relation, rel.tail)
        for result in results
        for rel in result.relations
    ]


def safe_label(raw: str, fallback: str) -> str:
    """把 LLM 产出的类型名清洗成合法 Cypher 标签，防止注入"""
    label = "".join(c if c.isalnum() or c == "_" else "_" for c in raw or "")
    return label if label and label[0].isalpha() else fallback


def write_to_neo4j(results: list[ExtractionResult]) -> int:
    """
    把实体与三元组写入 Neo4j（MERGE 幂等），返回写入的关系条数。

    先按实体类型建节点，再让关系端点复用这些节点，
    避免同一实体同时以“带类型标签”和“无名标签”两种形式存在。
    """
    from neo4j import GraphDatabase

    from src.config.settings import settings

    driver = GraphDatabase.driver(
        settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password)
    )
    written = 0
    try:
        with driver.session() as session:
            for result in results:
                label_of: dict[str, str] = {}
                for ent in result.entities:
                    label = safe_label(ent.node_label, "Concept")
                    label_of[ent.name] = label
                    session.run(
                        f"MERGE (n:`{label}` {{name: $name}}) "
                        "SET n.description = $description",
                        name=ent.name,
                        description=ent.description,
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