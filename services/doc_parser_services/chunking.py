"""文档分块组件：把解析出的文本切分为文档块，并标注元数据"""

from __future__ import annotations

from services.document_types import DocType, DocumentChunk


class Chunker:
    """
    文档分块器。

    CHUNK_SIZE 是「入向量库」的分块尺寸，统一在此处生效：
      - 表格文本按行切分，每块重复表头，避免一行数据被字符边界截断
      - 其余文本按字符切分，保留重叠
    """

    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(
        self,
        texts: list[str],
        doc_id: str,
        doc_type: DocType,
        source: str,
    ) -> list[DocumentChunk]:
        """将文本列表转换为文档块"""
        chunks: list[DocumentChunk] = []
        idx = 0
        for text in texts:
            parts = (
                self._table_chunks(text)
                if doc_type == DocType.TABLE
                else self._text_chunks(text)
            )
            for content, start, end in parts:
                if not content.strip():
                    continue
                chunks.append(DocumentChunk(
                    content=content.strip(),
                    doc_id=doc_id,
                    chunk_index=idx,
                    doc_type=doc_type,
                    metadata={"source": source, "char_start": start, "char_end": end},
                ))
                idx += 1
        return chunks

    def _text_chunks(self, text: str) -> list[tuple[str, int, int]]:
        """按字符切分，返回 (内容, 起止偏移) 列表"""
        parts: list[tuple[str, int, int]] = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            parts.append((text[start:end], start, end))
            if end >= len(text):
                break
            start = end - self.chunk_overlap
        return parts

    def _table_chunks(self, text: str) -> list[tuple[str, int, int]]:
        """
        表格文本按行切分：每块都重复表头（含标题行），保证单块不超过 chunk_size，
        且不会把某一行数据从中间切开。

        返回的起止偏移指向「数据行」在原文中的位置（不含重复的表头块）。
        """
        lines = text.split("\n")
        offsets: list[int] = []
        pos = 0
        for line in lines:
            offsets.append(pos)
            pos += len(line) + 1

        sep_idx = next(
            (i for i, line in enumerate(lines) if self._is_md_separator(line)),
            None,
        )
        if sep_idx is None:
            return self._text_chunks(text)

        header_block = "\n".join(lines[: sep_idx + 1])
        data_start = sep_idx + 1
        if data_start >= len(lines):
            return [(header_block, 0, len(text))]

        budget = self.chunk_size - len(header_block) - 1
        if budget <= 0:
            return self._text_chunks(text)

        parts: list[tuple[str, int, int]] = []
        batch: list[str] = []
        batch_size = 0
        batch_from = data_start
        for i in range(data_start, len(lines)):
            line = lines[i]
            if len(line) > budget:
                # 单行超长（单元格内容过多）：拆成多个带头部的块，仍保证不超 chunk_size
                if batch:
                    parts.append((
                        header_block + "\n" + "\n".join(batch),
                        offsets[batch_from],
                        offsets[i - 1] + len(lines[i - 1]),
                    ))
                    batch, batch_size = [], 0
                step = budget - 1
                for s in range(0, len(line), step):
                    parts.append((
                        header_block + "\n" + line[s : s + step],
                        offsets[i] + s,
                        offsets[i] + min(s + step, len(line)),
                    ))
                batch_from = i + 1
                continue
            if batch and batch_size + len(line) + 1 > budget:
                parts.append((
                    header_block + "\n" + "\n".join(batch),
                    offsets[batch_from],
                    offsets[i - 1] + len(lines[i - 1]),
                ))
                batch, batch_size, batch_from = [], 0, i
            batch.append(line)
            batch_size += len(line) + 1
        if batch:
            last = len(lines) - 1
            parts.append((
                header_block + "\n" + "\n".join(batch),
                offsets[batch_from],
                offsets[last] + len(lines[last]),
            ))
        return parts

    @staticmethod
    def _is_md_separator(line: str) -> bool:
        """判断是否为 Markdown 表格的分隔行（| --- | --- |）"""
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        return bool(cells) and all(c and set(c) == {"-"} for c in cells)