"""文档分类组件：根据文件扩展名判定文档类型，并生成文档 ID"""

from __future__ import annotations

import hashlib
import os

from src.services.document_types import DocType

SUPPORTED_EXTENSIONS: dict[str, DocType] = {
    ".pdf": DocType.PDF,
    ".png": DocType.IMAGE,
    ".jpg": DocType.IMAGE,
    ".jpeg": DocType.IMAGE,
    ".csv": DocType.TABLE,
    ".xlsx": DocType.TABLE,
    ".xls": DocType.TABLE,
    ".txt": DocType.TEXT,
    ".md": DocType.MARKDOWN,
}


def classify(file_path: str) -> DocType:
    """根据文件扩展名分类文档类型"""
    ext = os.path.splitext(file_path)[1].lower()
    return SUPPORTED_EXTENSIONS.get(ext, DocType.UNKNOWN)


def make_doc_id(file_path: str) -> str:
    """根据文件路径生成文档 ID（哈希）"""
    return hashlib.sha256(file_path.encode()).hexdigest()[:16]