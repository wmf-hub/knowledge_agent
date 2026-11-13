"""文档监听共享类型：变更类型与文件事件"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ChangeType(str, Enum):
    ADDED = "added"
    MODIFIED = "modified"
    DELETED = "deleted"


@dataclass
class FileEvent:
    """一次文件变更事件"""
    path: str
    change: ChangeType
    fingerprint: str = ""