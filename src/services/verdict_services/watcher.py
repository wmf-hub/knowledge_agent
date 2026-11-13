"""文件监听组件：扫描目录、计算指纹、比对差异

指纹用「文件大小 + 修改时间」生成，成本远低于内容哈希，
足以判定文件是否被改动，适合轮询式增量同步。
"""

from __future__ import annotations

import hashlib
import os

from src.services.doc_parser_services.doc_classification import SUPPORTED_EXTENSIONS
from src.services.verdict_services.sync_types import ChangeType, FileEvent

IGNORED_DIRS = {
    "__pycache__",
    ".git",
    ".idea",
    ".vscode",
    ".venv",
    "node_modules",
    "models",
}


def fingerprint(path: str) -> str:
    """计算文件指纹，文件不可读时返回空串"""
    try:
        st = os.stat(path)
    except OSError:
        return ""
    raw = f"{st.st_size}:{st.st_mtime_ns}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def scan(root: str, extensions: set[str] | None = None) -> dict[str, str]:
    """扫描目录下受支持的文件，返回 {绝对路径: 指纹}"""
    exts = extensions if extensions is not None else set(SUPPORTED_EXTENSIONS)
    snapshot: dict[str, str] = {}
    if not os.path.isdir(root):
        return snapshot

    for dirpath, dirnames, filenames in os.walk(root):
        # 跳过缓存目录与隐藏目录，避免无效变更
        dirnames[:] = [
            d for d in dirnames if d not in IGNORED_DIRS and not d.startswith(".")
        ]
        for name in filenames:
            if os.path.splitext(name)[1].lower() not in exts:
                continue
            path = os.path.abspath(os.path.join(dirpath, name))
            fp = fingerprint(path)
            if fp:
                snapshot[path] = fp
    return snapshot


def diff(previous: dict[str, str], current: dict[str, str]) -> list[FileEvent]:
    """比对两次快照，产出新增 / 修改 / 删除事件"""
    events: list[FileEvent] = []
    for path, fp in current.items():
        if path not in previous:
            events.append(FileEvent(path, ChangeType.ADDED, fp))
        elif previous[path] != fp:
            events.append(FileEvent(path, ChangeType.MODIFIED, fp))
    for path in previous:
        if path not in current:
            events.append(FileEvent(path, ChangeType.DELETED))
    return events