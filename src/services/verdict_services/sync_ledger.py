"""同步台账组件：持久化已同步文件的指纹，用于跨进程比对变更

删除事件无法从「当前目录」推断，必须依赖上一次的快照，
因此台账需要落盘。写入采用临时文件 + 原子替换，避免中途崩溃留下半截 JSON。
"""

from __future__ import annotations

import json
import os


class SyncLedger:
    """记录每个已同步文件的指纹"""

    def __init__(self, path: str) -> None:
        self.path = path

    def load(self) -> dict[str, str]:
        """读取台账，文件缺失或损坏时返回空快照"""
        if not os.path.exists(self.path):
            return {}
        try:
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            return {}
        if not isinstance(data, dict):
            return {}
        return {str(k): str(v) for k, v in data.items()}

    def save(self, snapshot: dict[str, str]) -> None:
        """原子写入台账"""
        directory = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(directory, exist_ok=True)
        tmp_path = f"{self.path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, self.path)