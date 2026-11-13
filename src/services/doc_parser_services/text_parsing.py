"""纯文本 / Markdown 解析组件"""

from __future__ import annotations

import asyncio


def parse_text(file_path: str) -> list[str]:
    """读取纯文本或 Markdown 文件内容"""
    with open(file_path, encoding="utf-8") as f:
        return [f.read()]


async def parse_text_async(file_path: str) -> list[str]:
    """异步读取入口，避免阻塞事件循环"""
    return await asyncio.to_thread(parse_text, file_path)