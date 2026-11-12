"""PDF 多模态解析组件：文字层 + 矢量表格 + 嵌入图片 OCR + LLM 视觉精修"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor

from services.doc_parser_services.ocr import (
    OCR_LOW_QUALITY_THRESHOLD,
    describe_image_with_llm,
    ocr,
)


def extract_with_pdfplumber(file_path: str) -> list[dict]:
    """获取 PDF 每一页的信息，并提取出“文字和表格”部分的内容"""
    import pdfplumber

    pages_info: list[dict] = []
    with pdfplumber.open(file_path) as pdf:
        for page_no, page in enumerate(pdf.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""
            try:
                tables = page.extract_tables() or []
            except Exception:
                tables = []
            try:
                has_images = len(page.images) > 0
            except Exception:
                has_images = False

            page_texts: list[str] = []
            if text.strip():
                page_texts.append(text.strip())
            for table in tables:
                if not table:
                    continue
                rows = []
                for row in table:
                    cells = [
                        (str(c).replace("\n", " ").strip() if c is not None else "")
                        for c in row
                    ]
                    rows.append(" | ".join(cells))
                page_texts.append("[表格]\n" + "\n".join(rows))

            pages_info.append({
                "page_no": page_no,
                "page_content": "\n".join(page_texts),
                "has_tables": bool(tables),
                "has_images": has_images,
            })
    return pages_info


async def parse_pdf(file_path: str) -> list[str]:
    """
    PDF 多模态解析（逐页策略）：
        1. 用 pdfplumber 抽文字 + 表格（表格转结构化文本）
        2. 该页有图片 → OCR 提取页面截图的文字
        3. OCR 置信度低 或者 该页有表（图片图表需 LLM 理解结构） → 转图 + LLM 视觉描述
        4. 保留该页原有文本，以便 OCR 或 LLM 失败时有兜底内容，不中断整体流程
    """
    # PDF 解析是同步阻塞的，放到线程池避免卡事件循环
    loop = asyncio.get_running_loop()
    executor = ThreadPoolExecutor(max_workers=2)

    try:
        pages_info = await loop.run_in_executor(executor, extract_with_pdfplumber, file_path)
    except Exception as e:
        return [f"[PDF 解析失败] {file_path}: {e}"]

    if not pages_info:
        return [f"[PDF 无内容] {file_path}"]

    # 仅对有图片的页做 OCR
    ocr_pages = [p for p in pages_info if p["has_images"]]
    if ocr_pages:
        tasks = [
            loop.run_in_executor(executor, ocr, file_path, page["page_no"])
            for page in ocr_pages
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for page, res in zip(ocr_pages, results):
            if isinstance(res, Exception):
                page["ocr_error"] = str(res)
                page["ocr_text"] = ""
                page["ocr_confidence"] = 0.0
            else:
                page["ocr_text"], page["ocr_confidence"], page["ocr_image"] = res

    # OCR 置信度低 或 这页有表格 → 交给 LLM 精修
    llm_pages = [
        p for p in ocr_pages
        if p.get("ocr_confidence", 0) < OCR_LOW_QUALITY_THRESHOLD or p.get("has_tables")
    ]
    if llm_pages:
        tasks = [
            describe_image_with_llm(page["ocr_image"], page["ocr_text"])
            for page in llm_pages
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for page, desc in zip(llm_pages, results):
            if isinstance(desc, Exception):
                page["llm_refine_error"] = str(desc)
                continue
            if desc and desc.strip():
                page["llm_refine_desc"] = desc.strip()

    # 组装最终文本（优先级：LLM 精修 > OCR > 文字层+表格）
    texts: list[str] = []
    for page in pages_info:
        if page.get("llm_refine_desc"):
            texts.append(page["llm_refine_desc"])
        elif page.get("ocr_text"):
            texts.append(page["ocr_text"])
        else:
            texts.append(page["page_content"])

    return texts if texts else [f"[PDF 解析失败或无内容] {file_path}"]