"""图片解析组件：OCR + LLM 视觉理解"""

from __future__ import annotations

from services.doc_parser_services.ocr import OCR_LOW_QUALITY_THRESHOLD, describe_image_with_llm, ocr


async def parse_image(file_path: str) -> list[str]:
    """图片解析: OCR 识别，置信度不足时交由 LLM 视觉理解"""
    try:
        ocr_text, avg_conf, image = ocr(file_path)
        if avg_conf < OCR_LOW_QUALITY_THRESHOLD:
            return [await describe_image_with_llm(image, ocr_text)]
        return [ocr_text]
    except Exception as e:
        return [f"[图片解析失败] {file_path}: {e}"]