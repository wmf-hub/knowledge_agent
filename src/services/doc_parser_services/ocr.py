"""OCR 组件：页面转图、PaddleOCR 识别、LLM 多模态视觉描述（PDF / 图片共用）"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.config.settings import settings

OCR_LOW_QUALITY_THRESHOLD = 0.6
IMAGE_EXTENSIONS = (".img", ".png", ".jpg", ".jpeg")

DESCRIBE_SYSTEM_PROMPT = (
    "你是专业的文档分析助手。用户会给你一张文档页面图片，"
    "以及该页面经过 OCR 识别出的文本（可能含错字、排版错乱）。\n"
    "请完成：\n"
    "1. 对照图片修正 OCR 文本中的错字、乱序、漏字；\n"
    "2. 还原表格结构（用 Markdown 表格）；\n"
    "3. 如果页面含图表，补充描述图表表达的含义；\n"
    "4. 如果 OCR 文本为空或严重错误，直接以图片内容为准转录。\n"
    "输出应为干净的 Markdown 文本结果，不要加额外解释。"
)


def _get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        temperature=0,
    )


def render_page_to_image(file_path: str, page_no: int, dpi: int = 150):
    """单页转图"""
    from pdf2image import convert_from_path

    images = convert_from_path(
        file_path, dpi=dpi, first_page=page_no, last_page=page_no
    )
    if not images:
        raise RuntimeError(f"{file_path} 第 {page_no} 页转图失败")
    return images[0]


def ocr(file_path: str, page_no: int | None = None) -> tuple[str, float, Any]:
    """PaddleOCR，返回 (文本, 平均置信度, 原图)。非图片文件需传页码参数 page_no"""
    import numpy as np
    from PIL import Image
    from paddleocr import PaddleOCR

    if not file_path.lower().endswith(IMAGE_EXTENSIONS):
        if page_no is None:
            raise ValueError(f"【非图片文件 {file_path} 转图时未传页码参数 page_no】")
        image = render_page_to_image(file_path, page_no, 200)
    else:
        image = Image.open(file_path)

    img = image.convert("L")   # 灰度提升识别率
    img_np = np.array(img)

    engine = PaddleOCR(use_angle_cls=True, lang="ch", show_log=False)
    result = engine.ocr(img_np, cls=True)
    if not result or not result[0]:
        return "", 0.0, image

    lines, confs = [], []
    for line in result[0]:
        text, conf = line[1]
        lines.append(text)
        confs.append(conf)
    avg_conf = sum(confs) / len(confs) if confs else 0.0
    return "\n".join(lines), avg_conf, image


async def describe_image_with_llm(image: Any, ocr_text: str = "") -> str:
    """调用 LLM 多模态能力描述图片的内容，或加入 OCR 文本进行细化"""
    import base64
    import io

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    messages = [
        SystemMessage(content=DESCRIBE_SYSTEM_PROMPT),
        HumanMessage(content=[
            {"type": "text", "text": "请处理这张图片："},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            {"type": "text", "text": f"\n【OCR 文本如下】\n{ocr_text}\n"},
        ]),
    ]
    resp = await _get_llm().ainvoke(messages)
    return resp.content