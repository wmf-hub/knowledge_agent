"""表格解析组件：CSV / Excel → 结构化 Markdown 表格文本"""

from __future__ import annotations

import os
from typing import Any


async def parse_table(file_path: str) -> list[str]:
    """表格解析入口：CSV 走 csv 模块，Excel 走 openpyxl / xlrd"""
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == ".csv":
            return parse_csv(file_path)
        return parse_excel(file_path)
    except Exception as e:
        return [f"[表格解析失败] {file_path}: {e}"]


def parse_csv(file_path: str) -> list[str]:
    """CSV → Markdown 表格文本，自动探测编码与分隔符"""
    import csv
    import io

    text = _read_text_with_fallback(file_path)
    try:
        dialect = csv.Sniffer().sniff(text[:4096], delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel

    rows = [_clean_row(r) for r in csv.reader(io.StringIO(text), dialect)]
    rows = [r for r in rows if any(r)]
    if not rows:
        return [f"[空 CSV] {os.path.basename(file_path)}"]

    headers = _normalize_headers(rows[0])
    return _table_to_texts(headers, rows[1:], title=f"文件: {os.path.basename(file_path)}")


def parse_excel(file_path: str) -> list[str]:
    """Excel → Markdown 表格文本，按工作表逐个转换"""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".xls":
        return _parse_xls(file_path)

    import openpyxl

    wb = openpyxl.load_workbook(file_path, read_only=True, data_only=True)
    try:
        texts: list[str] = []
        for sheet in wb.worksheets:
            rows = [
                [_clean_cell(c) for c in row]
                for row in sheet.iter_rows(values_only=True)
            ]
            rows = [r for r in rows if any(r)]
            if not rows:
                continue
            headers = _normalize_headers(rows[0])
            texts.extend(_table_to_texts(headers, rows[1:], title=f"工作表: {sheet.title}"))
        return texts or [f"[空 Excel] {os.path.basename(file_path)}"]
    finally:
        wb.close()


def _parse_xls(file_path: str) -> list[str]:
    """旧版 .xls 解析（openpyxl 不支持，需 xlrd）"""
    try:
        import xlrd
    except ImportError as e:
        raise RuntimeError("解析 .xls 需要 xlrd（pip install xlrd），或先将文件另存为 .xlsx") from e

    book = xlrd.open_workbook(file_path)
    texts: list[str] = []
    for sheet in book.sheets():
        rows = [
            [_clean_cell(sheet.cell_value(r, c)) for c in range(sheet.ncols)]
            for r in range(sheet.nrows)
        ]
        rows = [r for r in rows if any(r)]
        if not rows:
            continue
        headers = _normalize_headers(rows[0])
        texts.extend(_table_to_texts(headers, rows[1:], title=f"工作表: {sheet.name}"))
    return texts or [f"[空 Excel] {os.path.basename(file_path)}"]


def _read_text_with_fallback(file_path: str) -> str:
    """按常见编码依次尝试解码，避免中文 CSV 因编码报错"""
    with open(file_path, "rb") as f:
        raw = f.read()
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16")
    for enc in ("utf-8-sig", "utf-8", "gb18030", "big5"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _clean_cell(value: Any) -> str:
    """单元格清洗：空值→空串、压缩换行/空白、数值与日期格式化、转义竖线"""
    from datetime import date, datetime

    if value is None:
        return ""
    if isinstance(value, datetime):
        if value.hour or value.minute or value.second:
            value = value.strftime("%Y-%m-%d %H:%M:%S")
        else:
            value = value.strftime("%Y-%m-%d")
    elif isinstance(value, date):
        value = value.isoformat()
    elif isinstance(value, float) and value.is_integer():
        value = int(value)

    text = " ".join(str(value).replace("\t", " ").split())
    return text.replace("|", "\\|")


def _clean_row(row: list[Any]) -> list[str]:
    """整行清洗，并去掉行尾的空单元格"""
    cells = [_clean_cell(c) for c in row]
    while cells and not cells[-1]:
        cells.pop()
    return cells


def _normalize_headers(raw: list[str]) -> list[str]:
    """表头补全与去重：空表头补“列N”，重名追加序号"""
    headers: list[str] = []
    seen: dict[str, int] = {}
    for i, h in enumerate(raw):
        name = h.strip() or f"列{i + 1}"
        if name in seen:
            seen[name] += 1
            name = f"{name}({seen[name]})"
        else:
            seen[name] = 0
        headers.append(name)
    return headers


def _table_to_texts(
    headers: list[str], rows: list[list[str]], title: str = ""
) -> list[str]:
    """表头 + 数据行 → 结构化 Markdown 表格文本（此处不分块，交由 chunking 组件）"""
    width = max([len(headers)] + [len(r) for r in rows])
    if width > len(headers):
        headers = headers + [f"列{j + 1}" for j in range(len(headers), width)]
    if not headers:
        return []

    lines: list[str] = []
    if title:
        lines.append(title)
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        cells = [row[j] if j < len(row) else "" for j in range(len(headers))]
        lines.append("| " + " | ".join(cells) + " |")
    return ["\n".join(lines)]