import base64
import io
import json
import os

import requests
import PyPDF2
import pymupdf
import re

def safe_json_loads(text: str):
    """
    安全解析LLM的JSON响应。
    处理包含Markdown格式、前后带有无关文本等常见LLM输出问题。
    """
    if not text or not isinstance(text, str):
        return None

    text = text.strip()

    # 1. 最理想情况：直接就是完美的 JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass  # 继续尝试下面的提取逻辑

    # 2. 尝试提取 Markdown 代码块中的 JSON (如 ```json ... ```)
    # 使用 re.DOTALL 使 . 能够匹配换行符
    match = re.search(r'```(?:json)?(.*?)```', text, re.DOTALL | re.IGNORECASE)
    if match:
        extracted_text = match.group(1).strip()
        try:
            return json.loads(extracted_text)
        except json.JSONDecodeError:
            pass

    # 按出现顺序解析完整容器，避免把数组中的单个对象误当成整个结果。
    # raw_decode 同时正确处理字符串里的括号和 JSON 之后的说明文字。
    decoder = json.JSONDecoder()
    # 遇到明显的 JSON 容器后，截断或语法错误必须失败；不能降级为其内部对象。
    candidate = re.search(r'\[(?=\s*(?:[\[{"\d\]-]|true\b|false\b|null\b))|\{(?=\s*(?:"|}))', text)
    if candidate:
        try:
            result, _ = decoder.raw_decode(text, candidate.start())
            return result
        except json.JSONDecodeError:
            pass

    # 4. 如果所有尝试都失败了，抛出异常或返回 None（这里选择记录日志并返回 None）
    print(f"[Warning] Failed to safely parse JSON. Raw text snippet: \n{text[:500]}...")
    return None

def load_local_dataset(file_path: str):
    """
    加载本地 json 数据集
    """
    with open(file_path, "r", encoding="utf-8") as f:
        local_data = json.load(f)

    if isinstance(local_data, list):
        if not all(isinstance(item, dict) for item in local_data):
            raise ValueError("Dataset rows must be JSON objects")
        # 保持列顺序一致；缺失的可选字段为 None，空数据集返回空列映射。
        all_keys = dict.fromkeys(key for item in local_data for key in item)
        return {key: [item.get(key) for item in local_data] for key in all_keys}
    if isinstance(local_data, dict):
        # 列式数据集以及 FACT 使用的 id -> report 映射都可读取。
        if any(isinstance(value, list) for value in local_data.values()):
            if not all(isinstance(value, list) for value in local_data.values()):
                raise ValueError("Dataset columns must all be lists")
            if len({len(value) for value in local_data.values()}) > 1:
                raise ValueError("Dataset columns must have equal lengths")
        return local_data
    raise ValueError("Dataset must be a JSON array or object")


def fix_pdf_newlines(text):
    # 逻辑：如果换行符前面不是句号、感叹号、问号或空格，就说明这段话没讲完
    # 我们把这个换行符替换成空格
    # 改进正则表达式，确保换行符前没有标点符号才替换为空格
    fixed_text = re.sub(r'(?<![.!?，。！？\s])\n', '', text)

    # 清理多余的空格
    fixed_text = re.sub(r' +', ' ', fixed_text)
    return fixed_text


def convert_page_to_images(pdf_path: str, page_id: int):
    """
    将PDF文件的某一页转换为PNG图像

    Args:
        pdf_path: PDF文件路径
        page_id: 页码
    """
    doc = pymupdf.open(pdf_path)

    page = doc[page_id-1]
    pix = page.get_pixmap()
    png_bytes = pix.tobytes("png")
    base64_image = base64.b64encode(png_bytes).decode('utf-8')

    return base64_image


def extract_text_from_page(file_path, page_id: int):
    """
    使用pymupdf从PDF文件提取某页内容
    """
    try:
        doc = pymupdf.open(file_path)

        page = doc[page_id-1]
        text = page.get_text()
        fixed_text = fix_pdf_newlines(text)

        return fixed_text.strip()
    
    except Exception as e:
        print(f"Error extracting text from PDF using pymupdf {file_path}: {e}")
        return f"无法从PDF文件提取内容: {str(e)}"


def _extract_text_from_pdf_url(pdf_url):
    """
    从PDF URL下载并提取文本内容
    """
    try:
        # 下载PDF文件
        response = requests.get(pdf_url, timeout=30)
        response.raise_for_status()

        # 使用PyPDF2提取文本
        pdf_file = io.BytesIO(response.content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)

        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

        if not text.strip():
            return "PDF文件内容为空或无法提取文本"

        return text.strip()

    except Exception as e:
        print(f"Error extracting text from PDF URL {pdf_url}: {e}")
        return f"无法从PDF链接提取内容: {str(e)}"

def _extract_text_from_local_pdf(file_path):
    """
    从本地PDF文件提取文本内容
    """
    try:
        # 检查文件是否存在
        if not os.path.exists(file_path):
            return f"本地PDF文件不存在: {file_path}"

        # 使用PyPDF2提取文本
        with open(file_path, "rb") as f:
            pdf_reader = PyPDF2.PdfReader(f)

            text_lines = []
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    lines = page_text.split("\n")
                    text_lines.extend(lines)

            if not text_lines:
                return "PDF文件内容为空或无法提取文本"

            # 清除首位空行
            cleaned_lines = [line.strip() for line in text_lines if line.strip()]
            cleaned_text = "".join(cleaned_lines)

            if not cleaned_text.strip():
                return "PDF文件内容为空或无法提取文本"

            return cleaned_text.strip()

    except Exception as e:
        print(f"Error extracting text from local PDF {file_path}: {e}")
        return f"无法从本地PDF文件提取内容: {str(e)}"



def extract_text_from_pdf(pdf_path):
    """
    从 pdf 链接或本地地址提取内容文本
    """
    if is_url(pdf_path):
        # 网络PDF链接
        print(f"从PDF网络链接提取内容: {pdf_path}")
        pdf_content = _extract_text_from_pdf_url(pdf_path)
        content = f"Reference Report: {pdf_content}"
    elif is_local_file(pdf_path):
        # 本地PDF文件
        print(f"从本地PDF文件提取内容: {pdf_path}")
        pdf_content = _extract_text_from_local_pdf(pdf_path)
        content = f"Reference Report: {pdf_content}"
    else:
        # 可能是相对路径或不存在的文件
        if os.path.exists(pdf_path) or os.path.exists(os.path.join(os.getcwd(), pdf_path)):
            actual_path = pdf_path if os.path.exists(pdf_path) else os.path.join(os.getcwd(), pdf_path)
            pdf_content = _extract_text_from_local_pdf(actual_path)
            content = f"Reference Report: {pdf_content}"
        else:
            content = f"Reference Report: {pdf_path} (PDF文件不存在)"

    return content.strip('\n')

def is_pdf_file(text):
    """检查文本是否为PDF文件（本地路径或URL）"""
    # 检查是否是PDF文件扩展名
    if isinstance(text, str) and text.lower().endswith('.pdf'):
        return True
    return False

def is_url(text):
    """检查文本是否为URL"""
    if not isinstance(text, str):
        return False
    return text.startswith(('http://', 'https://'))

def is_local_file(text):
    """检查文本是否为本地文件路径"""
    if not isinstance(text, str):
        return False
    # 基本的本地文件路径检查
    return os.path.exists(text) and os.path.isfile(text)


def get_project_root():
    """返回项目根目录（evaluation_toolkit/ 的父目录）。"""
    from pathlib import Path
    return str(Path(__file__).resolve().parent.parent)


def get_eval_data_dir():
    """
    返回评估数据根目录。
    优先读取 EVAL_DATA_DIR 环境变量，否则默认使用项目根目录下的 data/。
    """
    project_root = get_project_root()
    from pathlib import Path
    return os.environ.get("EVAL_DATA_DIR", str(Path(project_root) / "data"))


if __name__ == "__main__":

    # 测试
    pdf_path = "../report_collect_and_eval/capital_markets/胡宇洲-平安资管-权益研究室/20240127-华泰证券-东山精密（002384.SZ）：消费电子基本盘稳固，汽车再造东山.pdf"

    _extract_text_from_local_pdf(pdf_path)
