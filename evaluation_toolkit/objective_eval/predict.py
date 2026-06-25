import sys
from pathlib import Path
_p = Path(__file__).resolve()
while _p.name != "evaluation_toolkit" and _p.parent != _p:
    _p = _p.parent
if _p.name == "evaluation_toolkit":
    _root = _p.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

import base64
import os
import json
import argparse
import asyncio
import sys
import io

import requests
import pandas as pd
import fitz  # pymupdf
from datasets import load_dataset
from openai import AsyncOpenAI
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio
from dotenv import load_dotenv

from evaluation_toolkit import load_local_dataset
from evaluation_toolkit.model_clients import *
from evaluation_toolkit.DR_clients.QwenDR import TongyiDeepResearch, QwenDeepResearch
from evaluation_toolkit.DR_clients.GeminiDR import GeminiDeepResearch
from evaluation_toolkit.DR_clients.TavilyResearch import TavilyDeepResearch
from evaluation_toolkit.DR_clients.OpenaiDR import OpenaiDeepResearch

from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()

client = DMX

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts.json"), "r", encoding='utf-8') as f:
    prompt_templates = json.load(f)

def resolve_attachment_path(file_path):
    """Resolve attachment path relative to the dataset directory."""
    if os.path.isabs(file_path):
        return file_path
    # predict.py is in evaluation_toolkit/objective_eval/
    # attachments are in data/
    project_root = Path(__file__).resolve().parent.parent.parent
    return os.path.normpath(os.path.join(project_root, "data", file_path))


def load_image_as_base64(image_path):
    # local image or online image
    if os.path.isfile(image_path):
        # local image
        try:
            with open(image_path, "rb") as f:
                return base64.b64encode(f.read()).decode('utf-8')
        except Exception as e:
            print("Failed to read local image:", e)
            return ""
    else:
        # imitate an explorer
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
        }

        try:
            response = requests.get(image_path, headers=headers)
            response.raise_for_status()
            return base64.b64encode(response.content).decode('utf-8')
        except Exception as e:
            print("Failed to download image:", e)
            return ""


def extract_pdf_text(pdf_path, max_pages=10):
    """Extract text from PDF using pymupdf, preserving tables as Markdown."""
    try:
        text_parts = []
        with fitz.open(pdf_path) as doc:
            for i, page in enumerate(doc):
                if i >= max_pages:
                    text_parts.append(f"\n[PDF truncated after {max_pages} pages]\n")
                    break

                text_parts.append(f"\n--- Page {i+1} ---\n")

                # Find tables on this page
                tables = page.find_tables()
                table_bboxes = [t.bbox for t in tables.tables]
                table_markdowns = []
                for idx, t in enumerate(tables.tables):
                    try:
                        df = t.to_pandas()
                        if not df.empty:
                            df = df.fillna('')
                            md = df.to_markdown(index=False)
                            table_markdowns.append((t.bbox[1], f"\n[Table {idx+1}]\n{md}\n"))
                    except Exception:
                        pass

                # Extract non-table text blocks
                blocks = page.get_text("blocks")
                non_table_blocks = []
                for b in blocks:
                    bbox = b[:4]
                    block_text = b[4].strip()
                    if not block_text:
                        continue
                    # Check if block is inside any table bbox (with small margin)
                    in_table = False
                    for tb in table_bboxes:
                        if (bbox[0] >= tb[0] - 5 and bbox[1] >= tb[1] - 5 and
                            bbox[2] <= tb[2] + 5 and bbox[3] <= tb[3] + 5):
                            in_table = True
                            break
                    if not in_table:
                        non_table_blocks.append((bbox[1], block_text))

                # Merge tables and text blocks, sort by vertical position
                all_elements = [(y, text, "text") for y, text in non_table_blocks]
                all_elements += [(y, text, "table") for y, text in table_markdowns]
                all_elements.sort(key=lambda x: x[0])

                for _, text, kind in all_elements:
                    text_parts.append(text)

        return "\n".join(text_parts)
    except Exception as e:
        print(f"Failed to extract PDF text from {pdf_path}: {e}")
        return ""


def extract_excel_text(excel_path, max_rows=200):
    """Read Excel and convert to Markdown tables."""
    try:
        text_parts = []
        xls = pd.ExcelFile(excel_path)
        for sheet_name in xls.sheet_names:
            df = pd.read_excel(excel_path, sheet_name=sheet_name, header=None)
            text_parts.append(f"\n--- Sheet: {sheet_name} ---\n")
            display_df = df.head(max_rows)
            display_df = display_df.fillna('')
            text_parts.append(display_df.to_markdown(index=False))
            if len(df) > max_rows:
                text_parts.append(f"\n[Excel truncated after {max_rows} rows, total {len(df)} rows]\n")
        return "\n".join(text_parts)
    except Exception as e:
        print(f"Failed to extract Excel text from {excel_path}: {e}")
        return ""


def build_attachment_content(attachment_list):
    """Build multimodal content blocks from attachments.

    Returns a tuple (content_blocks, text_annotations) where:
    - content_blocks: list for OpenAI-compatible multimodal API
    - text_annotations: string for DR models that only accept text
    """
    content_blocks = []
    text_annotations = []

    for att in attachment_list:
        original_name = att.get("original_filename", "attachment")
        mime_type = att.get("mime_type", "")
        rel_path = att.get("file_path", "")
        abs_path = resolve_attachment_path(rel_path)

        if not os.path.exists(abs_path):
            print(f"Attachment not found: {abs_path}")
            continue

        # Image types -> base64 image_url
        if mime_type.startswith("image/") or any(original_name.lower().endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"]):
            b64 = load_image_as_base64(abs_path)
            if b64:
                content_blocks.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type or 'image/png'};base64,{b64}"
                    }
                })
                text_annotations.append(f"[Image attachment: {original_name}]")

        # PDF -> extract text
        elif mime_type == "application/pdf" or original_name.lower().endswith(".pdf"):
            pdf_text = extract_pdf_text(abs_path)
            if pdf_text:
                snippet = f"[PDF attachment: {original_name}]\n{pdf_text}\n[/PDF: {original_name}]"
                content_blocks.append({"type": "text", "text": snippet})
                text_annotations.append(snippet)

        # Excel -> extract text
        elif mime_type in ("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "application/vnd.ms-excel") or any(original_name.lower().endswith(ext) for ext in [".xlsx", ".xls"]):
            excel_text = extract_excel_text(abs_path)
            if excel_text:
                snippet = f"[Excel attachment: {original_name}]\n{excel_text}\n[/Excel: {original_name}]"
                content_blocks.append({"type": "text", "text": snippet})
                text_annotations.append(snippet)

        else:
            print(f"Unsupported attachment type '{mime_type}' for {original_name}, skipping.")

    return content_blocks, "\n\n".join(text_annotations)

def format_message(question, for_dr_model=False):
    language = "zh" if question["language"] in ["zh", "chinese", "中文"] else "en"
    prompt_template = prompt_templates["answer"][language]
    SYSTEM_PROMPT = prompt_template.format(
        ans_format=question.get("answer_format_prompt", ""),
        format_example=question.get("answer_format_example", "")
    )

    user_text = question['question']
    attachment_list = question.get("attachment", [])

    if attachment_list:
        content_blocks, text_annotations = build_attachment_content(attachment_list)

        if for_dr_model:
            # DR models only accept plain text
            if text_annotations:
                user_text = f"{user_text}\n\n[Attachments]\n{text_annotations}"
            messages = [
                {"role": "user", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_text}
            ]
        else:
            # OpenAI-compatible multimodal API
            if content_blocks:
                # First block is the question text, followed by attachments
                content = [{"type": "text", "text": user_text}]
                content.extend(content_blocks)
            else:
                content = user_text
            system_role = "user" if "o1" in args.model else "system"
            messages = [
                {"role": system_role, "content": SYSTEM_PROMPT},
                {"role": "user", "content": content}
            ]
    else:
        system_role = "user" if "o1" in args.model else "system"
        messages = [
            {"role": system_role, "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text}
        ]

    return messages


def attempt_question(question):
    messages = format_message(question)

    DR_models = {
        "gemini-deep-research": GeminiDeepResearch(),
        "tongyi-deepresearch-30b-a3b": TongyiDeepResearch(),
        "Qwen-deep-research": QwenDeepResearch(),
        "tavily-research": TavilyDeepResearch(),
        "o4-mini-deep-research": OpenaiDeepResearch(model_choose=args.model),
        "o3-deep-research": OpenaiDeepResearch(model_choose=args.model)
    }

    Openrouter_mapping = {
        "perplexity-deep-research": "perplexity/sonar-deep-research",
    }

    if args.model in DR_models:
        dr_client = DR_models[args.model]
        # Pass multimodal content to DR client; each client handles it according to its capability
        user_content = messages[1].get("content")
        try:
            response = dr_client.run_research(
                system_prompt=messages[0]["content"],
                dr_task="",
                multimodal_content=user_content
            )
            content = response.get("content")
            tokens = response.get("usage", None)
        except Exception as e:
            error_msg = str(e).lower()
            # Model does not support multimodal format: 404 (no image input) or 400 (param validation error)
            if ("image input" in error_msg or "no endpoints found" in error_msg
                    or "request param validation error" in error_msg
                    or "has invalid field" in error_msg
                    or "content" in error_msg and ("invalid" in error_msg or "validation" in error_msg)):
                print(f"\nModel '{args.model}' does not support multimodal content, retrying with text-only mode...")
                fallback_messages = format_message(question, for_dr_model=True)
                response = dr_client.run_research(
                    system_prompt=fallback_messages[0]["content"],
                    dr_task="",
                    multimodal_content=fallback_messages[1].get("content")
                )
                content = response.get("content")
                tokens = response.get("usage", None)
            else:
                raise
    else:
        content = None
        tokens = None
        try:
            response = client.chat.completions.create(
                model=Openrouter_mapping[args.model] if client in (OPENROUTER, OPENROUTER2) else args.model,
                temperature=args.temperature if "o1" not in args.model else None,
                # max_completion_tokens=args.max_completion_tokens,
                max_tokens=args.max_completion_tokens,
                messages=messages,
                stream=False
            )
            content = response.choices[0].message.content
            tokens = response.usage.model_dump()
        except Exception as e:
            error_msg = str(e)
            error_code = getattr(e, 'status_code', None)

            # If the model does not support multimodal format, auto-fallback to text-only mode
            is_multimodal_error = (
                (error_code == 404 and "image input" in error_msg.lower())
                or (error_code == 400 and ("request param validation error" in error_msg.lower()
                                            or "has invalid field" in error_msg.lower()
                                            or "content" in error_msg.lower() and "invalid" in error_msg.lower()))
            )
            if is_multimodal_error:
                print(f"\nModel '{args.model}' does not support multimodal content, retrying with text-only mode...")
                try:
                    # Re-format messages without image attachments
                    fallback_messages = format_message(question, for_dr_model=True)
                    response = client.chat.completions.create(
                        model=Openrouter_mapping[args.model] if client in (OPENROUTER, OPENROUTER2) else args.model,
                        temperature=args.temperature if "o1" not in args.model else None,
                        max_tokens=args.max_completion_tokens,
                        messages=fallback_messages,
                        stream=False
                    )
                    content = response.choices[0].message.content
                    tokens = response.usage.model_dump()
                except Exception as e2:
                    print("\nRetry failed:", e2)
                    error_code = getattr(e2, 'status_code', None)
                    # Fall through to normal error handling
            else:
                print("\nError:", e)

            # Decide whether to exit based on error type
            if content is None:
                if error_code == 401:
                    print("Fatal error: API authentication failed, please check API key")
                    sys.exit(1)
                elif error_code == 403:
                    print("Fatal error: no permission to access this model, please check permissions")
                    sys.exit(1)
                elif error_code == 404:
                    print(f"Fatal error: model '{args.model}' not found")
                    sys.exit(1)
                elif error_code == 503:
                    print(f"Fatal error: model '{args.model}' not found")
                    sys.exit(1)
                elif error_code == 500:
                    print("Internal server error, skipping this question")
                    return None  # Non-fatal, continue processing other questions
                elif error_code == 429:
                    print("Rate limit hit, consider lowering --num_workers")
                    return None  # Can choose to continue or exit
                else:
                    # Network errors and other issues, continue processing
                    return None

    if content is None:  # failed
        print(f"\nWarning: question {question['id']} returned empty content")
        return None

    return question["id"], content, tokens

def attempt_all(questions):
    results = []
    # Use thread pool to control concurrency (num_workers)
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        # Submit all tasks
        future_to_q = {executor.submit(attempt_question, q): q for q in questions}

        # Show progress with tqdm
        for future in tqdm(as_completed(future_to_q), total=len(questions)):
            try:
                res = future.result()
                if res:
                    results.append(res)
            except Exception as exc:
                print(f"Generated an exception: {exc}")

    return results

def main(args):

    assert args.num_workers >= 1, "num_workers must be 1 or greater"

    if args.local_dataset:  # load from local dataset
        dataset = load_local_dataset(args.local_dataset)
    else:   # load from HuggingFace
        dataset = load_dataset(args.dataset, split="test").to_dict()
        # dataset = {}

    # convert to list of json for async parallelism
    questions = [dict(zip(dataset.keys(), values)) for values in zip(*dataset.values())]

    # If max_samples is set, limit the number of questions
    if args.max_samples:
        questions = questions[:args.max_samples]

    project_root = Path(__file__).resolve().parent.parent.parent
    output_dir = os.path.normpath(os.path.join(project_root, "eval_result", "objective_eval", "prediction_results"))
    os.makedirs(output_dir, exist_ok=True)

    output_filepath = f"{output_dir}/{os.path.basename(args.model)}.json"

    # load only questions without responses
    if os.path.exists(output_filepath):
        with open(output_filepath, "r", encoding='utf-8') as f:
            predictions = json.load(f)
        questions = [q for q in questions if q["id"] not in predictions]
    else:
        predictions = {}

    results = attempt_all(questions)

    # It is allowed to rerun this script multiple times if there are failed API calls
    for result in results:
        if result is None:  # API call failed
            continue
        unique_id, response, usage = result
        predictions[unique_id] = {
            "model": args.model,
            "response": response,
            "usage": usage
        }

    # cache responses
    with open(output_filepath, "w", encoding='utf-8') as f:
        json.dump(predictions, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="HLE HF Dataset")
    parser.add_argument("--local_dataset", type=str, default=None, help="Local JSON dataset file path")
    parser.add_argument("--model", type=str, help="Model Endpoint Name")
    parser.add_argument("--max_completion_tokens", type=int, default=None,
                        help="Optional. Limit completion tokens. It can be used to avoid model collapse.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for sampling.")
    parser.add_argument("--num_workers", type=int, default=5,
                        help="Async semaphore size. This depends on your rate limit. Only work in the async version.")
    parser.add_argument("--max_samples", type=int, default=None, help="Optional. Limit evaluation to first N samples")
    args = parser.parse_args()
    main(args)