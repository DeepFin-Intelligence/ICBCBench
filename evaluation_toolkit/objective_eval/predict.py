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
import hashlib
import tempfile
import threading
from urllib.parse import urlparse

import requests
import pandas as pd
import pymupdf
from openai import AsyncOpenAI
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio
from dotenv import load_dotenv

from evaluation_toolkit import load_local_dataset
from evaluation_toolkit.model_clients import *
from evaluation_toolkit.DR_clients import create_dr_client
from evaluation_toolkit.objective_eval.metrics import validate_questions
from evaluation_toolkit.dataset_loading import add_hf_dataset_arguments, load_hf_dataset

from concurrent.futures import ThreadPoolExecutor, as_completed


client = OPENROUTER

URL_ATTACHMENT_PROCESSING = "pdf-page-png-v1"
PDF_PAGE_RENDER_SCALE = 2
PDF_DOWNLOAD_TIMEOUT = 30
TEXT_ONLY_DR_MODELS = {"qwen-deep-research", "Qwen-deep-research", "tavily-research"}

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts.json"), "r", encoding='utf-8') as f:
    prompt_templates = json.load(f)

def selected_dataset_path(options):
    return getattr(options, "local_dataset", None) or getattr(options, "_hf_dataset_path", None)


def resolve_attachment_path(file_path, dataset_path=None):
    """Resolve attachment path relative to the dataset directory."""
    if os.path.isabs(file_path):
        return file_path
    if dataset_path:
        # HF snapshot files are symlinks to blobs; attachments belong beside
        # the snapshot JSON, rather than beside its resolved blob target.
        return str(Path(dataset_path).absolute().parent.resolve() / file_path)
    # The bundled dataset and attachments are in data/.
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


def url_attachment_reference(attachment):
    """Validate a URL reference without downloading third-party content."""
    source_url = attachment.get("source_url")
    if not isinstance(source_url, str) or not source_url.strip():
        raise ValueError("URL attachment requires a nonempty source_url")
    parsed = urlparse(source_url)
    if parsed.scheme not in ("https", "http") or not parsed.netloc:
        raise ValueError(f"Attachment source_url must be an HTTP(S) URL: {source_url}")
    if attachment.get("mime_type") != "application/pdf":
        raise ValueError("URL attachments currently require mime_type application/pdf")
    pdf_page = attachment.get("pdf_page")
    if isinstance(pdf_page, bool) or not isinstance(pdf_page, int) or pdf_page < 1:
        raise ValueError("URL PDF attachment pdf_page must be a positive 1-based integer")
    return {
        "source_url": source_url,
        "mime_type": "application/pdf",
        "pdf_page": pdf_page,
        "printed_page": attachment.get("printed_page"),
        "description": attachment.get("description") or "",
    }


def load_remote_pdf_page(source_url, pdf_page):
    """Download a PDF in memory and render only its specified 1-based page."""
    if isinstance(pdf_page, bool) or not isinstance(pdf_page, int) or pdf_page < 1:
        raise ValueError("URL PDF attachment pdf_page must be a positive 1-based integer")
    try:
        response = requests.get(
            source_url, headers={"User-Agent": "ICBCBench/1.0"},
            timeout=PDF_DOWNLOAD_TIMEOUT,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"Failed to download required PDF attachment {source_url}: {exc}") from exc
    pdf_bytes = response.content
    if not isinstance(pdf_bytes, bytes) or b"%PDF-" not in pdf_bytes[:1024]:
        raise ValueError(f"Attachment source did not return a PDF: {source_url}")
    try:
        document = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        raise ValueError(f"Failed to open PDF attachment {source_url}: {exc}") from exc
    with document:
        if not document.is_pdf or document.needs_pass:
            raise ValueError(f"Attachment source must be an unencrypted PDF: {source_url}")
        if pdf_page > document.page_count:
            raise ValueError(
                f"PDF attachment page {pdf_page} is out of range for "
                f"{source_url} ({document.page_count} pages)"
            )
        try:
            page = document.load_page(pdf_page - 1)
            pixmap = page.get_pixmap(
                matrix=pymupdf.Matrix(PDF_PAGE_RENDER_SCALE, PDF_PAGE_RENDER_SCALE),
                alpha=False,
            )
            return pixmap.tobytes("png")
        except Exception as exc:
            raise ValueError(f"Failed to render PDF page {pdf_page} from {source_url}: {exc}") from exc


def url_attachment_annotation(reference):
    page_label = f"PDF page {reference['pdf_page']} (1-based)"
    if reference.get("printed_page") is not None:
        page_label += f"; printed page {reference['printed_page']}"
    description = reference.get("description") or "PDF page attachment"
    return f"[Source attachment: {description}\nSource URL: {reference['source_url']}\n{page_label}]"


def extract_pdf_text(pdf_path, max_pages=None):
    """Extract text from PDF using pymupdf, preserving tables as Markdown."""
    try:
        text_parts = []
        with pymupdf.open(pdf_path) as doc:
            for i, page in enumerate(doc):
                if max_pages is not None and i >= max_pages:
                    text_parts.append(f"\n[PDF truncated after {max_pages} pages]\n")
                    break

                # Find tables on this page
                try:
                    tables = page.find_tables().tables
                except Exception:
                    tables = []
                table_bboxes = []
                table_markdowns = []
                for idx, t in enumerate(tables):
                    try:
                        df = t.to_pandas()
                        if not df.empty:
                            df = df.fillna('')
                            md = df.to_markdown(index=False)
                            table_markdowns.append((t.bbox[1], f"\n[Table {idx+1}]\n{md}\n"))
                            table_bboxes.append(t.bbox)
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
                if all_elements:
                    text_parts.append(f"\n--- Page {i+1} ---\n")
                for _, text, kind in all_elements:
                    text_parts.append(text)

        return "\n".join(text_parts)
    except Exception as e:
        print(f"Failed to extract PDF text from {pdf_path}: {e}")
        return ""


def extract_excel_text(excel_path, max_rows=None):
    """Read Excel and convert to Markdown tables."""
    try:
        text_parts = []
        xls = pd.ExcelFile(excel_path)
        for sheet_name in xls.sheet_names:
            df = pd.read_excel(excel_path, sheet_name=sheet_name, header=None)
            if df.empty or not df.notna().any().any():
                continue
            text_parts.append(f"\n--- Sheet: {sheet_name} ---\n")
            display_df = df if max_rows is None else df.head(max_rows)
            display_df = display_df.fillna('')
            text_parts.append(display_df.to_markdown(index=False))
            if max_rows is not None and len(df) > max_rows:
                text_parts.append(f"\n[Excel truncated after {max_rows} rows, total {len(df)} rows]\n")
        return "\n".join(text_parts)
    except Exception as e:
        print(f"Failed to extract Excel text from {excel_path}: {e}")
        return ""


def build_attachment_content(attachment_list, dataset_path=None, materialize_images=True):
    """Build multimodal content blocks from attachments.

    Returns a tuple (content_blocks, text_annotations) where:
    - content_blocks: list for OpenAI-compatible multimodal API
    - text_annotations: string for DR models that only accept text
    URL PDF references render the selected page in memory unless
    materialize_images is False, in which case their source annotation is kept.
    """
    content_blocks = []
    text_annotations = []

    for att in attachment_list:
        # Source references take precedence over legacy bundled file paths.
        # Only the selected page is materialized, never the entire source PDF.
        if att.get("source_url") is not None:
            reference = url_attachment_reference(att)
            text_annotations.append(url_attachment_annotation(reference))
            if materialize_images:
                png = load_remote_pdf_page(reference["source_url"], reference["pdf_page"])
                b64 = base64.b64encode(png).decode("ascii")
                content_blocks.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                })
            continue

        original_name = att.get("original_filename", "attachment")
        mime_type = att.get("mime_type", "")
        rel_path = att.get("file_path", "")
        abs_path = resolve_attachment_path(rel_path, dataset_path)

        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"Attachment not found: {abs_path}")

        # Image types -> base64 image_url
        if mime_type.startswith("image/") or any(original_name.lower().endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"]):
            if not materialize_images:
                raise ValueError(
                    f"Text-only research requires a source URL for image attachment: {original_name}"
                )
            b64 = load_image_as_base64(abs_path)
            if not b64:
                raise ValueError(f"Failed to read required image attachment: {abs_path}")
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
            if not pdf_text.strip():
                raise ValueError(f"Failed to extract required PDF attachment: {abs_path}")
            if pdf_text:
                snippet = f"[PDF attachment: {original_name}]\n{pdf_text}\n[/PDF: {original_name}]"
                content_blocks.append({"type": "text", "text": snippet})
                text_annotations.append(snippet)

        # Excel -> extract text
        elif mime_type in ("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "application/vnd.ms-excel") or any(original_name.lower().endswith(ext) for ext in [".xlsx", ".xls"]):
            excel_text = extract_excel_text(abs_path)
            if not excel_text.strip():
                raise ValueError(f"Failed to extract required Excel attachment: {abs_path}")
            if excel_text:
                snippet = f"[Excel attachment: {original_name}]\n{excel_text}\n[/Excel: {original_name}]"
                content_blocks.append({"type": "text", "text": snippet})
                text_annotations.append(snippet)

        else:
            raise ValueError(f"Unsupported required attachment type '{mime_type}' for {original_name}")

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
        content_blocks, text_annotations = build_attachment_content(
            attachment_list, selected_dataset_path(args),
            materialize_images=not for_dr_model,
        )

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
            if any(att.get("source_url") is not None for att in attachment_list):
                user_text = f"{user_text}\n\n[Attachments]\n{text_annotations}"
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
    fingerprint = request_fingerprint(question, args)
    source_references = any(
        attachment.get("source_url") is not None
        for attachment in question.get("attachment") or []
    )
    messages = format_message(
        question, for_dr_model=args.model in TEXT_ONLY_DR_MODELS and source_references,
    )
    dr_client = create_dr_client(args.model)
    if dr_client is not None:
        dr_client.cancel_event = getattr(args, "research_cancel_event", None)

    def request(messages):
        if dr_client is not None:
            response = dr_client.run_research(
                system_prompt=messages[0]["content"], dr_task="",
                multimodal_content=messages[1]["content"]
            )
            return response.get("content"), response.get("usage")
        response = client.chat.completions.create(
            model=resolve_openrouter_model(args.model),
            temperature=args.temperature if "o1" not in args.model else None,
            max_tokens=args.max_completion_tokens, messages=messages, stream=False
        )
        require_complete_chat(response)
        usage = response.usage.model_dump() if response.usage else None
        return response.choices[0].message.content, usage

    content, tokens = request(messages)
    if not isinstance(content, str) or not content.strip():
        raise ValueError(f"Question {question['id']} returned empty content")
    return question["id"], content, tokens, fingerprint


def request_fingerprint(question, options):
    attachment_hashes = []
    for attachment in question.get("attachment") or []:
        if attachment.get("source_url") is not None:
            reference = url_attachment_reference(attachment)
            attachment_hashes.append({
                "reference": reference,
                "processing": URL_ATTACHMENT_PROCESSING,
                "render_scale": PDF_PAGE_RENDER_SCALE,
            })
            continue
        path = Path(resolve_attachment_path(attachment.get("file_path", ""), selected_dataset_path(options)))
        attachment_hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
    payload = {"question": question, "attachments": attachment_hashes,
               "model": resolve_openrouter_model(options.model),
               "temperature": options.temperature,
               "max_completion_tokens": options.max_completion_tokens,
               "answer_prompts": prompt_templates["answer"], "schema_version": 3}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def write_json_atomic(path, value):
    """Replace a cache only after the complete JSON has been written."""
    path = Path(path)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary_path = handle.name
        try:
            json.dump(value, handle, indent=4, ensure_ascii=False)
        except BaseException:
            os.unlink(temporary_path)
            raise
    try:
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def load_questions(options, required_fields=("id", "language", "question")):
    if options.local_dataset:
        options._hf_dataset_path = None
        with open(options.local_dataset, encoding="utf-8-sig") as handle:
            dataset = json.load(handle)
    else:
        dataset = load_hf_dataset(options, "objective")
    if isinstance(dataset, dict):
        if not dataset or any(not isinstance(column, list) for column in dataset.values()):
            raise ValueError("Column datasets must contain lists of equal length")
        if len({len(column) for column in dataset.values()}) != 1:
            raise ValueError("Dataset columns have different lengths; questions would be dropped")
        dataset = [dict(zip(dataset.keys(), values)) for values in zip(*dataset.values())]
    return validate_questions(dataset, required_fields)

def attempt_all(questions, on_result=None):
    results = []
    # Use thread pool to control concurrency (num_workers)
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        # Submit all tasks
        future_to_q = {executor.submit(attempt_question, q): q for q in questions}

        # Show progress with tqdm
        try:
            for future in tqdm(as_completed(future_to_q), total=len(questions)):
                try:
                    res = future.result()
                except Exception as exc:
                    print(f"Generated an exception: {exc}")
                    results.append(None)
                else:
                    if res:
                        results.append(res)
                        if on_result is not None:
                            on_result(res)
        except BaseException:
            event = getattr(args, "research_cancel_event", None)
            if event is not None:
                event.set()
            for future in future_to_q:
                future.cancel()
            raise

    return results

def _run(args):
    if args.num_workers < 1:
        raise ValueError("num_workers must be 1 or greater")
    if args.max_samples is not None and args.max_samples < 1:
        raise ValueError("max_samples must be 1 or greater")
    if not args.local_dataset and not getattr(args, "dataset", None):
        raise ValueError("Provide --local_dataset or --dataset")

    questions = load_questions(args)
    current_questions = {question["id"]: question for question in questions}

    # If max_samples is set, limit the number of questions
    if args.max_samples:
        questions = questions[:args.max_samples]

    project_root = Path(__file__).resolve().parent.parent.parent
    output_dir = getattr(args, "output_dir", None) or os.path.normpath(os.path.join(project_root, "eval_result", "objective_eval", "prediction_results"))
    os.makedirs(output_dir, exist_ok=True)

    output_filepath = f"{output_dir}/{os.path.basename(args.model)}.json"

    # load only questions without responses
    if os.path.exists(output_filepath):
        with open(output_filepath, "r", encoding='utf-8') as f:
            predictions = json.load(f)
        if not isinstance(predictions, dict):
            raise ValueError("Prediction cache must be a JSON object")
        for previous in predictions.values():
            if (isinstance(previous, dict) and previous.get("model") is not None
                    and resolve_openrouter_model(previous["model"]) != resolve_openrouter_model(args.model)):
                raise ValueError("Prediction output filename is already used by a different model; select a separate output_dir")
        reusable_predictions = {}
        for task_id, previous in predictions.items():
            question = current_questions.get(task_id)
            if question is None:
                continue
            try:
                if (isinstance(previous, dict) and isinstance(previous.get("response"), str)
                        and previous["response"].strip()
                        and previous.get("request_fingerprint") == request_fingerprint(question, args)):
                    reusable_predictions[task_id] = previous
            except (OSError, ValueError):
                pass
        predictions = reusable_predictions
        pending = []
        for question in questions:
            if question["id"] not in predictions:
                pending.append(question)
        questions = pending
    else:
        predictions = {}

    def save_result(result):
        unique_id, response, usage, fingerprint = result
        predictions[unique_id] = {
            "model": args.model,
            "response": response,
            "usage": usage,
            "request_fingerprint": fingerprint,
        }
        write_json_atomic(output_filepath, predictions)

    # Remove invalid cache records before requesting replacements. Each completed
    # response is then committed immediately, including before an interruption.
    write_json_atomic(output_filepath, predictions)
    args.research_cancel_event = threading.Event()
    try:
        results = attempt_all(questions, on_result=save_result)
    except BaseException:
        write_json_atomic(output_filepath, predictions)
        raise

    failed = len(questions) - sum(result is not None for result in results)
    if failed:
        print(f"Prediction incomplete: {failed} question(s) failed. Saved successful responses to {output_filepath}.")
    return 1 if failed else 0


def main(options):
    globals()["args"] = options
    try:
        return _run(options)
    except Exception as error:
        print(f"Prediction failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--dataset", type=str, help="Hugging Face dataset")
    sources.add_argument("--local_dataset", type=str, default=None, help="Local JSON dataset file path")
    add_hf_dataset_arguments(parser)
    parser.add_argument("--model", type=str, required=True, help="Model Endpoint Name")
    parser.add_argument("--max_completion_tokens", type=positive_int, default=None,
                        help="Optional. Limit completion tokens. It can be used to avoid model collapse.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for sampling.")
    parser.add_argument("--num_workers", type=positive_int, default=5,
                        help="Async semaphore size. This depends on your rate limit. Only work in the async version.")
    parser.add_argument("--max_samples", type=positive_int, default=None, help="Optional. Limit evaluation to first N samples")
    parser.add_argument("--output_dir", type=str, default=None, help="Override prediction output directory")
    args = parser.parse_args()
    sys.exit(main(args))
