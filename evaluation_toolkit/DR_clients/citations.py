"""Preserve provider citations and render only references with a known text mapping."""

import re
from urllib.parse import quote, urlsplit


def json_value(value):
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    elif not isinstance(value, (dict, list, tuple, str, int, float, bool, type(None))) and hasattr(value, "__dict__"):
        value = vars(value)
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    return value


def normalize_citations(raw):
    citations = []
    for item in json_value(raw or []):
        if not isinstance(item, dict):
            continue
        url = item.get("url")
        if not isinstance(url, str) or urlsplit(url).scheme not in ("http", "https") or not urlsplit(url).netloc:
            continue
        citation = {"url": url, "title": item.get("title") or "Source"}
        for field in ("start_index", "end_index", "index_number"):
            if isinstance(item.get(field), int) and not isinstance(item[field], bool):
                citation[field] = item[field]
        if citation not in citations:
            citations.append(citation)
    return citations


def markdown_link(citation, label=None):
    label = str(label or citation.get("title") or "Source").replace("\\", "\\\\").replace("[", "\\[").replace("]", "\\]").replace("\n", " ")
    url = quote(citation["url"], safe=":/?#[]@!$&'*+,;=%-._~")
    return f"[{label}]({url})"


def render_annotations(text, raw, *, byte_offsets=False):
    """Insert each mapped URL at its attributed segment, keeping the claim intact."""
    citations = normalize_citations(raw)
    insertions = {}
    for citation in citations:
        start, end = citation.get("start_index"), citation.get("end_index")
        if start is None or end is None or not 0 <= start <= end:
            continue
        if byte_offsets:
            encoded = text.encode("utf-8")
            if end > len(encoded):
                continue
            try:
                start = len(encoded[:start].decode("utf-8"))
                end = len(encoded[:end].decode("utf-8"))
            except UnicodeDecodeError:
                continue
        if end > len(text) or citation["url"] in text[start:end]:
            continue
        link = markdown_link(citation)
        if link not in insertions.setdefault(end, []):
            insertions[end].append(link)
    for position, links in sorted(insertions.items(), reverse=True):
        text = text[:position] + " " + " ".join(links) + text[position:]
    return text, citations


def render_numbered(text, raw):
    """Use explicit provider reference numbers; never infer IDs from source order."""
    citations = normalize_citations(raw)
    by_number = {}
    ambiguous = set()
    for citation in citations:
        number = citation.get("index_number")
        if number is None or number < 1:
            continue
        if number in by_number and by_number[number]["url"] != citation["url"]:
            ambiguous.add(number)
        by_number[number] = citation

    def replace(match):
        number = int(match.group(1))
        if number not in by_number or number in ambiguous:
            return match.group(0)
        return markdown_link(by_number[number], str(number))

    return re.sub(r"(?<![\\\[])\[(\d+)\](?![(:])", replace, text), citations
