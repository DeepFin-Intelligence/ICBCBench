import json
import os
from typing import Dict, Any
import logging
import tempfile
from urllib.parse import urlparse
from evaluation_toolkit.model_clients import load_project_env
from curl_cffi import requests

load_project_env()

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{file_path}:{line_no}: expected a JSON object")
            data.append(value)
    return data


def write_jsonl(file_path, data):
    """Replace a checkpoint atomically, preserving the previous file on failure."""
    directory = os.path.dirname(os.path.abspath(file_path))
    os.makedirs(directory, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.fact-', suffix='.jsonl', dir=directory)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as handle:
            for record in data:
                handle.write(json.dumps(record, ensure_ascii=False) + '\n')
        os.replace(temporary, file_path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def records_by_id(data):
    result = {}
    for record in data:
        if not isinstance(record, dict):
            raise ValueError('Every FACT record must be a JSON object')
        identifier = record.get('id')
        if isinstance(identifier, bool) or not isinstance(identifier, (str, int)) or not str(identifier).strip():
            raise ValueError('Every FACT record must have a nonempty ID')
        identifier = str(identifier).strip()
        if identifier in result:
            raise ValueError(f'Duplicate FACT task ID: {identifier}')
        result[identifier] = dict(record, id=identifier)
    return result


def valid_deduplication_cache(current, cached):
    """A cached deduplication must retain source facts for every original URL."""
    if current.get('extraction_error') or cached.get('dedup_error'):
        return False
    try:
        raw = validate_citations([dict(item, ref_idx=item.get('ref_idx', 0))
                                  for item in current['citations']])
        groups = {}
        for citation in raw:
            groups.setdefault(citation['url'], set()).add(citation['fact'])
        deduped = cached['citations_deduped']
        if not isinstance(deduped, dict) or set(deduped) != set(groups):
            return False
        for url, value in deduped.items():
            if not isinstance(value, dict):
                return False
            facts = value.get('facts')
            if (not isinstance(facts, list) or not facts
                    or any(not isinstance(fact, str) or fact not in groups[url] for fact in facts)
                    or len(set(facts)) != len(facts)):
                return False
        return True
    except (KeyError, TypeError, ValueError, AttributeError):
        return False


def validate_citations(citations):
    if not isinstance(citations, list):
        raise ValueError('Citations must be a JSON list')
    result = []
    for citation in citations:
        if not isinstance(citation, dict):
            raise ValueError('Each citation must be an object')
        fact, url, index = citation.get('fact'), citation.get('url'), citation.get('ref_idx')
        if not isinstance(fact, str) or not fact.strip():
            raise ValueError('Citation fact must be nonempty text')
        if not isinstance(url, str) or urlparse(url).scheme not in ('http', 'https') or not urlparse(url).hostname:
            raise ValueError('Citation URL must be an HTTP(S) URL')
        if isinstance(index, str) and index.isdigit():
            index = int(index)
        if type(index) is not int or index < 0:
            raise ValueError('Citation ref_idx must be a nonnegative integer')
        result.append(dict(citation, fact=fact.strip(), url=url.strip(), ref_idx=index))
    return result


def normalize_validation(results, count, one_based=False):
    """Check completeness and return judgments in statement order."""
    if not isinstance(results, list) or len(results) != count:
        raise ValueError('Validation must contain one judgment per fact')
    normalized = []
    for item in results:
        if not isinstance(item, dict) or type(item.get('idx')) is not int:
            raise ValueError('Validation indices must be integers')
        index = item['idx'] - int(one_based)
        if not 0 <= index < count or item.get('result') not in ('supported', 'unsupported', 'unknown'):
            raise ValueError('Invalid validation index or result')
        normalized.append(dict(item, idx=index))
    if len({item['idx'] for item in normalized}) != count:
        raise ValueError('Validation indices must be unique and complete')
    return sorted(normalized, key=lambda item: item['idx'])


def validation_cache_matches(current, cached, judge_model=None):
    if judge_model is not None and cached.get('validation_model') != judge_model:
        return False
    if cached.get('validate_error') is not None:
        return False
    if any(current.get(key) != cached.get(key) for key in ('facts', 'url_content')):
        return False
    try:
        normalize_validation(cached.get('validate_res'), len(current['facts']))
        return True
    except (KeyError, TypeError, ValueError):
        return False

def safe_json_parse(response_text):
    """Reject incomplete JSON rather than silently dropping citations."""
    return json.loads(response_text)


logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logging.getLogger('httpx').setLevel(logging.WARNING)

class WebScrapingJinaTool:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get("JINA_API_KEY")
        if not self.api_key:
            raise ValueError("Jina API key not provided! Please set JINA_API_KEY environment variable.")

    def __call__(self, url: str) -> Dict[str, Any]:
        try:
            jina_url = f'https://r.jina.ai/{url}'
            headers = {
                "Accept": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "X-Timeout": "60000",
                "X-With-Generated-Alt": "true",
            }

            proxy_url = os.environ.get("HTTP_PROXY") or os.environ.get("HTTPS_PROXY")
            proxies = {"http": proxy_url, "https":proxy_url}

            response = requests.get(jina_url, headers=headers, proxies=proxies, timeout=30)

            if response.status_code != 200:
                raise Exception(f"Jina AI Reader Failed for {url}: {response.status_code}")

            response_dict = response.json()

            return {
                'url': response_dict['data']['url'],
                'title': response_dict['data']['title'],
                'description': response_dict['data']['description'],
                'content': response_dict['data']['content'],
                'publish_time': response_dict['data'].get('publishedTime', 'unknown')
            }

        except Exception as e:
            logger.error(str(e))
            return {
                'url': url,
                'content': '',
                'error': str(e)
            }

def get_redirected_url(target_url):
    """
    获取重定向链接：
    1. 先直连
    2. 切换代理重试
    """

    # 配置代理
    proxy_url = os.environ.get('HTTP_PROXY') or os.environ.get('HTTPS_PROXY')
    my_proxies = {"http": proxy_url, "https": proxy_url} if proxy_url else None

    # print(f"\n正在尝试直连: {target_url}")
    try:
        response = requests.get(
            target_url,
            impersonate="chrome110",
            allow_redirects=True,
            timeout=5
        )
        # print(">>> 直连成功！")
        return response.url

    except Exception as e:
        pass
        # print(f">>> 直连失败: {str(e)}")
        # print(">>> 准备切换代理重试...")

    if not my_proxies:
        return target_url
    try:
        response = requests.get(
            target_url,
            impersonate="chrome110",
            proxies=my_proxies,
            allow_redirects=True,
            timeout=20
        )
        # print(">>> 代理重试成功！")
        return response.url

    except Exception as e:
        # print(f">>> 连接失败: {e}")
        return target_url



def _local_scrape(url: str) -> Dict[str, Any]:
    """本地抓取降级：当 jina.ai 不可用时直接抓取网页"""
    try:
        proxy_url = os.environ.get("HTTP_PROXY") or os.environ.get("HTTPS_PROXY")
        proxies = {"http": proxy_url, "https": proxy_url} if proxy_url else None

        response = requests.get(
            url,
            impersonate="chrome110",
            proxies=proxies,
            timeout=30,
            allow_redirects=True
        )
        response.raise_for_status()

        content_type = response.headers.get('content-type', '').lower()
        if 'application/pdf' in content_type or response.content.startswith(b'%PDF-'):
            import pymupdf
            with pymupdf.open(stream=response.content, filetype='pdf') as document:
                text = '\n'.join(page.get_text() for page in document).strip()
            if not text:
                raise ValueError('PDF has no extractable text')
            return {'url': response.url, 'title': '', 'description': '',
                    'content': text, 'publish_time': 'unknown'}

        # 简单的内容提取
        text = response.text
        title = ""
        description = ""
        publish_time = 'unknown'

        # 尝试提取 title
        import re
        title_match = re.search(r'<title[^>]*>(.*?)</title>', text, re.IGNORECASE | re.DOTALL)
        if title_match:
            title = re.sub(r'<[^>]+>', '', title_match.group(1)).strip()

        # 尝试提取 meta description
        desc_match = re.search(r'<meta[^>]*name=["\']description["\'][^>]*content=["\']([^"\']+)', text, re.IGNORECASE)
        if desc_match:
            description = desc_match.group(1).strip()

        # Parse metadata attributes in either order, including entity escapes.
        from html import unescape
        for tag in re.findall(r'<(?:meta|time)\b[^>]*>', text, re.IGNORECASE):
            attrs = {key.lower(): unescape(value) for key, _, value in
                     re.findall(r'([\w:-]+)\s*=\s*([\"\'])(.*?)\2', tag, re.DOTALL)}
            name = (attrs.get('property') or attrs.get('name') or attrs.get('itemprop') or '').lower()
            if name == 'description':
                description = attrs.get('content', '')
            if name in ('article:published_time', 'datepublished', 'date', 'dc.date', 'pubdate'):
                publish_time = attrs.get('content', publish_time)
            elif tag.lower().startswith('<time') and attrs.get('datetime') and publish_time == 'unknown':
                publish_time = attrs['datetime']

        # 移除 script/style 标签
        text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.IGNORECASE | re.DOTALL)
        # 移除所有 HTML 标签
        text = re.sub(r'<[^>]+>', ' ', text)
        # 合并空白
        text = unescape(' '.join(text.split()))
        if not text:
            raise ValueError('Webpage has no extractable text')

        return {
            'url': response.url,
            'title': title,
            'description': description,
            'content': text,
            'publish_time': publish_time
        }
    except Exception as e:
        logger.error(f"Local scrape failed for {url}: {e}")
        return {
            'url': url,
            'content': '',
            'error': str(e)
        }


def scrape_url(url: str) -> Dict[str, Any]:
    # 首先访问URL获取可能重定向后的最终URL
    final_url = get_redirected_url(url)

    # Missing optional Jina credentials must still allow direct extraction.
    if not os.environ.get('JINA_API_KEY'):
        return _local_scrape(final_url)
    jina_tool = WebScrapingJinaTool()
    result = jina_tool(final_url)

    # 如果 jina 失败，降级到本地抓取
    if 'error' in result or not result.get('content'):
        logger.warning(f"Jina failed for {final_url}, falling back to local scrape")
        result = _local_scrape(final_url)

    return result

if __name__ == "__main__":
    url = "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQE3e0U0t3j-uixtVOK-kZwZAv5aL750Q4XaGOHBOA9n6qd_CKHYqmNhbPh0Wr0ArsIkySnCJE4yziYpbUgVZ2TBrhDzFkqBSSW38kSSWXkeK_VlW6kcRDAUZ1kXpBwZuIPcbZyMPuCyr8bw8kuY2h3RLkNaBS6XHQA7dOQUvYN6A9pZ5AExqosg1_GkZnf98PjOa3Tm57tlGpQx9w=="
    result = scrape_url(url)
    print(result)
