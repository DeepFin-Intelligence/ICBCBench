import json
import os
from typing import Optional, Dict, Any
import requests
import logging
from dotenv import load_dotenv
from curl_cffi import requests

load_dotenv()

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            data.append(json.loads(line.strip()))
    return data

def safe_json_parse(response_text):
    """安全解析可能截断的JSON响应"""
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        # 尝试修复截断的JSON
        # 查找最后一个完整的JSON对象结束位置
        for i in range(len(response_text), 0, -1):
            try:
                partial_json = response_text[:i].rstrip(', \n\r')
                if partial_json.endswith('}'):
                    return json.loads(partial_json + ']')
            except:
                continue
        raise ValueError("无法解析JSON响应")


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
    my_proxies = {
        "http": "http://127.0.0.1:7890",
        "https": "http://127.0.0.1:7890",
    }

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

        # 简单的内容提取
        text = response.text
        title = ""
        description = ""

        # 尝试提取 title
        import re
        title_match = re.search(r'<title[^>]*>(.*?)</title>', text, re.IGNORECASE | re.DOTALL)
        if title_match:
            title = re.sub(r'<[^>]+>', '', title_match.group(1)).strip()

        # 尝试提取 meta description
        desc_match = re.search(r'<meta[^>]*name=["\']description["\'][^>]*content=["\']([^"\']+)', text, re.IGNORECASE)
        if desc_match:
            description = desc_match.group(1).strip()

        # 移除 script/style 标签
        text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.IGNORECASE | re.DOTALL)
        # 移除所有 HTML 标签
        text = re.sub(r'<[^>]+>', ' ', text)
        # 合并空白
        text = ' '.join(text.split())

        return {
            'url': response.url,
            'title': title,
            'description': description,
            'content': text[:8000],  # 限制长度
            'publish_time': 'unknown'
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

    jina_tool = WebScrapingJinaTool()

    # 优先使用 jina_tool
    result = jina_tool(final_url)

    # 如果 jina 失败，降级到本地抓取
    if 'error' in result:
        logger.warning(f"Jina failed for {final_url}, falling back to local scrape")
        result = _local_scrape(final_url)

    return result

if __name__ == "__main__":
    url = "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQE3e0U0t3j-uixtVOK-kZwZAv5aL750Q4XaGOHBOA9n6qd_CKHYqmNhbPh0Wr0ArsIkySnCJE4yziYpbUgVZ2TBrhDzFkqBSSW38kSSWXkeK_VlW6kcRDAUZ1kXpBwZuIPcbZyMPuCyr8bw8kuY2h3RLkNaBS6XHQA7dOQUvYN6A9pZ5AExqosg1_GkZnf98PjOa3Tm57tlGpQx9w=="
    result = scrape_url(url)
    print(result)