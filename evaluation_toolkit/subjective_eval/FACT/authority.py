from urllib.parse import urlparse
import pandas as pd
import requests
import json
import re
import os

class AuthorityDatabaseBuilder:
    def __init__(self):
        self.domain_map = {} # 格式: {'domain': weight}

    def add_gov_domains(self):
        """1. 获取美国官方 .gov 列表"""
        url = "https://raw.githubusercontent.com/cisagov/dotgov-data/main/current-full.csv"
        try:
            df = pd.read_csv(url)
            for domain in df['Domain Name'].dropna():
                self.domain_map[domain.lower()] = 1.0
            print(f"[Info] Added {len(df)} .gov domains.")
        except Exception as e:
            print(f"[Error] Failed to fetch .gov data: {e}")

    def add_university_domains(self):
        """2. 获取全球大学列表"""
        url = "https://raw.githubusercontent.com/Hipo/university-domains-list/master/world_universities_and_domains.json"
        try:
            resp = requests.get(url)
            data = resp.json()
            count = 0
            for entry in data:
                for domain in entry.get('domains', []):
                    self.domain_map[domain.lower()] = 1.0
                    count += 1
            print(f"[Info] Added {count} university domains.")
        except Exception as e:
            print(f"[Error] Failed to fetch university data: {e}")

    def add_manual_whitelist(self):
        """3. 添加手动维护的顶级白名单"""
        whitelist = {
            'nature.com': 1.0, 'science.org': 1.0, 'reuters.com': 1.0,
            'bloomberg.com': 1.0, 'apnews.com': 1.0, 'who.int': 1.0,
            'arxiv.org': 1.0, 'bbc.com': 0.9, 'nytimes.com': 0.9,
            'wsj.com': 0.9, 'economist.com': 0.9
        }
        self.domain_map.update(whitelist)

    def export_csv(self, filename="authority_domains.csv"):
        df = pd.DataFrame(list(self.domain_map.items()), columns=['domain', 'weight'])
        df.to_csv(filename, index=False)
        print(f"Database saved to {filename} with {len(df)} records.")

# # 使用
# builder = AuthorityDatabaseBuilder()
# builder.add_gov_domains()
# builder.add_university_domains()
# builder.add_manual_whitelist()
# builder.export_csv()


# 推荐：从外部加载配置文件，使得代码和配置解耦
def load_authority_config(config_path=None):
    if config_path is None:
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "authority_config.json")
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"[Warning] Failed to load {config_path}: {e}. Using fallback config.")
        return {
            "high_trust_suffixes": [".gov", ".edu", ".mil", ".int"],
            "high_trust_domains": ["bloomberg.com", "reuters.com", "wsj.com"],
            "medium_high_trust_domains": ["cnbc.com", "economist.com"],
            "medium_trust_domains": ["wikipedia.org", "zhihu.com"],
            "default_weight": 0.5
        }


AUTHORITY_CONFIG = load_authority_config()


def get_authority_weight(url):
    """
    解析 URL 并返回特定领域的权威度权重。
    """
    COMPLEX_SUFFIXES = {'com', 'org', 'net', 'gov', 'edu', 'co', 'ac', 'mil'}

    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()

        # 1. 清洗域名 (去除 www. 或其他常见二级前缀以便于主域名匹配)
        # 考虑到金融领域可能存在诸如 finance.yahoo.com，需要提取主域名进行匹配
        parts = domain.split('.')
        # 场景1：带有双层后缀的域名（如 pbc.gov.cn, finance.sina.com.cn, a.b.hkex.com.hk）
        # 至少需要3段，且倒数第二段属于双层后缀列表
        if len(parts) >= 3 and parts[-2] in COMPLEX_SUFFIXES:
            root_domain = f"{parts[-3]}.{parts[-2]}.{parts[-1]}"

        # 场景2：普通的单层后缀域名（如 szse.cn, bloomberg.com, finance.yahoo.com）
        # 至少需要2段，且倒数第二段不是特殊的双层后缀
        elif len(parts) >= 2 and parts[-2] not in COMPLEX_SUFFIXES:
            root_domain = f"{parts[-2]}.{parts[-1]}"

        # 场景3：极端兜底（如 localhost，或者输入的就是 com.cn 本身）
        else:
            root_domain = domain[4:] if domain.startswith("www.") else domain

        domain_to_check = root_domain
        # print(f"[Info] Checking authority weight for {domain_to_check}...")

        # 2. 判定 Tier 1 (后缀匹配 - 如 sec.gov, pbc.gov.cn)
        # 注意: AUTHORITY_CONFIG["high_trust_suffixes"] 需要转换为 tuple 才能传入 endswith
        if domain.endswith(tuple(AUTHORITY_CONFIG.get("high_trust_suffixes", []))):
            return 1.0

        # 3. 判定 Tier 1 (顶级权威域名精确匹配)
        if domain_to_check in AUTHORITY_CONFIG.get("high_trust_domains", []):
            return 1.0

        # 4. 判定 Tier 2 (高质量/机构研究)
        if domain_to_check in AUTHORITY_CONFIG.get("medium_high_trust_domains", []):
            return 0.85  # 或 0.9

        # 5. 判定 Tier 3 (大众媒体/UGC/公关稿)
        if domain_to_check in AUTHORITY_CONFIG.get("medium_trust_domains", []):
            return 0.7  # 原来是 0.8，针对严谨的金融研究予以适当降权

        # 6. 默认 Tier 4
        return AUTHORITY_CONFIG.get("default_weight", 0.5)

    except Exception:
        # 如果 URL 解析失败，按最低权重处理
        return 0.5

if __name__ == "__main__":

    test_url = [
        "https://www.sse.com.cn/?PC=PC",

    ]

    for url in test_url:
        weight = get_authority_weight(url)
        print(f"[Info] {url} 的权威度权重为 {weight:.2f}")