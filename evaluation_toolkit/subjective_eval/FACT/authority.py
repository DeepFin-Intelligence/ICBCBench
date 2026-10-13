from urllib.parse import urlparse
import pandas as pd
import requests
import json
import re
import os

class AuthorityDatabaseBuilder:
    def __init__(self):
        self.domain_map = {}  # Format: {'domain': weight}

    def add_gov_domains(self):
        """1. Fetch official US .gov domain list"""
        url = "https://raw.githubusercontent.com/cisagov/dotgov-data/main/current-full.csv"
        try:
            df = pd.read_csv(url)
            for domain in df['Domain Name'].dropna():
                self.domain_map[domain.lower()] = 1.0
            print(f"[Info] Added {len(df)} .gov domains.")
        except Exception as e:
            print(f"[Error] Failed to fetch .gov data: {e}")

    def add_university_domains(self):
        """2. Fetch global university domain list"""
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
        """3. Add manually maintained top-tier whitelist"""
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

# # Usage
# builder = AuthorityDatabaseBuilder()
# builder.add_gov_domains()
# builder.add_university_domains()
# builder.add_manual_whitelist()
# builder.export_csv()


# Recommended: Load config from external file to decouple code and configuration
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
    Parse URL and return authority weight for the specific domain.
    """
    try:
        parsed = urlparse(url)
        domain = (parsed.hostname or '').lower().rstrip('.')
        if not domain:
            return AUTHORITY_CONFIG.get('default_weight', 0.5)

        # 2. Determine Tier 1 (suffix match - e.g., sec.gov, pbc.gov.cn)
        # Note: AUTHORITY_CONFIG["high_trust_suffixes"] must be converted to tuple for endswith
        if domain.endswith(tuple(AUTHORITY_CONFIG.get("high_trust_suffixes", []))):
            return 1.0

        # 3. Determine Tier 1 (exact match with top authority domains)
        if any(domain == candidate or domain.endswith('.' + candidate)
               for candidate in AUTHORITY_CONFIG.get("high_trust_domains", [])):
            return 1.0

        # 4. Determine Tier 2 (high-quality/institutional research)
        if any(domain == candidate or domain.endswith('.' + candidate)
               for candidate in AUTHORITY_CONFIG.get("medium_high_trust_domains", [])):
            return 0.85  # or 0.9

        # 5. Determine Tier 3 (mass media/UGC/PR articles)
        if any(domain == candidate or domain.endswith('.' + candidate)
               for candidate in AUTHORITY_CONFIG.get("medium_trust_domains", [])):
            return 0.7  # Previously 0.8, reduced for rigorous financial research

        # 6. Default Tier 4
        return AUTHORITY_CONFIG.get("default_weight", 0.5)

    except Exception:
        # If URL parsing fails, use the minimum weight
        return 0.5

if __name__ == "__main__":

    test_url = [
        "https://www.sse.com.cn/?PC=PC",

    ]

    for url in test_url:
        weight = get_authority_weight(url)
        print(f"[Info] Authority weight for {url} is {weight:.2f}")
