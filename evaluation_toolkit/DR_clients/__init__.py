import sys
from pathlib import Path
_p = Path(__file__).resolve()
while _p.name != "evaluation_toolkit" and _p.parent != _p:
    _p = _p.parent
if _p.name == "evaluation_toolkit":
    _root = _p.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from evaluation_toolkit.DR_clients.QwenDR import TongyiDeepResearch, QwenDeepResearch
from evaluation_toolkit.DR_clients.GeminiDR import GeminiDeepResearch
from evaluation_toolkit.DR_clients.TavilyResearch import TavilyDeepResearch
from evaluation_toolkit.DR_clients.OpenaiDR import OpenaiDeepResearch

__all__ = [
    "TongyiDeepResearch",
    "GeminiDeepResearch",
    "TavilyDeepResearch",
    "OpenaiDeepResearch",
    "QwenDeepResearch", "create_dr_client"
]


def create_dr_client(model):
    """Instantiate only the provider selected by the caller."""
    factories = {
        "gemini-deep-research": GeminiDeepResearch,
        "tongyi-deepresearch-30b-a3b": TongyiDeepResearch,
        "tongyi-deep-research": TongyiDeepResearch,
        "Qwen-deep-research": QwenDeepResearch,
        "qwen-deep-research": QwenDeepResearch,
        "tavily-research": TavilyDeepResearch,
        "o4-mini-deep-research": lambda: OpenaiDeepResearch(model_choose=model),
        "o3-deep-research": lambda: OpenaiDeepResearch(model_choose=model),
    }
    factory = factories.get(model)
    return factory() if factory else None
