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
    "QwenDeepResearch"
]