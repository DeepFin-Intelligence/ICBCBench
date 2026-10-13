import sys
from pathlib import Path
_p = Path(__file__).resolve()
while _p.name != "evaluation_toolkit" and _p.parent != _p:
    _p = _p.parent
if _p.name == "evaluation_toolkit":
    _root = _p.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from tavily import TavilyClient
import os
from dotenv import load_dotenv
import time
import requests
import re
from evaluation_toolkit.DR_clients.DeepResearch import DeepResearchModel
from evaluation_toolkit.model_clients import load_project_env
from evaluation_toolkit.DR_clients.citations import json_value, render_numbered


class _TimeoutSession(requests.Session):
    """The SDK's get_research has no timeout argument; bound its HTTP request."""

    def request(self, method, url, **kwargs):
        remaining = getattr(self, "deadline", float("inf")) - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Tavily research exceeded the configured timeout")
        kwargs["timeout"] = min(kwargs.get("timeout") or 60, remaining)
        return super().request(method, url, **kwargs)


class TavilyDeepResearch(DeepResearchModel):
    """Tavily的深度研究模型"""

    def validate_env(self):
        """
        验证并初始化环境配置，确保API密钥可用并创建Tavily客户端实例。

        Raises:
            ValueError: 当无法获取 `TAVILY_API_KEY` 环境变量时抛出。
        """
        load_project_env()
        if not self.api_key:
            self.api_key = os.getenv("TAVILY_API_KEY")
        if not self.api_key:
            raise ValueError("Missing TAVILY_API_KEY")
        self.client = TavilyClient(api_key=self.api_key, session=_TimeoutSession())

    def run_research(self, system_prompt: str, dr_task: str, multimodal_content=None):
        """
        执行研究任务并返回结果。

        参数:
            system_prompt (str): 系统提示信息，用于指导研究任务的执行。
            dr_task (str): 具体的研究任务描述。
            multimodal_content: OpenAI-compatible multimodal content list or str

        返回:
            Dict[str, Any]: 包含研究结果的字典，通常包括研究内容和相关来源。
        """
        if isinstance(multimodal_content, list):
            if any(block.get("type") == "image_url" for block in multimodal_content):
                raise ValueError("Tavily Research does not accept image inputs; use a model that supports multimodal research")
            text_parts = [block["text"] for block in multimodal_content if block.get("type") == "text"]
            query = f"{system_prompt}\n" + "\n".join(text_parts)
        else:
            task = multimodal_content if multimodal_content is not None else dr_task
            query = f"{system_prompt}\n{task}"

        deadline = self.deadline()
        if isinstance(getattr(self.client, "session", None), _TimeoutSession):
            self.client.session.deadline = deadline
        response = self.client.research(query, timeout=min(getattr(self, "timeout_seconds", 3600), 60))

        # Get the request ID
        request_id = response.get("request_id")
        if not request_id:
            raise RuntimeError("Tavily research response has no request_id")

        # Poll for results
        try:
            while True:
                self.poll_wait(deadline)
                result = self.client.get_research(request_id)
                self.check_deadline(deadline)
                if result.get("status") == "completed":
                    break
                elif result.get("status") == "failed":
                    raise RuntimeError(f"Tavily research failed: {result.get('error', request_id)}")
        except BaseException as exc:
            exc.research_task_id = request_id
            print(f"Tavily research polling stopped (research_task_id={request_id}): {exc}")
            raise

        raw_citations = json_value(result.get("sources") or [])
        content, citations = render_numbered(self.require_content(result.get("content")), raw_citations)
        result = {"content": content, "citations": citations, "raw_citations": raw_citations,
                  "usage": result.get("usage")}
        if (citations and re.search(r"\[\d+\](?!\()", content)
                and not any(citation["url"] in content for citation in citations)):
            result["citation_mapping_error"] = "Numeric citations have no explicit source-to-number mapping; FACT evaluation cannot reconstruct the URLs safely"
        return result
