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
from evaluation_toolkit.DR_clients.DeepResearch import DeepResearchModel


class TavilyDeepResearch(DeepResearchModel):
    """Tavily的深度研究模型"""

    def validate_env(self):
        """
        验证并初始化环境配置，确保API密钥可用并创建Tavily客户端实例。

        Raises:
            ValueError: 当无法获取 `TAVILY_API_KEY` 环境变量时抛出。
        """
        load_dotenv()
        if not self.api_key:
            self.api_key = os.getenv("TAVILY_API_KEY")
        if not self.api_key:
            raise ValueError("Missing TAVILY_API_KEY")
        self.client = TavilyClient(api_key=self.api_key)

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
            text_parts = [block["text"] for block in multimodal_content if block.get("type") == "text"]
            query = f"{system_prompt}\n" + "\n".join(text_parts)
        else:
            query = f"{system_prompt}\n{dr_task}"

        response = self.client.research(query)

        # Get the request ID
        request_id = response.get("request_id")

        # Poll for results
        while True:
            time.sleep(10)
            result = self.client.get_research(request_id)
            if result.get("status") == "completed":
                break
            elif result.get("status") == "failed":
                print("Research failed")
                break

        return {"content": result.get("content")}
