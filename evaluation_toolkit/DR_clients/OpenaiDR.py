import sys
from pathlib import Path
_p = Path(__file__).resolve()
while _p.name != "evaluation_toolkit" and _p.parent != _p:
    _p = _p.parent
if _p.name == "evaluation_toolkit":
    _root = _p.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

import os
import time
from dotenv import load_dotenv
from evaluation_toolkit.DR_clients.DeepResearch import DeepResearchModel
from typing import Optional
from openai import OpenAI


class OpenaiDeepResearch(DeepResearchModel):
    def __init__(self,
                 model_choose: str = ""):
        self.model_name = f"{model_choose}"

        super().__init__()

    def validate_env(self):
        """加载环境并初始化 OpenAI 客户端"""
        load_dotenv()
        if not self.api_key:
            self.api_key = os.getenv("OPENAI_API_KEY")

        if not self.api_key:
            raise ValueError("Missing API Key. Please set OPENROUTER_API_KEY.")

        self.client = OpenAI(
            api_key=self.api_key
        )

    def run_research(self, system_prompt: str, dr_task: str, multimodal_content=None):
        """
        执行研究任务。

        Args:
            system_prompt: 系统提示词
            dr_task: 用户任务/问题
            multimodal_content: OpenAI-compatible multimodal content list or str
        """
        if isinstance(multimodal_content, list):
            # Build multimodal input for Responses API
            content = []
            for block in multimodal_content:
                if block.get("type") == "text":
                    content.append({"type": "input_text", "text": block["text"]})
                elif block.get("type") == "image_url":
                    url = block.get("image_url", {}).get("url", "")
                    content.append({"type": "input_image", "image_url": url})
            input_data = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content}
            ]
        else:
            query = f"{system_prompt}\n{dr_task}"
            input_data = query

        try:
            # 3. 发起请求
            response = self.client.responses.create(
                model=self.model_name,
                input=input_data,
                tools=[
                    {"type": "web_search_preview"},
                    {
                        "type": "code_interpreter",
                        "container": {"type": "auto"}
                    },
                ],
            )

            # 4. 解析结果
            result_content = response.output_text

            return {
                "content": result_content,
                "usage": None
            }

        except Exception as e:
            return {
                "content": f"Request failed: {str(e)}"
            }
