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
from evaluation_toolkit.model_clients import load_project_env
from typing import Optional
from openai import OpenAI
from evaluation_toolkit.DR_clients.citations import json_value, render_annotations


class OpenaiDeepResearch(DeepResearchModel):
    def __init__(self,
                 model_choose: str = "", **kwargs):
        self.model_name = f"{model_choose}"

        super().__init__(**kwargs)

    def validate_env(self):
        """加载环境并初始化 OpenAI 客户端"""
        load_project_env()
        if not self.api_key:
            self.api_key = os.getenv("OPENAI_API_KEY")

        if not self.api_key:
            raise ValueError("Missing OPENAI_API_KEY.")

        self.client = OpenAI(
            api_key=self.api_key, timeout=60, max_retries=1
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
            task = multimodal_content if multimodal_content is not None else dr_task
            query = f"{system_prompt}\n{task}"
            input_data = query

        response = None
        task_id = None
        try:
            deadline = self.deadline()
            self.check_deadline(deadline)
            # 3. 发起请求
            response = self.client.responses.create(
                model=self.model_name,
                input=input_data,
                background=True,
                tools=[
                    {"type": "web_search_preview"},
                    {
                        "type": "code_interpreter",
                        "container": {"type": "auto"}
                    },
                ],
                timeout=min(getattr(self, "timeout_seconds", 3600), 60),
            )
            task_id = getattr(response, "id", None)
            self.check_deadline(deadline)
            while getattr(response, "status", "completed") in ("queued", "in_progress"):
                self.poll_wait(deadline)
                response = self.client.responses.retrieve(response.id, timeout=min(60, max(0.1, deadline - time.monotonic())))
                self.check_deadline(deadline)
            if getattr(response, "status", "completed") != "completed":
                raise RuntimeError(f"Research ended with {response.status}: {getattr(response, 'error', None)}")
            parts, raw_citations, citations = [], [], []
            for item in (getattr(response, "output", None) or []):
                if getattr(item, "type", None) != "message":
                    continue
                for part in (getattr(item, "content", None) or []):
                    if getattr(part, "type", None) != "output_text":
                        continue
                    raw = json_value(getattr(part, "annotations", None) or [])
                    rendered, normalized = render_annotations(part.text, raw)
                    parts.append(rendered)
                    raw_citations.extend(raw)
                    citations.extend(normalized)
            result_content = self.require_content("".join(parts) if parts else response.output_text)
            usage = getattr(response, "usage", None)

            return {
                "content": result_content,
                "usage": usage.model_dump() if usage is not None else None,
                "citations": citations,
                "raw_citations": raw_citations,
            }

        except BaseException as e:
            if task_id and getattr(response, "status", None) not in ("completed", "failed", "cancelled", "incomplete"):
                self.cancel_background(self.client.responses, task_id, e)
            if isinstance(e, Exception):
                task_detail = f" (research_task_id={task_id})" if task_id else ""
                raise RuntimeError(f"OpenAI research request failed{task_detail}: {e}") from e
            raise
