import os
import time
from dotenv import load_dotenv
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any

class DeepResearchModel(ABC):
    """
    深度研究模型的抽象基类。
    所有具体的模型实现（Tavily, Gemini, Qwen等）都必须继承此类。
    """

    def __init__(self, api_key: Optional[str] = None, timeout_seconds: float = 3600, poll_interval: float = 10):
        """
        初始化模型。
        :param api_key: API Key。如果为None，子类通常应尝试从环境变量加载。
        """
        self.api_key = api_key
        if timeout_seconds <= 0 or poll_interval < 0:
            raise ValueError("Research timeout must be positive and poll interval nonnegative")
        self.timeout_seconds = timeout_seconds
        self.poll_interval = poll_interval
        self.validate_env()

    def deadline(self):
        return time.monotonic() + getattr(self, "timeout_seconds", 3600)

    def check_deadline(self, deadline):
        if getattr(self, "cancel_event", None) is not None and self.cancel_event.is_set():
            raise RuntimeError("Research interrupted by the calling pipeline")
        if time.monotonic() >= deadline:
            raise TimeoutError("Research exceeded the configured timeout")

    def poll_wait(self, deadline):
        self.check_deadline(deadline)
        delay = min(getattr(self, "poll_interval", 10), max(0, deadline - time.monotonic()))
        event = getattr(self, "cancel_event", None)
        if event is None:
            time.sleep(delay)
        else:
            event.wait(delay)
        self.check_deadline(deadline)

    @staticmethod
    def cancel_background(api, task_id, failure):
        """Cancel this invocation's running task without masking its original failure."""
        failure.research_task_id = task_id
        try:
            api.cancel(task_id, timeout=10)
        except Exception as exc:
            failure.research_cancel_error = str(exc)
            print(f"Research task {task_id} cancellation failed: {exc}")

    @staticmethod
    def require_content(content):
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("Research completed without a nonempty text report")
        return content

    @abstractmethod
    def run_research(self, system_prompt: str, dr_task: str, multimodal_content: Optional[Any] = None):
        """
        执行深度研究任务。

        Args:
            system_prompt (str): 系统提示词/角色定义
            dr_task (str): 具体的研究任务或问题（当 multimodal_content 为 None 时使用）
            multimodal_content (str | list | None):
                - None 或 str: 纯文本内容（回退到 dr_task）
                - list: OpenAI-compatible 多模态 content 列表，例如
                  [{"type": "text", "text": "..."}, {"type": "image_url", "image_url": {"url": "data:..."}}]

        Returns:
            dict: {"content": str, "usage": dict | None}
        """
        pass

    def validate_env(self):
        """可选：用于验证必要的环境变量或依赖是否存在"""
        pass
