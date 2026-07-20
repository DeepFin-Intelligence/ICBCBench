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

    def __init__(self, api_key: Optional[str] = None):
        """
        初始化模型。
        :param api_key: API Key。如果为None，子类通常应尝试从环境变量加载。
        """
        self.api_key = api_key
        self.validate_env()

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