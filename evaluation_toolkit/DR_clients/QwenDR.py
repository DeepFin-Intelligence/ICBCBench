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
import dashscope
from dotenv import load_dotenv
from evaluation_toolkit.model_clients import *

from evaluation_toolkit.DR_clients.DeepResearch import DeepResearchModel


def _convert_to_qwen_multimodal(content):
    """Convert OpenAI-compatible multimodal content to Qwen/DashScope format."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)

    result = []
    for block in content:
        if block.get("type") == "text":
            result.append({"type": "text", "text": block["text"]})
        elif block.get("type") == "image_url":
            url = block.get("image_url", {}).get("url", "")
            # DashScope qwen-vl format uses "image" key
            result.append({"image": url})
    return result


class TongyiDeepResearch(DeepResearchModel):
    def __init__(self, api_key: str = None, model_name: str = "tongyi-deepresearch-30b-a3b-ssvip"):
        self.client = DMX
        self.model_name = model_name
        super().__init__(api_key)

    def validate_env(self):
        load_dotenv()
        pass

    def run_research(self, system_prompt: str, dr_task: str, multimodal_content=None):
        user_content = multimodal_content if multimodal_content is not None else dr_task

        first_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]

        # First API call with reasoning
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=first_messages,
            extra_body={"reasoning": {"enabled": True}}
        )

        # Extract the assistant message with reasoning_details
        response = response.choices[0].message

        # Preserve the assistant message with reasoning_details
        second_messages = [
            {
                "role": "assistant",
                "content": response.content,
                "reasoning_details": response.reasoning_details  # Pass back unmodified
            },
            {
                "role": "user",
                "content": user_content
            }
        ]

        messages = first_messages + second_messages

        # Second API call - model continues reasoning from where it left off
        response2 = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            extra_body={"reasoning": {"enabled": True}}
        )

        content = response2.choices[0].message.content
        usage = {
            "input_tokens": response2.usage.prompt_tokens,
            "output_tokens": response2.usage.completion_tokens,
            "total_tokens": response2.usage.total_tokens
        }

        return {"content": content, "usage": usage}


class QwenDeepResearch(DeepResearchModel):
    def __init__(self, api_key: str = None, model_name: str = "qwen-deep-research"):
        super().__init__(api_key)

    def validate_env(self):
        load_dotenv()
        pass

    def run_research(self, system_prompt: str, dr_task: str, multimodal_content=None):
        user_content = multimodal_content if multimodal_content is not None else dr_task
        # Convert to Qwen-compatible format if it's a list
        if isinstance(user_content, list):
            user_content = _convert_to_qwen_multimodal(user_content)

        first_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]

        # First API call with reasoning
        responses = dashscope.Generation.call(
            api_key=os.getenv('DASHSCOPE_API_KEY'),
            model="qwen-deep-research",
            messages=first_messages,
            stream=True
        )

        # 获取模型反问内容
        step1_content = ""
        for response in responses:
            if hasattr(response, 'output') and response.output:
                message = response.output.get('message', {})
                content = message.get('content', '')
                if content:
                    step1_content += content

        # 第二步：深入研究
        second_messages = [
            {'role': 'assistant', 'content': step1_content},
            {'role': 'user', 'content': user_content}
        ]

        messages = first_messages + second_messages

        # Second API call - model continues reasoning from where it left off
        responses2 = dashscope.Generation.call(
            api_key=os.getenv('DASHSCOPE_API_KEY'),
            model="qwen-deep-research",
            messages=messages,
            stream=True
        )

        final_content = ""
        for response in responses2:
            if hasattr(response, 'output') and response.output:
                message = response.output.get('message', {})
                content = message.get('content', '')
                final_content += content

        return {"content": final_content, "usage": None}
