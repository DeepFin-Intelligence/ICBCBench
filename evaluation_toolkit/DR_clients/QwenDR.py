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
import dashscope
from dotenv import load_dotenv
from evaluation_toolkit.model_clients import *

from evaluation_toolkit.DR_clients.DeepResearch import DeepResearchModel
from evaluation_toolkit.DR_clients.citations import json_value, render_numbered


def _convert_to_qwen_multimodal(content):
    """Use the dedicated Deep Research API's text-only message contract."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise ValueError("Qwen Deep Research input must be text")

    result = []
    for block in content:
        if block.get("type") == "text":
            result.append(block["text"])
        elif block.get("type") == "image_url":
            raise ValueError("This Qwen Deep Research client supports text only; image support is inconsistent across official API documents")
        else:
            raise ValueError(f"Unsupported Qwen input block: {block.get('type')}")
    return "\n".join(result)


class TongyiDeepResearch(DeepResearchModel):
    def __init__(self, api_key: str = None, model_name: str = "alibaba/tongyi-deepresearch-30b-a3b", **kwargs):
        self.client = OPENROUTER
        self.model_name = model_name
        super().__init__(api_key, **kwargs)

    def validate_env(self):
        load_project_env()
        pass

    def run_research(self, system_prompt: str, dr_task: str, multimodal_content=None):
        deadline = self.deadline()
        user_content = multimodal_content if multimodal_content is not None else dr_task

        first_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]

        # First API call with reasoning
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=first_messages,
            extra_body={"reasoning": {"enabled": True}},
            timeout=min(60, max(.1, deadline - time.monotonic()))
        )
        require_complete_chat(response)
        self.check_deadline(deadline)
        first_usage = response.usage

        # Extract the assistant message with reasoning_details
        response = response.choices[0].message

        # Preserve the assistant message with reasoning_details
        assistant_message = {"role": "assistant", "content": response.content}
        reasoning_details = getattr(response, "reasoning_details", None)
        if reasoning_details is not None:
            assistant_message["reasoning_details"] = reasoning_details
        second_messages = [
            assistant_message,
            {
                "role": "user",
                "content": user_content
            }
        ]

        messages = first_messages + second_messages
        self.check_deadline(deadline)

        # Second API call - model continues reasoning from where it left off
        response2 = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            extra_body={"reasoning": {"enabled": True}},
            timeout=min(60, max(.1, deadline - time.monotonic()))
        )

        require_complete_chat(response2)
        self.check_deadline(deadline)
        content = self.require_content(response2.choices[0].message.content)
        usage_records = [u for u in (first_usage, response2.usage) if u is not None]
        usage = None if not usage_records else {
            "input_tokens": sum(u.prompt_tokens for u in usage_records),
            "output_tokens": sum(u.completion_tokens for u in usage_records),
            "total_tokens": sum(u.total_tokens for u in usage_records)
        }

        return {"content": content, "usage": usage}


class QwenDeepResearch(DeepResearchModel):
    def __init__(self, api_key: str = None, model_name: str = "qwen-deep-research", **kwargs):
        self.model_name = model_name
        super().__init__(api_key, **kwargs)

    def validate_env(self):
        load_project_env()
        if not (self.api_key or os.getenv("DASHSCOPE_API_KEY")):
            raise ValueError("Missing DASHSCOPE_API_KEY.")

    def run_research(self, system_prompt: str, dr_task: str, multimodal_content=None):
        user_content = multimodal_content if multimodal_content is not None else dr_task
        user_content = _convert_to_qwen_multimodal(user_content)
        self.require_content(user_content)

        first_messages = [{"role": "user", "content": f"{system_prompt}\n{user_content}"}]

        deadline = self.deadline()
        responses = dashscope.Generation.call(
            api_key=self.api_key or os.getenv('DASHSCOPE_API_KEY'),
            model=self.model_name,
            messages=first_messages,
            stream=True,
            incremental_output=True,
            request_timeout=min(getattr(self, "timeout_seconds", 3600), 60)
        )

        step1_content, first_usage = self._collect_answer(responses, deadline)

        # 第二步：深入研究
        second_messages = [
            {'role': 'assistant', 'content': step1_content},
            {'role': 'user', 'content': user_content}
        ]

        messages = first_messages + second_messages

        # Second API call - model continues reasoning from where it left off
        responses2 = dashscope.Generation.call(
            api_key=self.api_key or os.getenv('DASHSCOPE_API_KEY'),
            model=self.model_name,
            messages=messages,
            stream=True,
            incremental_output=True,
            request_timeout=min(60, max(0.1, deadline - time.monotonic()))
        )

        final_content, second_usage = self._collect_answer(responses2, deadline)
        usages = [usage for usage in (first_usage, second_usage) if usage]
        usage = None if not usages else {
            "input_tokens": sum(item.get("input_tokens", 0) for item in usages),
            "output_tokens": sum(item.get("output_tokens", 0) for item in usages),
        }
        if usage:
            usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
        return {"content": final_content, "usage": usage,
                "citations": self._last_citations,
                "raw_citations": self._last_raw_citations}

    def _collect_answer(self, responses, deadline):
        answer = []
        finished = False
        usage = None
        raw_citations = []
        self._last_citations = []
        self._last_raw_citations = []
        try:
            for response in responses:
                self.check_deadline(deadline)
                if getattr(response, "status_code", 200) != 200:
                    raise RuntimeError(f"Qwen research failed: {getattr(response, 'code', '')} {getattr(response, 'message', '')}")
                output = getattr(response, "output", None) or {}
                message = output.get("message", {})
                finish_reason = (output.get("finish_reason") or message.get("finish_reason")
                                 or message.get("finished_reason") or message.get("fininshed_reason"))
                if finish_reason and finish_reason not in ("null", "stop"):
                    raise RuntimeError(f"Qwen research ended with {finish_reason}")
                if message.get("phase") != "answer":
                    continue
                references = (message.get("extra") or {}).get("deep_research", {}).get("references", [])
                raw_citations.extend(json_value(references))
                content = message.get("content", "")
                if content:
                    if not isinstance(content, str):
                        raise RuntimeError("Qwen answer content must be text")
                    answer.append(content)
                if message.get("status") == "finished":
                    finished = True
                    raw_usage = getattr(response, "usage", None)
                    if raw_usage:
                        usage = dict(raw_usage)
            if not finished:
                raise RuntimeError("Qwen answer stream ended before the answer finished")
            content, citations = render_numbered(self.require_content("".join(answer)), raw_citations)
            self._last_citations, self._last_raw_citations = citations, raw_citations
            return content, usage
        finally:
            close = getattr(responses, "close", None)
            if close:
                close()
