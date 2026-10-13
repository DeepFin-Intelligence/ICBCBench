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
from google import genai
from evaluation_toolkit.DR_clients.DeepResearch import DeepResearchModel
from evaluation_toolkit.model_clients import load_project_env
from evaluation_toolkit.DR_clients.citations import json_value, render_annotations

class GeminiDeepResearch(DeepResearchModel):
    def __init__(self, api_key: str = None, agent_name: str = 'deep-research-preview-04-2026', **kwargs):
        self.agent_name = agent_name
        super().__init__(api_key, **kwargs)

    def validate_env(self):
        load_project_env()
        if not self.api_key:
            self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Missing GEMINI_API_KEY")
        self.client = genai.Client(api_key=self.api_key)

    def run_research(self, system_prompt, dr_task, multimodal_content=None):
        if isinstance(multimodal_content, list):
            query = [{"type": "text", "text": system_prompt}]
            for block in multimodal_content:
                if block.get("type") == "text":
                    query.append({"type": "text", "text": block["text"]})
                elif block.get("type") == "image_url":
                    url = block.get("image_url", {}).get("url", "")
                    if url.startswith("data:"):
                        header, data = url.split(",", 1)
                        query.append({"type": "image", "mime_type": header[5:].split(";")[0], "data": data})
                    else:
                        query.append({"type": "image", "uri": url})
        else:
            task = multimodal_content if multimodal_content is not None else dr_task
            query = f"{system_prompt}\n{task}"

        deadline = self.deadline()
        interaction = None
        task_id = None
        try:
            self.check_deadline(deadline)
            interaction = self.client.interactions.create(
                input=query,
                agent=self.agent_name,
                background=True,
                timeout=min(getattr(self, "timeout_seconds", 3600), 60)
            )
            task_id = getattr(interaction, "id", None)
            while True:
                self.check_deadline(deadline)
                interaction = self.client.interactions.get(interaction.id, timeout=min(60, max(0.1, deadline - time.monotonic())))
                self.check_deadline(deadline)
                if interaction.status == "completed":
                    parts, citations, raw_citations = [], [], []
                    for step in (getattr(interaction, "steps", None) or []):
                        if getattr(step, "type", "") != "model_output":
                            continue
                        for part in (getattr(step, "content", None) or []):
                            text = getattr(part, "text", None)
                            if not isinstance(text, str):
                                continue
                            raw = json_value(getattr(part, "annotations", None) or [])
                            rendered, normalized = render_annotations(text, raw, byte_offsets=True)
                            parts.append(rendered)
                            citations.extend(normalized)
                            raw_citations.extend(raw)
                    result = "".join(parts) or getattr(interaction, "output_text", None)
                    if not result:
                        outputs = getattr(interaction, "outputs", None) or []
                        result = "".join(getattr(output, "text", "") or "" for output in outputs)
                    return {"content": self.require_content(result), "citations": citations, "raw_citations": raw_citations}
                elif interaction.status in ("failed", "cancelled", "requires_action"):
                    detail = getattr(interaction, "errors", None) or getattr(interaction, "error", None)
                    raise RuntimeError(f"Research ended with {interaction.status}: {detail}")
                self.poll_wait(deadline)
        except BaseException as exc:
            if task_id and getattr(interaction, "status", None) not in ("completed", "failed", "cancelled"):
                self.cancel_background(self.client.interactions, task_id, exc)
            if task_id and hasattr(exc, "add_note"):
                exc.add_note(f"research_task_id={task_id}")
            raise
