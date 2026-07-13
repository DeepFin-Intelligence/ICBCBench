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

load_dotenv()


class GeminiDeepResearch(DeepResearchModel):
    def __init__(self, api_key: str = None, agent_name: str = 'deep-research-pro-preview-12-2025'):
        self.agent_name = agent_name
        super().__init__(api_key)

    def validate_env(self):
        load_dotenv()
        if not self.api_key:
            self.api_key = os.getenv("GEMINI_API_KEY")
        self.client = genai.Client(api_key=self.api_key)

    def run_research(self, system_prompt, dr_task, multimodal_content=None):
        # Gemini interactions.create only accepts string input.
        # Extract text from multimodal content if provided.
        if isinstance(multimodal_content, list):
            text_parts = [block["text"] for block in multimodal_content if block.get("type") == "text"]
            image_count = sum(1 for block in multimodal_content if block.get("type") == "image_url")
            if image_count > 0:
                text_parts.append(f"[Note: {image_count} image(s) attached but not processable by Gemini Deep Research Agent]")
            query = f"{system_prompt}\n" + "\n".join(text_parts)
        else:
            query = f"{system_prompt}\n{dr_task}"

        interaction = self.client.interactions.create(
            input=query,
            agent=self.agent_name,
            background=True
        )

        while True:
            interaction = self.client.interactions.get(interaction.id)
            if interaction.status == "completed":
                result = interaction.outputs[-1].text
                return {"content": result}
            elif interaction.status == "failed":
                error_msg = f"Research failed: {interaction.error}"
                return {"content": error_msg}
            time.sleep(10)
