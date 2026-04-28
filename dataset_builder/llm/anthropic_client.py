from __future__ import annotations
import os
import threading
import time
from anthropic import Anthropic
from .base_client import BaseLLMClient
from ..config import BuilderConfig

class AnthropicClient(BaseLLMClient):
    _lock = threading.Lock()
    _last_call = 0.0
    _min_interval = 1.2  # 50 RPM safety limit

    def __init__(self, cfg: BuilderConfig):
        super().__init__(cfg)
        raw_key = os.environ.get("CLAUDE_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
        api_key = raw_key.strip() if raw_key else None
        self.client = Anthropic(api_key=api_key)

    def _generate_inner(self, prompt: str, system_prompt: str | None = None, **kwargs) -> str:
        with self._lock:
            now = time.time()
            elapsed = now - self._last_call
            if elapsed < self._min_interval:
                time.sleep(self._min_interval - elapsed)
            self.__class__._last_call = time.time()

        create_kwargs = {
            "model": self.model,
            "max_tokens": kwargs.get("max_tokens", 1024),
            "messages": [{"role": "user", "content": prompt}],
        }

        if system_prompt:
            create_kwargs["system"] = [
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"}
                }
            ]

        response = self.client.messages.create(**create_kwargs)
        # Claude 3 returns a list of content blocks
        return response.content[0].text if response.content else ""
