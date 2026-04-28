from __future__ import annotations
import os
import threading
import time
from openai import OpenAI
from .base_client import BaseLLMClient
from ..config import BuilderConfig

class OpenAIClient(BaseLLMClient):
    _lock = threading.Lock()
    _last_call = 0.0
    _min_interval = 1.0  # 60 RPM safety limit

    def __init__(self, cfg: BuilderConfig):
        super().__init__(cfg)
        raw_key = os.environ.get("OPENAI_API_KEY")
        self.client = OpenAI(api_key=raw_key.strip() if raw_key else None)

    def _generate_inner(self, prompt: str, system_prompt: str | None = None, **kwargs) -> str:
        with self._lock:
            now = time.time()
            elapsed = now - self._last_call
            if elapsed < self._min_interval:
                time.sleep(self._min_interval - elapsed)
            self.__class__._last_call = time.time()

        # Handle parameter renaming for newer models (o1, etc.)
        if "max_tokens" in kwargs:
            # Models starting with 'o1' require 'max_completion_tokens'
            if self.model.startswith("o1-") or "gpt-5" in self.model:
                kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **kwargs
        )
        return response.choices[0].message.content or ""
