from __future__ import annotations
import os
from openai import OpenAI
import threading
import time
from .base_client import BaseLLMClient
from ..config import BuilderConfig

class GroqClient(BaseLLMClient):
    # Class-level rate limiting to share across all instances/workers
    _lock = threading.Lock()
    _last_request_time = 0.0
    _min_interval = 2.0  # 30 RPM -> 2 seconds per request

    def __init__(self, cfg: BuilderConfig):
        super().__init__(cfg)
        
        # 1. Use a low-tier model if default or unspecified
        if self.model in ("gpt-5-nano", "none", "", "gpt-4o"):
            self.model = "llama-3.1-8b-instant"
            
        # Groq is OpenAI-compatible
        raw_key = os.environ.get("GROQ_API_KEY")
        self.client = OpenAI(
            api_key=raw_key.strip() if raw_key else None,
            base_url="https://api.groq.com/openai/v1"
        )

    def _generate_inner(self, prompt: str, system_prompt: str | None = None, **kwargs) -> str:
        # 2. Rate limiting: ensure at least _min_interval between requests
        with self._lock:
            now = time.time()
            elapsed = now - GroqClient._last_request_time
            if elapsed < self._min_interval:
                wait_time = self._min_interval - elapsed
                time.sleep(wait_time)
            GroqClient._last_request_time = time.time()

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
