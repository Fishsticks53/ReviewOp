import os
import requests
import time
import threading
from .base_client import BaseLLMClient
from ..config import BuilderConfig

class HuggingFaceClient(BaseLLMClient):
    _lock = threading.Lock()
    _last_call = 0.0
    _min_interval = 2.0  # ~30 RPM for free tier safety

    def __init__(self, cfg: BuilderConfig):
        super().__init__(cfg)
        raw_key = os.environ.get("HUGGINGFACE_API_KEY")
        self.api_key = raw_key.strip() if raw_key else None
        self.base_url = f"https://api-inference.huggingface.co/models/{self.model}"

    def _generate_inner(self, prompt: str, system_prompt: str | None = None, **kwargs) -> str:
        with self._lock:
            now = time.time()
            elapsed = now - self._last_call
            if elapsed < self._min_interval:
                time.sleep(self._min_interval - elapsed)
            self.__class__._last_call = time.time()

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        full_input = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt

        # HuggingFace Inference API expects specific payload format
        payload = {
            "inputs": full_input,
            "parameters": {
                "max_new_tokens": kwargs.get("max_tokens", 1024),
                "return_full_text": False,
            }
        }
        
        # Simple retry logic for 503 (Model loading) which is common in free tier
        max_retries = 3
        for attempt in range(max_retries):
            response = requests.post(self.base_url, headers=headers, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                # Serverless API usually returns a list
                if isinstance(result, list) and len(result) > 0:
                    return result[0].get("generated_text", "")
                return str(result)
            
            elif response.status_code == 503:
                # Model is loading, wait and retry
                wait_time = response.json().get("estimated_time", 20)
                if attempt < max_retries - 1:
                    time.sleep(min(wait_time, 10)) # Don't wait too long in a single block
                    continue
            
            response.raise_for_status()
            
        return ""
