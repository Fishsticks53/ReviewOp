from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List

import requests

from config import LLMSettings


@dataclass
class LLMFallbackResult:
    aspect: str
    sentiment: str
    confidence: float
    is_novel_aspect: bool


class BaseLLMClient:
    def infer(self, *, sentence: str, candidate_aspects: List[str]) -> LLMFallbackResult | None:
        raise NotImplementedError


class DisabledLLMClient(BaseLLMClient):
    def infer(self, *, sentence: str, candidate_aspects: List[str]) -> LLMFallbackResult | None:
        return None


def _parse_json_result(text: str) -> LLMFallbackResult | None:
    if not text:
        return None
    try:
        data: Dict[str, Any] = json.loads(text)
    except json.JSONDecodeError:
        return None
    return LLMFallbackResult(
        aspect=str(data.get("aspect", "general")).strip() or "general",
        sentiment=str(data.get("sentiment", "neutral")).strip().lower() or "neutral",
        confidence=float(data.get("confidence", 0.5)),
        is_novel_aspect=bool(data.get("is_novel_aspect", False)),
    )


class GroqLLMClient(BaseLLMClient):
    def __init__(self, settings: LLMSettings) -> None:
        self.settings = settings

    def infer(self, *, sentence: str, candidate_aspects: List[str]) -> LLMFallbackResult | None:
        if not self.settings.groq_api_key:
            return None

        prompt = (
            "Task: Extract the IMPLICIT aspect and sentiment from the review sentence. "
            "An implicit aspect is one where the aspect word is NOT mentioned, but a symptom is present. "
            "IMPORTANT: If the aspect word is literally in the text, it is EXPLICIT—do not return it. "
            "Focus on symptoms: 'waited' -> service, 'lag' -> performance, 'blurred' -> camera. "
            "Return valid JSON only with keys: aspect, sentiment, confidence, is_novel_aspect. "
            f"Candidate aspects: {candidate_aspects}. Sentence: {sentence}"
        )
        try:
            response = requests.post(
                f"{self.settings.groq_base_url.rstrip('/')}/chat/completions",
                headers={
                    "authorization": f"Bearer {self.settings.groq_api_key}",
                    "content-type": "application/json",
                },
                json={
                    "model": self.settings.groq_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0,
                    "max_completion_tokens": 128,
                },
                timeout=self.settings.timeout_seconds,
            )
            response.raise_for_status()
        except Exception as e:
            print(f"LLM Error (Groq): {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response: {e.response.text}")
            return None

        payload = response.json()
        text = (
            payload.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        return _parse_json_result(text)


class OpenAILLMClient(BaseLLMClient):
    def __init__(self, settings: LLMSettings) -> None:
        self.settings = settings

    def infer(self, *, sentence: str, candidate_aspects: List[str]) -> LLMFallbackResult | None:
        if not self.settings.openai_api_key:
            return None

        prompt = (
            "Task: Extract the IMPLICIT aspect and sentiment from the review sentence. "
            "An implicit aspect is one where the aspect word is NOT mentioned, but a symptom is present. "
            "IMPORTANT: If the aspect word is literally in the text, it is EXPLICIT—do not return it. "
            "Focus on symptoms: 'waited' -> service, 'lag' -> performance, 'blurred' -> camera. "
            "Return valid JSON only with keys: aspect, sentiment, confidence, is_novel_aspect. "
            f"Candidate aspects: {candidate_aspects}. Sentence: {sentence}"
        )
        try:
            response = requests.post(
                f"{self.settings.openai_base_url.rstrip('/')}/chat/completions",
                headers={
                    "authorization": f"Bearer {self.settings.openai_api_key}",
                    "content-type": "application/json",
                },
                json={
                    "model": self.settings.openai_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0,
                    "max_completion_tokens": 128,
                },
                timeout=self.settings.timeout_seconds,
            )
            response.raise_for_status()
        except Exception as e:
            print(f"LLM Error (OpenAI): {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response: {e.response.text}")
            return None

        payload = response.json()
        text = (
            payload.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        return _parse_json_result(text)


def build_llm_client(settings: LLMSettings, *, enabled: bool) -> BaseLLMClient:
    if not enabled:
        return DisabledLLMClient()
    provider = settings.provider.lower()
    if provider == "groq":
        return GroqLLMClient(settings)
    if provider == "openai":
        return OpenAILLMClient(settings)
    if settings.groq_api_key:
        return GroqLLMClient(settings)
    if settings.openai_api_key:
        return OpenAILLMClient(settings)
    return DisabledLLMClient()

