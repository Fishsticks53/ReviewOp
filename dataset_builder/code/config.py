from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent


@dataclass
class LLMSettings:
    enabled: bool = False
    provider: str = "groq"
    groq_api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    groq_model: str = field(default_factory=lambda: os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"))
    groq_base_url: str = field(default_factory=lambda: os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1"))
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    openai_model: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-5.4-mini"))
    openai_base_url: str = field(default_factory=lambda: os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    timeout_seconds: int = 45


@dataclass
class BuilderConfig:
    input_dir: Path = ROOT / "input"
    output_dir: Path = ROOT / "output"
    random_seed: int = 42
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    text_column_override: str | None = None
    aspect_vocab_size: int = 30
    confidence_threshold: float = 0.65
    imbalance_correction: str = "oversample"
    enable_llm_fallback: bool = True
    llm_sample_size: int = 20  # Only use LLM on the first N rows per split to save cost
    drop_near_duplicates: bool = True
    near_duplicate_threshold: float = 0.95
    min_text_tokens: int = 5
    train_only_aspect_sample_size: int = 10_000
    max_one_hot_cardinality: int = 50
    reports_subdir: str = "reports"
    priors_path: Path = (ROOT / "priors" / "generic_reviews.json").resolve()
    llm: LLMSettings = field(default_factory=LLMSettings)

    @property
    def explicit_dir(self) -> Path:
        return self.output_dir / "explicit"

    @property
    def implicit_dir(self) -> Path:
        return self.output_dir / "implicit"

    @property
    def reports_dir(self) -> Path:
        return self.output_dir / self.reports_subdir

    @property
    def protonet_export_dir(self) -> Path:
        return self.output_dir / "protonet_implicit"

    @property
    def backend_export_dir(self) -> Path:
        return self.output_dir / "backend_implicit"

    @property
    def protonet_input_dir(self) -> Path:
        return REPO_ROOT / "protonet" / "input" / "episodic"

    @property
    def backend_implicit_dir(self) -> Path:
        return REPO_ROOT / "backend" / "data" / "implicit"

    def ensure_dirs(self) -> None:
        for path in [
            self.output_dir,
            self.explicit_dir,
            self.implicit_dir,
            self.reports_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)
