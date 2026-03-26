import sys
import os
from pathlib import Path

# Add dataset_builder/code to path
sys.path.append(str(Path("dataset_builder/code").resolve()))

import pandas as pd
from config import BuilderConfig
from io_utils import load_inputs
from implicit_features import discover_aspects, _extract_candidate_phrases

cfg = BuilderConfig()
frame = load_inputs(cfg.input_dir)
rows = frame.to_dict(orient="records")

print(f"Total rows: {len(rows)}")

aspects = discover_aspects(rows[:2000], text_column="review", vocab_size=30, seed_vocab=set())
print("Discovered aspects (no seeds):")
print(aspects)

# Check why 'menu' might be missing
text_with_menu = "The menu was great."
phrases = _extract_candidate_phrases(text_with_menu, seed_vocab=set())
print(f"Phrases for '{text_with_menu}': {phrases}")
