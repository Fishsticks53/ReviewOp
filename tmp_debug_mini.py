import sys
import os
from pathlib import Path

# Add dataset_builder/code to path
sys.path.append(str(Path("dataset_builder/code").resolve()))

from implicit_features import _extract_phrases_fallback, _extract_candidate_phrases, _is_valid_aspect

seed_vocab = set()
text = "The menu was great."
raw_phrases = _extract_phrases_fallback(text, seed_vocab=seed_vocab)
print(f"Raw phrases: {raw_phrases}")

cand_phrases = _extract_candidate_phrases(text, seed_vocab=seed_vocab)
print(f"Candidate phrases: {cand_phrases}")

for p in raw_phrases:
    print(f"Testing '{p}': valid? {_is_valid_aspect(p, seed_vocab=seed_vocab)}")
