import sys
import os
from pathlib import Path

# Add dataset_builder/code to path
sys.path.append(str(Path("dataset_builder/code").resolve()))

from implicit_features import _aspect_rejection_reason

seed_vocab = set()
reason = _aspect_rejection_reason("menu", seed_vocab=seed_vocab)
print(f"Reason for 'menu': {reason}")
