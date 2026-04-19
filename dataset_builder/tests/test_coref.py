from __future__ import annotations

import sys
import unittest
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1] / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))


class CorefTests(unittest.TestCase):
    def test_heuristic_coref_uses_previous_sentence_antecedent_for_initial_pronoun(self) -> None:
        from coref import heuristic_coref

        result = heuristic_coref("The battery lasts all day. It charges fast.")

        self.assertEqual(result.text, "The battery lasts all day. battery charges fast.")
        self.assertEqual(result.chains, [{"pronoun": "It", "antecedent": "battery"}])


if __name__ == "__main__":
    unittest.main()
