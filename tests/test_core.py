import unittest

import numpy as np

from can_ai.features import entropy
from can_ai.labels import fallback_label_for_file, majority_label
from can_ai.preprocessing import parse_numeric_token


class CoreBehaviorTests(unittest.TestCase):
    def test_hex_parser_handles_can_id_formats(self):
        self.assertEqual(parse_numeric_token("0x164", base="hex"), 356)
        self.assertEqual(parse_numeric_token("0164", base="hex"), 356)
        self.assertEqual(parse_numeric_token("0080", base="hex"), 128)
        self.assertEqual(parse_numeric_token("dc", base="hex"), 220)

    def test_auto_parser_keeps_plain_decimal_when_unambiguous(self):
        self.assertEqual(parse_numeric_token("215", base="auto"), 215)
        self.assertEqual(parse_numeric_token("0164", base="auto"), 356)

    def test_label_helpers(self):
        self.assertEqual(fallback_label_for_file("dataset1.csv"), 1)
        self.assertEqual(fallback_label_for_file("dataset2.csv"), 0)
        self.assertEqual(majority_label([1, 1, 2, 3]), 1)

    def test_entropy(self):
        self.assertAlmostEqual(entropy(np.array([1, 1, 1, 1])), 0.0, places=6)
        self.assertAlmostEqual(entropy(np.array([1, 2, 1, 2])), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
