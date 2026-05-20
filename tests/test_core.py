import unittest

import numpy as np

from can_anomaly.features import entropy
from can_anomaly.config import TEST_RAW_FILES, TRAIN_RAW_FILES, Paths
from can_anomaly.labels import fallback_label_for_file, majority_label
from can_anomaly.preprocessing import parse_numeric_token


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
        self.assertEqual(fallback_label_for_file("normal_train.csv"), 1)
        self.assertEqual(fallback_label_for_file("dos_test.csv"), 0)
        self.assertEqual(majority_label([1, 1, 2, 3]), 1)

    def test_entropy(self):
        self.assertAlmostEqual(entropy(np.array([1, 1, 1, 1])), 0.0, places=6)
        self.assertAlmostEqual(entropy(np.array([1, 2, 1, 2])), 1.0, places=6)

    def test_train_and_test_paths_are_separate(self):
        paths = Paths()
        self.assertEqual(paths.train_raw_dir.name, "train")
        self.assertEqual(paths.test_raw_dir.name, "test")
        self.assertNotEqual(paths.train_features_path, paths.test_features_path)
        self.assertIn("normal_train.csv", TRAIN_RAW_FILES)
        self.assertIn("normal_test.csv", TEST_RAW_FILES)


if __name__ == "__main__":
    unittest.main()

