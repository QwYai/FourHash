import unittest

import numpy as np

from src.rz_merge import compile_reference, merge_two_lists, radius_population, radius_table


class RZMergeTests(unittest.TestCase):
    def test_compiled_moments_match_direct_scan(self):
        rng = np.random.default_rng(7)
        bank = rng.choice((-1, 1), size=(257, 64)).astype(np.int8)
        query = rng.choice((-1, 1), size=64).astype(np.int8)
        state = compile_reference(bank)
        mean, variance = radius_population(query, state)
        radii = np.count_nonzero(bank != query, axis=1)
        self.assertAlmostEqual(mean, float(radii.mean()), places=12)
        self.assertAlmostEqual(variance, float(radii.var()), places=12)

    def test_radius_table_preserves_hamming_order_and_ties(self):
        bank = np.asarray([[1, 1, 1, 1], [1, -1, 1, -1], [-1, -1, -1, -1]])
        table = radius_table(np.asarray([1, 1, 1, 1]), compile_reference(bank))
        self.assertTrue(np.all(np.diff(table) < 0))
        self.assertEqual(table[2], table[2])

    def test_two_list_merge(self):
        image = np.asarray([3.0, 2.0, 1.0])
        text = np.asarray([2.5, 2.25, 0.0])
        merged = merge_two_lists([10, 11], [0, 1], [20, 21], [0, 1], image, text, 3)
        self.assertEqual([(row[0], row[1]) for row in merged], [("image", 10), ("text", 20), ("text", 21)])


if __name__ == "__main__":
    unittest.main()

