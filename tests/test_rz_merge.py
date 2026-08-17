import unittest

import numpy as np

from src.rz_merge import (
    _strict_float32_table,
    compile_reference,
    csls_integer_key,
    dual_radius_tables,
    merge_two_lists,
    radius_population,
    radius_table,
    shell_composite_key,
)


class RZMergeTests(unittest.TestCase):
    def test_compiled_moments_match_direct_scan(self):
        rng = np.random.default_rng(7)
        for bits in (16, 32, 64):
            bank = rng.choice((-1, 1), size=(257, bits)).astype(np.int8)
            query = rng.choice((-1, 1), size=bits).astype(np.int8)
            state = compile_reference(bank)
            mean, variance = radius_population(query, state)
            radii = np.count_nonzero(bank != query, axis=1)
            self.assertAlmostEqual(mean, float(radii.mean()), places=12)
            self.assertAlmostEqual(variance, float(radii.var()), places=12)

    def test_radius_table_is_strict_float32(self):
        bank = np.asarray([[1, 1, 1, 1], [1, -1, 1, -1], [-1, -1, -1, -1]])
        table = radius_table(np.asarray([1, 1, 1, 1]), compile_reference(bank))
        self.assertEqual(table.dtype, np.float32)
        self.assertTrue(np.all(np.diff(table) < 0))

    def test_float32_projection_separates_rounded_coalescence(self):
        raw = np.asarray([1.0 + 1e-8, 1.0, 0.5], dtype=np.float64)
        self.assertEqual(raw[:2].astype(np.float32)[0], raw[:2].astype(np.float32)[1])
        table = _strict_float32_table(raw)
        self.assertTrue(np.all(table[:-1] > table[1:]))
        self.assertEqual(table[1], np.nextafter(np.float32(1.0), np.float32(-np.inf)))

    def test_dual_tables_share_zero_variance_fallback(self):
        query = np.asarray([1, 1, 1, 1])
        degenerate = compile_reference(np.tile(query, (5, 1)))
        regular = compile_reference(
            np.asarray([[1, 1, 1, 1], [1, -1, 1, -1], [-1, -1, -1, -1]])
        )
        image, text = dual_radius_tables(query, degenerate, regular)
        expected = (-np.arange(5, dtype=np.float64) / 4.0).astype(np.float32)
        np.testing.assert_array_equal(image, expected)
        np.testing.assert_array_equal(text, expected)

    def test_csls_integer_key(self):
        key = csls_integer_key([3, 3, 5], [20, 26, 40], k=10)
        np.testing.assert_array_equal(key, [-40, -34, -60])

    def test_shell_key_refines_one_shell_and_preserves_shell_order(self):
        primary = np.asarray([3.0, 2.0, 1.0], dtype=np.float32)
        shell = np.asarray([0, 0, 1, 1, 2])
        secondary = np.asarray([2, 9, 100, 3, 500])
        key = shell_composite_key(primary, shell, secondary)
        self.assertGreater(key[1], key[0])
        self.assertTrue(np.all(key[shell == 0] > key[shell == 1].max()))
        self.assertTrue(np.all(key[shell == 1] > key[shell == 2].max()))

    def test_coalesced_distinct_shells_remain_tied(self):
        primary = np.asarray([2.0, 1.0, 1.0], dtype=np.float32)
        shell = np.asarray([1, 1, 2, 2])
        secondary = np.asarray([1, 8, 3, 20])
        key = shell_composite_key(primary, shell, secondary)
        self.assertEqual(len(np.unique(key)), 1)

    def test_two_list_merge(self):
        image = np.asarray([3.0, 2.0, 1.0])
        text = np.asarray([2.5, 2.25, 0.0])
        merged = merge_two_lists([10, 11], [0, 1], [20, 21], [0, 1], image, text, 3)
        self.assertEqual([(row[0], row[1]) for row in merged], [("image", 10), ("text", 20), ("text", 21)])


if __name__ == "__main__":
    unittest.main()
