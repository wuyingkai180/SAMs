from __future__ import annotations

import unittest

import numpy as np

from analysis.new_solvent_analysis.solvation import (
    assign_species_blocks,
    enrichment_factor,
    minimum_image,
)


class SolvationTests(unittest.TestCase):
    def test_enrichment_is_one_when_local_matches_bulk(self):
        self.assertAlmostEqual(
            enrichment_factor(local_count=5, local_total=20, bulk_fraction=0.25),
            1.0,
        )

    def test_species_assignment_consumes_every_atom_once(self):
        templates = {
            "acetone": ("C", "C", "C", "O"),
            "n-heptane": ("C", "C", "H", "H"),
        }
        symbols = templates["acetone"] * 2 + templates["n-heptane"] * 3
        blocks = assign_species_blocks(
            symbols, templates, {"acetone": 2, "n-heptane": 3}
        )
        self.assertEqual(sum(len(values) for values in blocks.values()), len(symbols))
        self.assertEqual(set(blocks), {"acetone", "n-heptane"})

    def test_species_assignment_rejects_wrong_order(self):
        templates = {"a": ("C", "O"), "b": ("N", "H")}
        with self.assertRaises(ValueError):
            assign_species_blocks(("N", "H", "C", "O"), templates, {"a": 1, "b": 1})

    def test_minimum_image_distance(self):
        displacement = minimum_image(
            np.array([9.0, 0.0, 0.0]), np.diag([10.0, 10.0, 10.0])
        )
        np.testing.assert_allclose(displacement, [-1.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
