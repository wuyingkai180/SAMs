from __future__ import annotations

import unittest
from collections import Counter
from pathlib import Path

from analysis.new_solvent_analysis.manifest import build_manifest


class ManifestTests(unittest.TestCase):
    def test_real_manifest_has_27_valid_systems_and_no_trash(self):
        records = build_manifest(Path("data"))
        self.assertEqual(len(records), 27)
        self.assertEqual(
            Counter(record.group for record in records),
            {"group1": 7, "group2": 13, "group3": 7},
        )
        self.assertFalse(any("trash" in str(record.csv_path) for record in records))

    def test_acetone_five_percent_resolves_nonmatching_file_prefix(self):
        record = next(
            record
            for record in build_manifest(Path("data"))
            if record.group == "group1"
            and record.system == "acetone_5_n-heptane_95"
        )
        self.assertEqual(record.file_prefix, "acetone_n-heptane")
        self.assertEqual(record.cosolvent_fraction, 0.05)

    def test_group3_roles_are_stratified(self):
        roles = {
            record.system: record.role
            for record in build_manifest(Path("data"))
            if record.group == "group3"
        }
        self.assertEqual(
            roles["thf_n-heptane"], "n-heptane-base-composition"
        )
        self.assertEqual(
            roles["thf_toluene"], "alternate-base-composition"
        )
        self.assertEqual(roles["ethyl_acetate"], "pure-reference")


if __name__ == "__main__":
    unittest.main()
