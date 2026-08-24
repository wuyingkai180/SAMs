from __future__ import annotations

import unittest
from collections import Counter
from pathlib import Path

from analysis.new_solvent_analysis.manifest import build_manifest


class ManifestTests(unittest.TestCase):
    def test_real_manifest_has_26_valid_systems_and_no_trash(self):
        records = build_manifest(Path("data"))
        self.assertEqual(len(records), 26)
        self.assertEqual(
            Counter(record.group for record in records),
            {"group1": 7, "group2": 14, "group3": 5},
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
        self.assertEqual(
            set(roles),
            {
                "acetone_5_n-heptane_95",
                "isopropanol_5_n-heptane_95",
                "thf_n-heptane",
                "thf_toluene",
                "toluene_n-heptane",
            },
        )

    def test_moved_pure_references_are_group2_pure_solvents(self):
        records = {
            (record.group, record.system): record
            for record in build_manifest(Path("data"))
        }
        for system in ("ethyl_acetate", "propylene_carbonate"):
            self.assertEqual(records[("group2", system)].role, "pure-solvent")
        self.assertNotIn(("group2", "prol"), records)


if __name__ == "__main__":
    unittest.main()
