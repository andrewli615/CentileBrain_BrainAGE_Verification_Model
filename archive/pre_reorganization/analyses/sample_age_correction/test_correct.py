import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from correct import correct_frame, discover
from compare_controls import reference_frame


def sample(ages, values):
    return pd.DataFrame({"SITE": ["site"] * len(ages),
                         "SubjectID": [f"{i:03}" for i in range(len(ages))],
                         "chronological_age": ages, "Adjusted_BrainAGE": values})


class CorrectionTests(unittest.TestCase):
    def test_reference_schema_and_source_preservation(self):
        source = pd.DataFrame({"age": [10, 20, 30],
                               "predicted brain age": [12, 23, 34],
                               "predicted brain age_adjusted": [11, 22, 33]})
        result, summary = correct_frame(reference_frame(source), "5-40")
        pd.testing.assert_frame_equal(result[source.columns], source)
        np.testing.assert_allclose(result.Adjusted_BrainAGE, [1, 2, 3])
        np.testing.assert_allclose(result.BrainAGE_S, [2, 2, 2])
        self.assertEqual(result.SubjectID.tolist(), ["source_row_1", "source_row_2", "source_row_3"])
        self.assertTrue(result.SITE.eq("not_available").all())

    def test_known_slope_and_preserved_identity(self):
        source = sample([20, 30, 40], [4.2, 3, 1.8])
        result, summary = correct_frame(source, "5-40")
        pd.testing.assert_frame_equal(result[source.columns], source)
        np.testing.assert_allclose(result["age_centered"], [-10, 0, 10])
        np.testing.assert_allclose(result["residual_age_effect"], [1.2, 0, -1.2])
        np.testing.assert_allclose(result["BrainAGE_S"], [3, 3, 3])
        self.assertAlmostEqual(summary["Slope_Before"], -.12)
        self.assertAlmostEqual(summary["Mean_Change"], 0)
        self.assertAlmostEqual(summary["Slope_After"], 0)

    def test_nonzero_noise_mean_and_slope(self):
        ages = np.arange(10, 31)
        values = 5 + .3 * ages + np.sin(ages)
        result, summary = correct_frame(sample(ages, values), "5-40")
        self.assertAlmostEqual(result["BrainAGE_S"].mean(), values.mean())
        self.assertAlmostEqual(summary["Slope_After"], 0)

    def test_boundaries_and_exclusions(self):
        source = sample([4, 5, 20, 40, 41, 90, 91, np.nan, np.inf, 25, 26],
                        [1, 2, 3, 4, 5, 6, 7, 8, 9, np.inf, "bad"])
        young, summary = correct_frame(source, "5-40")
        self.assertEqual(summary["N_Eligible"], 3)
        self.assertEqual(young.index[young["eligible"]].tolist(), [1, 2, 3])
        self.assertTrue(young.loc[~young["eligible"], "BrainAGE_S"].isna().all())
        self.assertIn("missing_or_nonfinite_age", young.loc[7, "exclusion_reason"])
        self.assertIn("missing_or_nonfinite_brainage", young.loc[9, "exclusion_reason"])
        older, summary = correct_frame(source, "40-90")
        self.assertEqual(older.index[older["eligible"]].tolist(), [4, 5])
        self.assertEqual(summary["Status"], "skipped")

    def test_constant_age_and_small_samples(self):
        for source, reason in [(sample([20, 20, 20], [1, 2, 3]), "fewer_than_two_distinct_ages"),
                               (sample([20, 30], [1, 2]), "fewer_than_three_eligible"),
                               (sample([], []), "fewer_than_three_eligible")]:
            result, summary = correct_frame(source, "5-40")
            self.assertEqual(summary["Reason"], reason)
            self.assertTrue(result["BrainAGE_S"].isna().all())

    def test_missing_folders_and_latest_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertEqual(len(list(discover(root))), 28)
            self.assertTrue(all(row[-1] is None for row in discover(root)))
            folder = root / "self_black" / "Female_40-90"
            folder.mkdir(parents=True)
            old = folder / "old_Adjusted_BrainAGE_female.csv"
            new = folder / "new_Adjusted_BrainAGE_female.csv"
            old.touch()
            new.touch()
            import os
            os.utime(old, (1, 1))
            os.utime(new, (2, 2))
            chosen = [row[-1] for row in discover(root) if row[-1] is not None]
            self.assertEqual(chosen, [new])


if __name__ == "__main__":
    unittest.main()
