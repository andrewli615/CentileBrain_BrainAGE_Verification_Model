import unittest

import numpy as np
import pandas as pd

from correct_chinese import assign_buckets, fit_training, apply_training


class BucketCorrectionTests(unittest.TestCase):
    def test_boundaries(self):
        ages = pd.Series([4, 5, 14, 15, 25, 35, 40, 41, 50, 90, 91, np.nan, np.inf])
        young = assign_buckets(ages, "5-40")
        old = assign_buckets(ages, "40-90")
        self.assertEqual(young.dropna().tolist(), [0, 0, 1, 2, 3, 3])
        self.assertEqual(old.dropna().tolist(), [0, 1, 4])
        self.assertTrue(pd.isna(old.iloc[6]))

    def test_subtracts_training_slope_and_intercept_without_refitting_chinese(self):
        training = pd.DataFrame({"chronological_age": [5., 8., 12.], "Raw_BrainAGE": [-1., -.4, .4]})
        fits = fit_training(training, "5-40")
        chinese = pd.DataFrame({"chronological_age": [6., 10., 14.], "Raw_BrainAGE": [4., 4., 4.]})
        adjusted = apply_training(chinese, fits, "5-40")
        np.testing.assert_allclose(adjusted.Training_Slope_a, .2)
        np.testing.assert_allclose(adjusted.Training_Intercept_b, -2)
        np.testing.assert_allclose(adjusted.Estimated_Training_Bias, [-.8, 0, .8], atol=1e-12)
        np.testing.assert_allclose(adjusted.BrainAGE_TrainingBucketAdjusted, [4.8, 4., 3.2])
        self.assertTrue(adjusted.Outside_Training_Age_Support.iloc[-1])
        corrected_training = apply_training(training, fits, "5-40")
        np.testing.assert_allclose(corrected_training.BrainAGE_TrainingBucketAdjusted, 0, atol=1e-12)

    def test_missing_and_unfit_training(self):
        training = pd.DataFrame({"chronological_age": [5., 6.], "Raw_BrainAGE": [1., 2.]})
        chinese = pd.DataFrame({"chronological_age": [5., 40., 41., np.nan, 6.], "Raw_BrainAGE": [1., 1., 1., 1., np.inf]})
        result = apply_training(chinese, fit_training(training, "5-40"), "5-40")
        self.assertEqual(result.Correction_Status.tolist(), ["skipped", "skipped", "excluded", "excluded", "excluded"])
        self.assertTrue(result.BrainAGE_TrainingBucketAdjusted.isna().all())


if __name__ == "__main__":
    unittest.main()
