import unittest

import numpy as np
import pandas as pd

from plot_residuals import draw_corrected_panel, plt, style


class PlottedRegressionTests(unittest.TestCase):
    def test_training_line_uses_actual_regression_not_a_forced_zero_slope(self):
        training = pd.DataFrame({"age": [10., 20., 30.], "residual": [1., 3., 5.]})
        observed = pd.DataFrame({"age": [10., 20., 30.], "residual": [2., 4., 6.]})
        panel = style.PlotData("synthetic", "Synthetic", "Female", "5-40", training, observed)
        fig, ax = plt.subplots()
        try:
            rows = draw_corrected_panel(ax, panel, 10, True)
            line = next(line for line in ax.lines if line.get_label().startswith("Training global fit"))
            np.testing.assert_allclose(line.get_ydata(), .2 * line.get_xdata() - 1)
            slope = next(row['Slope'] for row in rows if row['Data'] == 'Training' and row['Fit_Mode'] == 'linear')
            self.assertAlmostEqual(slope, .2)
            self.assertIn("+0.2000", line.get_label())
        finally:
            plt.close(fig)


if __name__ == "__main__":
    unittest.main()
