"""
Unit and integration tests for src/evaluation.py (Member 3).
Tests all metrics, edge cases (zero denominators, NaN), and diagnostic plotting.
"""

import unittest
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless testing
import matplotlib.pyplot as plt

from src.evaluation import (
    calculate_residuals,
    calculate_mae,
    calculate_rmse,
    calculate_r2,
    calculate_mape,
    calculate_durbin_watson,
    calculate_all_metrics,
    compare_models,
    plot_actual_vs_predicted,
    plot_residuals_diagnostic,
)


class TestModelEvaluation(unittest.TestCase):
    """Test suite for evaluation helper and diagnostic functions."""

    def setUp(self):
        # Known synthetic ground truth and predictions
        self.y_true = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
        self.y_perfect = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
        self.y_mean = np.full_like(self.y_true, np.mean(self.y_true))
        self.y_pred = np.array([110.0, 190.0, 310.0, 390.0, 510.0])

    def test_r2_score(self):
        # Perfect predictions must equal exactly 1.0
        self.assertAlmostEqual(calculate_r2(self.y_true, self.y_perfect), 1.0)
        # Mean prediction must equal 0.0
        self.assertAlmostEqual(calculate_r2(self.y_true, self.y_mean), 0.0)
        # Small perturbation should give high R2
        r2_val = calculate_r2(self.y_true, self.y_pred)
        self.assertGreater(r2_val, 0.95)
        self.assertLess(r2_val, 1.0)

    def test_rmse_and_mae(self):
        # Residuals are all ±10
        mae = calculate_mae(self.y_true, self.y_pred)
        rmse = calculate_rmse(self.y_true, self.y_pred)

        self.assertAlmostEqual(mae, 10.0)
        self.assertAlmostEqual(rmse, 10.0)

    def test_safe_mape_standard(self):
        # Errors: |(100-110)/100|=10%, |(200-190)/200|=5%, |(300-310)/300|=3.333%...
        y_t = np.array([100.0, 200.0])
        y_p = np.array([110.0, 190.0])
        expected_mape = np.mean([10.0 / 100.0, 10.0 / 200.0]) * 100.0  # (0.1 + 0.05)/2 * 100 = 7.5%
        calculated_mape = calculate_mape(y_t, y_p)
        self.assertAlmostEqual(calculated_mape, expected_mape)

    def test_safe_mape_with_zero_denominator(self):
        # Guard test: actual value contains zero. Must NOT throw ZeroDivisionError or return inf/nan
        y_zero = np.array([0.0, 100.0, 200.0])
        y_hat = np.array([5.0, 105.0, 195.0])

        mape = calculate_mape(y_zero, y_hat, epsilon=1e-6)
        self.assertTrue(np.isfinite(mape))
        self.assertGreater(mape, 0.0)

    def test_residuals_calculation(self):
        res = calculate_residuals(self.y_true, self.y_pred)
        expected = self.y_true - self.y_pred
        np.testing.assert_allclose(res, expected)

    def test_durbin_watson_range(self):
        # Test uncorrelated residuals around 2.0
        np.random.seed(42)
        random_res = np.random.normal(0, 1, 100)
        dw = calculate_durbin_watson(random_res)
        self.assertGreater(dw, 1.5)
        self.assertLess(dw, 2.5)

        # Zero residuals
        zero_res = np.zeros(10)
        self.assertEqual(calculate_durbin_watson(zero_res), 2.0)

    def test_input_validation(self):
        # Shape mismatch
        with self.assertRaises(ValueError):
            calculate_r2([1, 2, 3], [1, 2])

        # Empty inputs
        with self.assertRaises(ValueError):
            calculate_rmse([], [])

        # NaN values
        with self.assertRaises(ValueError):
            calculate_mae([1, np.nan, 3], [1, 2, 3])

    def test_compare_models_dataframe(self):
        preds = {
            "Baseline": self.y_mean,
            "Ridge": self.y_pred,
        }
        df_comp = compare_models(self.y_true, preds)
        self.assertEqual(len(df_comp), 2)
        self.assertIn("Model", df_comp.columns)
        self.assertIn("R2", df_comp.columns)
        self.assertIn("RMSE", df_comp.columns)
        self.assertIn("MAPE_pct", df_comp.columns)
        self.assertIn("Durbin_Watson", df_comp.columns)

    def test_diagnostic_plots_generation(self):
        dates = pd.date_range("2023-01-01", periods=5, freq="MS")
        fig1 = plot_actual_vs_predicted(
            dates=dates,
            y_true=self.y_true,
            y_pred=self.y_pred,
            baseline_pred=self.y_mean,
            split_date="2023-04-01",
        )
        self.assertIsInstance(fig1, plt.Figure)
        plt.close(fig1)

        fig2 = plot_residuals_diagnostic(
            dates=dates,
            y_true=self.y_true,
            y_pred=self.y_pred,
            split_date="2023-04-01",
        )
        self.assertIsInstance(fig2, plt.Figure)
        plt.close(fig2)


if __name__ == "__main__":
    unittest.main()
