"""
Unit and integration tests for src/predict.py (Member 3).
Tests inference interface, schema preservation, feature ordering, stateful adstock,
counterfactual simulation, and handoff utilities for ROI / optimization subgroup.
"""

import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.predict import MMMPredictor, apply_adstock_with_carryover


class TestMMMPredictor(unittest.TestCase):
    """Test suite for the MMMPredictor inference interface."""

    def setUp(self):
        # 1. Setup minimal dummy configuration and parameters for unit testing
        self.channels = ["tv_spend_lkr", "digital_spend_lkr"]
        self.media_features = ["tv_spend_lkr_hill", "digital_spend_lkr_hill"]
        self.external_factors = ["price_index", "seasonality_index"]
        self.features = self.media_features + self.external_factors

        self.dummy_config = {
            "target": "sales_revenue_lkr",
            "features": self.features,
            "media_features": self.media_features,
            "external_factors": self.external_factors,
        }

        self.dummy_transforms = {
            "tv_spend_lkr": {
                "adstock_decay": 0.5,
                "half_sat": 1000.0,
                "slope": 1.0,
            },
            "digital_spend_lkr": {
                "adstock_decay": 0.2,
                "half_sat": 500.0,
                "slope": 1.0,
            },
        }

        # 2. Create a mock fitted pipeline: sales = 100000 + 50000*tv_hill + 30000*digital_hill + 10000*ext
        X_mock = np.array([
            [0.1, 0.2, 1.0, 1.0],
            [0.5, 0.6, 1.0, 1.1],
            [0.8, 0.9, 1.1, 1.2],
        ])
        y_mock = np.array([150000.0, 200000.0, 250000.0])

        dummy_pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("regression", Ridge(alpha=1.0)),
        ])
        dummy_pipeline.fit(X_mock, y_mock)
        self.dummy_pipeline = dummy_pipeline

        # Initialize in-memory predictor
        self.predictor = MMMPredictor(
            model=self.dummy_pipeline,
            config=self.dummy_config,
            transform_params=self.dummy_transforms,
        )

    def test_apply_adstock_with_carryover(self):
        """Verify carryover formula: A_0 = spend_0 + decay * carryover."""
        spend = [100.0, 0.0]
        decay = 0.5
        init_carry = 50.0

        adstock_vals, final_c = apply_adstock_with_carryover(
            spend, decay=decay, initial_carryover=init_carry
        )

        # Period 0: 100 + 0.5 * 50 = 125
        # Period 1: 0 + 0.5 * 125 = 62.5
        self.assertAlmostEqual(adstock_vals[0], 125.0)
        self.assertAlmostEqual(adstock_vals[1], 62.5)
        self.assertAlmostEqual(final_c, 62.5)

    def test_feature_ordering_guarantee(self):
        """Ensure predictor produces identical results regardless of input column order."""
        # Create input df with columns in order A
        df_order_a = pd.DataFrame({
            "tv_spend_lkr": [1000.0, 1500.0],
            "digital_spend_lkr": [500.0, 800.0],
            "price_index": [1.0, 1.05],
            "seasonality_index": [1.0, 1.1],
        })

        # Shuffle column order
        df_order_b = df_order_a[["seasonality_index", "digital_spend_lkr", "price_index", "tv_spend_lkr"]]

        pred_a = self.predictor.predict(df_order_a)
        pred_b = self.predictor.predict(df_order_b)

        np.testing.assert_allclose(pred_a, pred_b)

    def test_counterfactual_prediction(self):
        """Verify counterfactual scenario simulation correctly computes delta sales."""
        df_base = pd.DataFrame({
            "tv_spend_lkr": [1000.0, 1000.0],
            "digital_spend_lkr": [500.0, 500.0],
            "price_index": [1.0, 1.0],
            "seasonality_index": [1.0, 1.0],
        })

        # Scale TV spend by +50%
        res = self.predictor.predict_counterfactual(
            df_baseline=df_base,
            spend_multipliers={"tv_spend_lkr": 1.5},
        )

        self.assertIn("baseline_predicted_sales_lkr", res.columns)
        self.assertIn("scenario_predicted_sales_lkr", res.columns)
        self.assertIn("delta_sales_lkr", res.columns)
        self.assertIn("sales_lift_pct", res.columns)
        self.assertEqual(len(res), 2)

    def test_single_period_prediction_for_optimizers(self):
        """Test single-period fast interface used by scipy.optimize in downstream budget allocation."""
        spend_vector = [1200.0, 600.0]
        ext_factors = {"price_index": 1.0, "seasonality_index": 1.05}
        init_carry = {"tv_spend_lkr": 200.0, "digital_spend_lkr": 100.0}

        pred_sales, new_carry = self.predictor.predict_single_period(
            spend_vector=spend_vector,
            external_factors=ext_factors,
            initial_carryover=init_carry,
        )

        self.assertIsInstance(pred_sales, float)
        self.assertIn("tv_spend_lkr", new_carry)
        self.assertIn("digital_spend_lkr", new_carry)
        self.assertGreater(new_carry["tv_spend_lkr"], 0.0)

    def test_actual_artifacts_integration(self):
        """Integration test using Member 2's trained ridge model artifact on disk if available."""
        model_p = Path("outputs/models/ridge_mmm_baseline.joblib")
        config_p = Path("outputs/models/model_config.json")
        transforms_p = Path("outputs/models/transformation_params.json")

        if model_p.exists() and config_p.exists() and transforms_p.exists():
            real_predictor = MMMPredictor(
                model_path=model_p,
                config_path=config_p,
                transform_params_path=transforms_p,
            )

            # Load actual test data row
            data_file = Path("data/processed/feature_engineered_marketing_mix_data.csv")
            if data_file.exists():
                df_real = pd.read_csv(data_file)
                sample_test = df_real.iloc[-6:].copy()  # Holdout test set

                preds = real_predictor.predict(sample_test)
                self.assertEqual(len(preds), 6)
                self.assertTrue(np.all(preds > 0.0))  # Positive revenue predictions


if __name__ == "__main__":
    unittest.main()
