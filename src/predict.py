"""
Module: predict.py
Author: Member 3 (Model Evaluation, Diagnostics, Validation & Handoff)
Description:
    Production-ready Inference and Counterfactual Prediction Interface for
    the Marketing Mix Model (MMM). Preserves:
    1. Fitted standard scaler and model weights (from ridge_mmm_baseline.joblib)
    2. Adstock carryover memory and state across consecutive months
    3. Hill saturation parameters derived during training
    4. Exact feature schema and column ordering for the downstream
       ROI / Budget Optimization team.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd

from src.transforms import geometric_adstock, hill_saturation


def apply_adstock_with_carryover(
    spend: Union[np.ndarray, pd.Series, list],
    decay: float,
    initial_carryover: float = 0.0,
) -> Tuple[np.ndarray, float]:
    """
    Apply geometric adstock with explicit initial carryover state.

    Formula:
        A_0 = spend_0 + decay * initial_carryover
        A_t = spend_t + decay * A_{t-1}

    Parameters
    ----------
    spend : array-like
        Chronological media spend.
    decay : float
        Decay carryover rate (0 <= decay < 1).
    initial_carryover : float, default 0.0
        Adstock carryover value from the period immediately preceding spend[0].

    Returns
    -------
    adstock_values : np.ndarray
        Adstock-transformed series.
    final_carryover : float
        Carryover state at the final time step, available for subsequent periods.
    """
    spend_arr = np.asarray(spend, dtype=float).squeeze()
    if spend_arr.ndim == 0:
        spend_arr = spend_arr.reshape(1)

    if not (0.0 <= decay < 1.0):
        raise ValueError(f"Decay must be between 0.0 and 1.0, got {decay}.")

    out = np.zeros(len(spend_arr), dtype=float)
    carry = float(initial_carryover)

    for t, x in enumerate(spend_arr):
        carry = float(x) + decay * carry
        out[t] = carry

    final_carry = carry
    return out, final_carry


class MMMPredictor:
    """
    Production-ready inference interface for the fitted MMM Ridge Regression model.

    Designed for handoff to downstream ROI Analysis and Budget Optimization subgroups.
    Guarantees consistent feature scaling, transformation parameter preservation,
    and strict feature column ordering.
    """

    DEFAULT_MODEL_PATH = Path("outputs/models/ridge_mmm_baseline.joblib")
    DEFAULT_CONFIG_PATH = Path("outputs/models/model_config.json")
    DEFAULT_TRANSFORMS_PATH = Path("outputs/models/transformation_params.json")

    def __init__(
        self,
        model: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
        transform_params: Optional[Dict[str, Any]] = None,
        model_path: Optional[Union[str, Path]] = None,
        config_path: Optional[Union[str, Path]] = None,
        transform_params_path: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize the predictor either with in-memory objects or by loading
        saved artifacts from disk.

        Parameters
        ----------
        model : object, optional
            Fitted scikit-learn Pipeline (StandardScaler + Ridge).
        config : dict, optional
            Model configuration dictionary containing feature lists.
        transform_params : dict, optional
            Channel transformation parameters (decay, half_sat, slope).
        model_path : str or Path, optional
            Path to ridge_mmm_baseline.joblib.
        config_path : str or Path, optional
            Path to model_config.json.
        transform_params_path : str or Path, optional
            Path to transformation_params.json.
        """
        # Load or assign config
        if config is not None:
            self.config = config
        else:
            cfg_p = Path(config_path or self.DEFAULT_CONFIG_PATH)
            if not cfg_p.exists():
                raise FileNotFoundError(f"Model config file not found: {cfg_p}")
            with open(cfg_p, "r", encoding="utf-8") as f:
                self.config = json.load(f)

        # Load or assign transform params
        if transform_params is not None:
            self.transform_params = transform_params
        else:
            tr_p = Path(transform_params_path or self.DEFAULT_TRANSFORMS_PATH)
            if not tr_p.exists():
                raise FileNotFoundError(f"Transformation params file not found: {tr_p}")
            with open(tr_p, "r", encoding="utf-8") as f:
                self.transform_params = json.load(f)

        # Load or assign model pipeline
        if model is not None:
            self.model = model
        else:
            m_p = Path(model_path or self.DEFAULT_MODEL_PATH)
            if not m_p.exists():
                raise FileNotFoundError(f"Model artifact not found: {m_p}")
            self.model = joblib.load(m_p)

        # Cache schema properties
        self.features: List[str] = self.config["features"]
        self.media_features: List[str] = self.config["media_features"]
        self.external_factors: List[str] = self.config["external_factors"]
        self.target: str = self.config.get("target", "sales_revenue_lkr")
        self.channels: List[str] = list(self.transform_params.keys())

    def transform_media_spend(
        self,
        spend_df: pd.DataFrame,
        initial_carryover: Optional[Dict[str, float]] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Transform raw media spend columns into Adstock + Hill Saturation features.

        Parameters
        ----------
        spend_df : pd.DataFrame
            DataFrame containing raw media spend columns (e.g. tv_spend_lkr).
        initial_carryover : dict, optional
            Mapping of channel -> initial carryover value from previous period.

        Returns
        -------
        transformed_media_df : pd.DataFrame
            DataFrame containing saturated media features (e.g. tv_spend_lkr_hill).
        final_carryover : dict
            Updated carryover states for each channel at the end of the input series.
        """
        carryover_dict = initial_carryover or {}
        media_transformed = pd.DataFrame(index=spend_df.index)
        new_carryover = {}

        for channel, params in self.transform_params.items():
            if channel not in spend_df.columns:
                raise ValueError(
                    f"Required media channel '{channel}' missing from input spend DataFrame."
                )

            decay = float(params["adstock_decay"])
            half_sat = float(params["half_sat"])
            slope = float(params.get("slope", 1.0))

            init_carry = float(carryover_dict.get(channel, 0.0))
            adstock_vals, final_c = apply_adstock_with_carryover(
                spend_df[channel].values, decay=decay, initial_carryover=init_carry
            )
            new_carryover[channel] = final_c

            # Apply Hill Saturation
            saturated_vals = hill_saturation(
                adstock_vals, half_sat=half_sat, slope=slope
            )
            feature_col_name = f"{channel}_hill"
            media_transformed[feature_col_name] = saturated_vals

        return media_transformed, new_carryover

    def prepare_feature_matrix(
        self,
        df: pd.DataFrame,
        initial_carryover: Optional[Dict[str, float]] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Assemble and strictly order features for model inference.

        If input already contains precomputed '_hill' features, they are used directly.
        Otherwise, raw spend columns are transformed with adstock & Hill saturation.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame with raw media spend (or pre-transformed media) and external factors.
        initial_carryover : dict, optional
            Carryover values for adstock continuation.

        Returns
        -------
        X : pd.DataFrame
            Strictly ordered feature matrix matching training schema exactly.
        final_carryover : dict
            Ending carryover state per channel.
        """
        final_carryover: Dict[str, float] = {}

        # Check whether pre-transformed hill features are supplied
        has_hill_features = all(mf in df.columns for mf in self.media_features)

        if has_hill_features:
            media_df = df[self.media_features].copy()
        else:
            media_df, final_carryover = self.transform_media_spend(
                df, initial_carryover=initial_carryover
            )

        # Check external factors
        for ext in self.external_factors:
            if ext not in df.columns:
                raise ValueError(
                    f"Required external factor '{ext}' missing from input DataFrame."
                )

        external_df = df[self.external_factors].copy()

        # Combine and strictly enforce the exact 14-feature column ordering
        full_features_df = pd.concat([media_df, external_df], axis=1)
        X = full_features_df[self.features].astype(float)

        return X, final_carryover

    def predict(
        self,
        df: pd.DataFrame,
        initial_carryover: Optional[Dict[str, float]] = None,
        return_carryover: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, float]]]:
        """
        Generate sales revenue predictions (in LKR) for the given input data.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing media spend and external factor columns.
        initial_carryover : dict, optional
            Initial carryover state per channel.
        return_carryover : bool, default False
            If True, returns a tuple (predictions, final_carryover_dict).

        Returns
        -------
        predictions : np.ndarray
            Predicted sales revenue in LKR.
        final_carryover : dict (optional)
            End-of-series carryover state.
        """
        X, final_carryover = self.prepare_feature_matrix(
            df, initial_carryover=initial_carryover
        )
        # Pass DataFrame if pipeline was fitted with feature names, else numpy array
        scaler_step = getattr(self.model, "named_steps", {}).get("scaler", None)
        if hasattr(self.model, "feature_names_in_") or hasattr(scaler_step, "feature_names_in_"):
            preds = self.model.predict(X)
        else:
            preds = self.model.predict(X.to_numpy())
        preds_1d = np.asarray(preds, dtype=float).reshape(-1)

        if return_carryover:
            return preds_1d, final_carryover
        return preds_1d

    def predict_counterfactual(
        self,
        df_baseline: pd.DataFrame,
        spend_multipliers: Optional[Dict[str, float]] = None,
        spend_deltas: Optional[Dict[str, float]] = None,
        initial_carryover: Optional[Dict[str, float]] = None,
    ) -> pd.DataFrame:
        """
        Simulate counterfactual marketing budget scenarios and compute sales delta.

        Parameters
        ----------
        df_baseline : pd.DataFrame
            Baseline input DataFrame with original spend and external factors.
        spend_multipliers : dict, optional
            Multiplier for specific channels (e.g. {'tv_spend_lkr': 1.20} for +20% spend).
        spend_deltas : dict, optional
            Absolute LKR change for specific channels (e.g. {'tv_spend_lkr': 500000}).
        initial_carryover : dict, optional
            Initial carryover state per channel.

        Returns
        -------
        pd.DataFrame
            DataFrame containing baseline sales, scenario sales, absolute delta,
            and percentage sales lift.
        """
        df_scenario = df_baseline.copy()
        mults = spend_multipliers or {}
        deltas = spend_deltas or {}

        for ch in self.channels:
            if ch in df_scenario.columns:
                m = float(mults.get(ch, 1.0))
                d = float(deltas.get(ch, 0.0))
                df_scenario[ch] = np.maximum(df_scenario[ch].astype(float) * m + d, 0.0)

        baseline_preds = self.predict(df_baseline, initial_carryover=initial_carryover)
        scenario_preds = self.predict(df_scenario, initial_carryover=initial_carryover)

        delta = scenario_preds - baseline_preds
        pct_change = (delta / np.maximum(np.abs(baseline_preds), 1e-8)) * 100.0

        res_df = pd.DataFrame(
            {
                "baseline_predicted_sales_lkr": baseline_preds,
                "scenario_predicted_sales_lkr": scenario_preds,
                "delta_sales_lkr": delta,
                "sales_lift_pct": pct_change,
            },
            index=df_baseline.index,
        )

        return res_df

    def predict_single_period(
        self,
        spend_vector: Union[Dict[str, float], List[float], np.ndarray],
        external_factors: Dict[str, float],
        initial_carryover: Optional[Dict[str, float]] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Lightweight single-period inference method specifically optimized for
        budget allocation solvers in downstream notebooks (e.g. scipy.optimize).

        Parameters
        ----------
        spend_vector : dict, list, or 1D array
            Spend amounts per channel for the single decision period.
            If array/list, assumes channel order matching self.channels.
        external_factors : dict
            Values for external macroeconomic/seasonal variables for that period.
        initial_carryover : dict, optional
            Carryover state from the prior month.

        Returns
        -------
        predicted_sales : float
            Predicted sales revenue in LKR.
        new_carryover : dict
            Carryover state produced by this period's spend.
        """
        # Convert spend_vector to dict if list or array
        if isinstance(spend_vector, (list, np.ndarray)):
            spend_dict = {
                ch: float(val) for ch, val in zip(self.channels, spend_vector)
            }
        elif isinstance(spend_vector, dict):
            spend_dict = {ch: float(spend_vector.get(ch, 0.0)) for ch in self.channels}
        else:
            raise TypeError("spend_vector must be a dict, list, or np.ndarray.")

        single_row = {**spend_dict, **external_factors}
        single_df = pd.DataFrame([single_row])

        preds, new_carry = self.predict(
            single_df, initial_carryover=initial_carryover, return_carryover=True
        )

        pred_val = float(preds[0])
        return pred_val, new_carry


def predict_sales(
    df: pd.DataFrame,
    model: Optional[Any] = None,
    config: Optional[Dict[str, Any]] = None,
    transform_params: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """
    Convenience functional interface for generating MMM sales predictions.

    Parameters
    ----------
    df : pd.DataFrame
        Input data containing media spend and external features.
    model : Pipeline, optional
        Pre-loaded fitted model pipeline.
    config : dict, optional
        Pre-loaded model configuration.
    transform_params : dict, optional
        Pre-loaded transformation parameters.

    Returns
    -------
    np.ndarray
        Array of sales predictions in LKR.
    """
    predictor = MMMPredictor(
        model=model, config=config, transform_params=transform_params
    )
    return predictor.predict(df)
