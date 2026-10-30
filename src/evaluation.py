"""
Module: evaluation.py
Author: Member 3 (Model Evaluation, Diagnostics, Validation & Handoff)
Description:
    Production-grade model evaluation, diagnostic metrics, statistical tests,
    and visualization utilities for Marketing Mix Modelling (MMM).
    Calculates R², RMSE (LKR), safe MAPE (zero-denominator protected),
    residual time-series diagnostics, and Durbin-Watson serial correlation.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _validate_inputs(
    y_true: Union[np.ndarray, pd.Series, list],
    y_pred: Union[np.ndarray, pd.Series, list],
) -> Tuple[np.ndarray, np.ndarray]:
    """Validate and convert inputs to 1D float numpy arrays."""
    y_true_arr = np.asarray(y_true, dtype=float).squeeze()
    y_pred_arr = np.asarray(y_pred, dtype=float).squeeze()

    if y_true_arr.ndim != 1 or y_pred_arr.ndim != 1:
        raise ValueError(
            f"Expected 1D arrays for y_true and y_pred, got shapes "
            f"{y_true_arr.shape} and {y_pred_arr.shape}."
        )

    if len(y_true_arr) != len(y_pred_arr):
        raise ValueError(
            f"Length mismatch: len(y_true)={len(y_true_arr)} != "
            f"len(y_pred)={len(y_pred_arr)}."
        )

    if len(y_true_arr) == 0:
        raise ValueError("Inputs cannot be empty.")

    if not np.all(np.isfinite(y_true_arr)) or not np.all(np.isfinite(y_pred_arr)):
        raise ValueError("Inputs contain NaN or infinite values.")

    return y_true_arr, y_pred_arr


def calculate_residuals(
    y_true: Union[np.ndarray, pd.Series, list],
    y_pred: Union[np.ndarray, pd.Series, list],
) -> np.ndarray:
    """
    Compute residual errors (Actual - Predicted) in LKR.

    Parameters
    ----------
    y_true : array-like
        Ground truth actual values (e.g. sales_revenue_lkr).
    y_pred : array-like
        Model predicted values.

    Returns
    -------
    np.ndarray
        Residuals: y_true - y_pred.
    """
    y_t, y_p = _validate_inputs(y_true, y_pred)
    return y_t - y_p


def calculate_mae(
    y_true: Union[np.ndarray, pd.Series, list],
    y_pred: Union[np.ndarray, pd.Series, list],
) -> float:
    """
    Calculate Mean Absolute Error (MAE) in LKR.

    MAE = (1/N) * sum(|y_true - y_pred|)
    """
    residuals = calculate_residuals(y_true, y_pred)
    return float(np.mean(np.abs(residuals)))


def calculate_rmse(
    y_true: Union[np.ndarray, pd.Series, list],
    y_pred: Union[np.ndarray, pd.Series, list],
) -> float:
    """
    Calculate Root Mean Squared Error (RMSE) in LKR.

    RMSE = sqrt((1/N) * sum((y_true - y_pred)^2))
    """
    residuals = calculate_residuals(y_true, y_pred)
    return float(np.sqrt(np.mean(residuals**2)))


def calculate_r2(
    y_true: Union[np.ndarray, pd.Series, list],
    y_pred: Union[np.ndarray, pd.Series, list],
) -> float:
    """
    Calculate the coefficient of determination (R²).

    R² = 1 - (SS_res / SS_tot)
    where SS_res = sum((y_true - y_pred)^2)
          SS_tot = sum((y_true - mean(y_true))^2)

    Edge case:
    If SS_tot == 0 (constant target), returns 1.0 if SS_res == 0 else 0.0.
    """
    y_t, y_p = _validate_inputs(y_true, y_pred)
    ss_res = np.sum((y_t - y_p) ** 2)
    ss_tot = np.sum((y_t - np.mean(y_t)) ** 2)

    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0

    return float(1.0 - (ss_res / ss_tot))


def calculate_mape(
    y_true: Union[np.ndarray, pd.Series, list],
    y_pred: Union[np.ndarray, pd.Series, list],
    epsilon: float = 1e-8,
) -> float:
    """
    Calculate safe Mean Absolute Percentage Error (MAPE) in percent (0 - 100%).

    Guards against division-by-zero when y_true contains zero or near-zero values
    by replacing denominators smaller than epsilon with epsilon.

    Parameters
    ----------
    y_true : array-like
        Ground truth actual values.
    y_pred : array-like
        Model predicted values.
    epsilon : float, default 1e-8
        Numerical floor to protect against division by zero.

    Returns
    -------
    float
        MAPE expressed as a percentage (e.g. 1.02 for 1.02%).
    """
    y_t, y_p = _validate_inputs(y_true, y_pred)
    denominator = np.maximum(np.abs(y_t), epsilon)
    absolute_percentage_errors = np.abs((y_t - y_p) / denominator)
    return float(np.mean(absolute_percentage_errors) * 100.0)


def calculate_durbin_watson(
    residuals: Union[np.ndarray, pd.Series, list],
) -> float:
    """
    Compute the Durbin-Watson statistic to test for first-order serial autocorrelation
    in time-series residuals.

    DW = sum_{t=2}^T (e_t - e_{t-1})^2 / sum_{t=1}^T (e_t)^2

    Interpretation:
    - DW ~ 2.0 : No first-order serial correlation (ideal for time series / MMM).
    - DW < 1.5 : Potential positive serial correlation (residual memory).
    - DW > 2.5 : Potential negative serial correlation.

    Parameters
    ----------
    residuals : array-like
        Time-ordered residual errors.

    Returns
    -------
    float
        Durbin-Watson statistic (between 0.0 and 4.0).
    """
    res = np.asarray(residuals, dtype=float).squeeze()
    if res.ndim != 1 or len(res) < 2:
        raise ValueError("Residuals must be a 1D array with at least 2 observations.")

    diff = np.diff(res)
    sum_sq_diff = np.sum(diff**2)
    sum_sq_res = np.sum(res**2)

    if sum_sq_res == 0.0:
        return 2.0  # Zero error represents neutral autocorrelation

    return float(sum_sq_diff / sum_sq_res)


def calculate_all_metrics(
    y_true: Union[np.ndarray, pd.Series, list],
    y_pred: Union[np.ndarray, pd.Series, list],
    epsilon: float = 1e-8,
) -> Dict[str, float]:
    """
    Calculate the complete suite of evaluation metrics for a model.

    Returns
    -------
    dict
        {
            "MAE": float (in LKR),
            "RMSE": float (in LKR),
            "R2": float (0-1 ideal, can be negative for poor models),
            "MAPE_pct": float (percentage),
            "Durbin_Watson": float (autocorrelation test)
        }
    """
    residuals = calculate_residuals(y_true, y_pred)
    return {
        "MAE": calculate_mae(y_true, y_pred),
        "RMSE": calculate_rmse(y_true, y_pred),
        "R2": calculate_r2(y_true, y_pred),
        "MAPE_pct": calculate_mape(y_true, y_pred, epsilon=epsilon),
        "Durbin_Watson": calculate_durbin_watson(residuals),
    }


def compare_models(
    y_true: Union[np.ndarray, pd.Series, list],
    predictions_dict: Dict[str, Union[np.ndarray, pd.Series, list]],
) -> pd.DataFrame:
    """
    Compare multiple models side-by-side using standard MMM evaluation metrics.

    Parameters
    ----------
    y_true : array-like
        Ground truth actual values.
    predictions_dict : dict
        Mapping of model_name -> predicted_array.

    Returns
    -------
    pd.DataFrame
        Table summarizing MAE, RMSE, R2, MAPE_pct, and Durbin_Watson per model.
    """
    rows = []
    for model_name, y_pred in predictions_dict.items():
        metrics = calculate_all_metrics(y_true, y_pred)
        rows.append({"Model": model_name, **metrics})

    df_metrics = pd.DataFrame(rows)
    return df_metrics


# ==============================================================================
# Model Diagnostic Plots
# ==============================================================================

def plot_actual_vs_predicted(
    dates: Union[pd.Series, list, np.ndarray],
    y_true: Union[np.ndarray, pd.Series, list],
    y_pred: Union[np.ndarray, pd.Series, list],
    baseline_pred: Optional[Union[np.ndarray, pd.Series, list]] = None,
    split_date: Optional[str] = "2025-07-01",
    model_name: str = "Ridge Regression",
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (12, 6),
) -> plt.Figure:
    """
    Generate an Actual vs. Predicted monthly line chart with holdout shading.

    Parameters
    ----------
    dates : array-like of dates/strings
        Monthly timestamps in chronological order.
    y_true : array-like
        Actual sales revenue in LKR.
    y_pred : array-like
        Model predicted sales revenue in LKR.
    baseline_pred : array-like, optional
        Baseline or naive benchmark predictions for visual comparison.
    split_date : str, optional
        Threshold date separating Train and Holdout Test periods (default: '2025-07-01').
    model_name : str, default 'Ridge Regression'
        Name of model for chart title and legend.
    save_path : str or Path, optional
        Path to save figure. If None, figure is returned without saving.
    figsize : tuple, default (12, 6)
        Figure size dimensions.

    Returns
    -------
    plt.Figure
    """
    dt_series = pd.to_datetime(dates)
    y_t, y_p = _validate_inputs(y_true, y_pred)

    fig, ax = plt.subplots(figsize=figsize, dpi=300)

    # Convert to Millions LKR for clean readable axis
    y_t_m = y_t / 1e6
    y_p_m = y_p / 1e6

    # Plot Actual vs Predicted
    ax.plot(
        dt_series,
        y_t_m,
        label="Actual Sales Revenue",
        color="#1E293B",
        linewidth=2.5,
        marker="o",
        markersize=5,
        zorder=4,
    )
    ax.plot(
        dt_series,
        y_p_m,
        label=f"Predicted ({model_name})",
        color="#2563EB",
        linewidth=2.2,
        linestyle="--",
        marker="s",
        markersize=4.5,
        zorder=5,
    )

    if baseline_pred is not None:
        b_arr = np.asarray(baseline_pred, dtype=float).squeeze() / 1e6
        ax.plot(
            dt_series,
            b_arr,
            label="Mean Baseline",
            color="#94A3B8",
            linewidth=1.5,
            linestyle=":",
            zorder=3,
        )

    # Shading for Holdout / Test set
    if split_date:
        split_dt = pd.to_datetime(split_date)
        if dt_series.min() <= split_dt <= dt_series.max():
            ax.axvline(
                split_dt,
                color="#DC2626",
                linestyle="--",
                linewidth=1.5,
                alpha=0.85,
                label=f"Train/Test Cutoff ({split_date[:7]})",
            )
            ax.axvspan(
                split_dt,
                dt_series.max(),
                color="#FEE2E2",
                alpha=0.45,
                label="Holdout Test Set (6 Months)",
            )

    # Annotate summary performance metrics on plot
    r2_val = calculate_r2(y_t, y_p)
    rmse_val = calculate_rmse(y_t, y_p) / 1e6
    mape_val = calculate_mape(y_t, y_p)
    metrics_str = (
        f"Evaluation Metrics:\n"
        f"• R² = {r2_val:.3f}\n"
        f"• RMSE = {rmse_val:.2f}M LKR\n"
        f"• MAPE = {mape_val:.2f}%"
    )
    ax.text(
        0.02,
        0.95,
        metrics_str,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(
            boxstyle="round,pad=0.6",
            facecolor="white",
            edgecolor="#CBD5E1",
            alpha=0.95,
            linewidth=1.2,
        ),
    )

    ax.set_title(
        f"Marketing Mix Model: Actual vs. Predicted Sales Revenue ({model_name})",
        fontsize=14,
        fontweight="bold",
        pad=15,
        color="#0F172A",
    )
    ax.set_xlabel("Month", fontsize=11, fontweight="semibold", labelpad=8)
    ax.set_ylabel("Sales Revenue (Million LKR)", fontsize=11, fontweight="semibold", labelpad=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45, ha="right")
    ax.grid(True, linestyle="--", alpha=0.5, color="#E2E8F0")
    ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#CBD5E1", fontsize=9.5)
    plt.tight_layout()

    if save_path:
        save_p = Path(save_path)
        save_p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_p, dpi=300, bbox_inches="tight")

    return fig


def plot_residuals_diagnostic(
    dates: Union[pd.Series, list, np.ndarray],
    y_true: Union[np.ndarray, pd.Series, list],
    y_pred: Union[np.ndarray, pd.Series, list],
    split_date: Optional[str] = "2025-07-01",
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (14, 8),
) -> plt.Figure:
    """
    Generate comprehensive residual diagnostic plots:
    1. Residuals over time (LKR Millions) with ±1.96 standard deviation bounds.
    2. Residuals vs. Fitted Values (heteroscedasticity / non-linearity check).
    3. Residual distribution histogram with KDE (normality check).

    Parameters
    ----------
    dates : array-like
        Timestamps in chronological order.
    y_true : array-like
        Actual sales revenue in LKR.
    y_pred : array-like
        Model predicted sales revenue in LKR.
    split_date : str, optional
        Cutoff date for holdout evaluation.
    save_path : str or Path, optional
        Filepath to save figure.
    figsize : tuple, default (14, 8)
        Dimensions of the grid figure.

    Returns
    -------
    plt.Figure
    """
    dt_series = pd.to_datetime(dates)
    y_t, y_p = _validate_inputs(y_true, y_pred)
    res_m = calculate_residuals(y_t, y_p) / 1e6
    y_p_m = y_p / 1e6

    mean_res = float(np.mean(res_m))
    std_res = float(np.std(res_m))
    dw_stat = calculate_durbin_watson(res_m)

    fig = plt.figure(figsize=figsize, dpi=300)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1.0], hspace=0.35, wspace=0.25)

    ax1 = fig.add_subplot(gs[0, :])  # Residuals over time across full width
    ax2 = fig.add_subplot(gs[1, 0])  # Residuals vs Predicted
    ax3 = fig.add_subplot(gs[1, 1])  # Residual Distribution

    # --- Plot 1: Residuals Over Time ---
    ax1.plot(
        dt_series,
        res_m,
        color="#2563EB",
        marker="o",
        linewidth=1.8,
        markersize=5,
        label="Monthly Residual (Actual - Predicted)",
    )
    ax1.axhline(0, color="#0F172A", linestyle="-", linewidth=1.2, alpha=0.8)
    ax1.axhline(
        mean_res + 1.96 * std_res,
        color="#DC2626",
        linestyle="--",
        linewidth=1.0,
        alpha=0.75,
        label=f"+1.96 SD (+{1.96 * std_res:.2f}M)",
    )
    ax1.axhline(
        mean_res - 1.96 * std_res,
        color="#DC2626",
        linestyle="--",
        linewidth=1.0,
        alpha=0.75,
        label=f"-1.96 SD (-{1.96 * std_res:.2f}M)",
    )

    if split_date:
        split_dt = pd.to_datetime(split_date)
        if dt_series.min() <= split_dt <= dt_series.max():
            ax1.axvline(
                split_dt,
                color="#EA580C",
                linestyle=":",
                linewidth=1.5,
                label=f"Holdout Cutoff ({split_date[:7]})",
            )
            ax1.axvspan(
                split_dt,
                dt_series.max(),
                color="#FFF7ED",
                alpha=0.5,
            )

    ax1.set_title(
        f"Residual Errors Over Time (Durbin-Watson = {dw_stat:.2f})",
        fontsize=12,
        fontweight="bold",
        color="#0F172A",
    )
    ax1.set_xlabel("Month", fontsize=10, fontweight="semibold")
    ax1.set_ylabel("Residual (Million LKR)", fontsize=10, fontweight="semibold")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax1.grid(True, linestyle="--", alpha=0.5, color="#E2E8F0")
    ax1.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#CBD5E1", fontsize=9)

    # --- Plot 2: Residuals vs. Fitted Values ---
    ax2.scatter(
        y_p_m,
        res_m,
        color="#0284C7",
        alpha=0.8,
        edgecolors="white",
        s=60,
    )
    ax2.axhline(0, color="#0F172A", linestyle="-", linewidth=1.2, alpha=0.8)
    ax2.set_title("Residuals vs. Fitted Sales Revenue", fontsize=12, fontweight="bold", color="#0F172A")
    ax2.set_xlabel("Fitted Revenue (Million LKR)", fontsize=10, fontweight="semibold")
    ax2.set_ylabel("Residual (Million LKR)", fontsize=10, fontweight="semibold")
    ax2.grid(True, linestyle="--", alpha=0.5, color="#E2E8F0")

    # --- Plot 3: Residual Distribution ---
    counts, bins, _ = ax3.hist(
        res_m,
        bins=10,
        density=True,
        color="#93C5FD",
        edgecolor="#1E40AF",
        alpha=0.75,
        label="Residual Histogram",
    )
    # Overlay theoretical normal distribution curve
    x_grid = np.linspace(np.min(res_m) - std_res, np.max(res_m) + std_res, 100)
    normal_pdf = (1.0 / (std_res * np.sqrt(2 * np.pi))) * np.exp(
        -0.5 * ((x_grid - mean_res) / std_res) ** 2
    )
    ax3.plot(x_grid, normal_pdf, color="#1E3A8A", linewidth=2.0, label="Fitted Normal")

    ax3.set_title("Residual Error Distribution", fontsize=12, fontweight="bold", color="#0F172A")
    ax3.set_xlabel("Residual (Million LKR)", fontsize=10, fontweight="semibold")
    ax3.set_ylabel("Density", fontsize=10, fontweight="semibold")
    ax3.grid(True, linestyle="--", alpha=0.5, color="#E2E8F0")
    ax3.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#CBD5E1", fontsize=9)

    fig.suptitle(
        "Marketing Mix Model: Residual Diagnostics & Autocorrelation Analysis",
        fontsize=14,
        fontweight="bold",
        y=0.98,
        color="#0F172A",
    )

    if save_path:
        save_p = Path(save_path)
        save_p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_p, dpi=300, bbox_inches="tight")

    return fig


def run_evaluation_pipeline(
    data_path: Union[str, Path] = "data/processed/feature_engineered_marketing_mix_data.csv",
    predictions_path: Union[str, Path] = "outputs/tables/model_predictions.csv",
    figures_dir: Union[str, Path] = "outputs/figures",
    tables_dir: Union[str, Path] = "outputs/tables",
) -> pd.DataFrame:
    """
    Run full end-to-end evaluation pipeline:
    1. Loads actual data and test predictions.
    2. Calculates holdout metrics (MAE, RMSE, R2, safe MAPE, Durbin-Watson).
    3. Produces diagnostic plots in outputs/figures/.
    4. Returns model comparison DataFrame.
    """
    from src.predict import MMMPredictor

    fig_dir = Path(figures_dir)
    tbl_dir = Path(tables_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    tbl_dir.mkdir(parents=True, exist_ok=True)

    # Load 36-month dataset
    df = pd.read_csv(data_path)
    predictor = MMMPredictor()

    # Generate full-series predictions
    df["predicted_sales_lkr"] = predictor.predict(df)

    # Save diagnostic plots
    plot_actual_vs_predicted(
        dates=df["month"],
        y_true=df["sales_revenue_lkr"],
        y_pred=df["predicted_sales_lkr"],
        split_date="2025-07-01",
        save_path=fig_dir / "model_actual_vs_predicted.png",
    )

    plot_residuals_diagnostic(
        dates=df["month"],
        y_true=df["sales_revenue_lkr"],
        y_pred=df["predicted_sales_lkr"],
        split_date="2025-07-01",
        save_path=fig_dir / "model_residuals_diagnostic.png",
    )

    # Load holdout test predictions to compare models
    df_preds = pd.read_csv(predictions_path)
    models_dict = {
        "Mean baseline": df_preds["baseline_sales_lkr"].values,
        "OLS regression": df_preds["ols_predicted_sales_lkr"].values,
        "Ridge regression": df_preds["ridge_predicted_sales_lkr"].values,
    }

    comp_df = compare_models(df_preds["actual_sales_lkr"].values, models_dict)
    return comp_df


if __name__ == "__main__":
    print("=" * 70)
    print("Running Marketing Mix Model Evaluation Pipeline (Member 3)...")
    print("=" * 70)
    metrics_summary = run_evaluation_pipeline()
    print("\nModel Comparison Table (6-Month Holdout Test Period: Jul-Dec 2025):")
    print(metrics_summary.to_string(index=False))
    print("\nDiagnostic plots saved to outputs/figures/:")
    print(" - outputs/figures/model_actual_vs_predicted.png")
    print(" - outputs/figures/model_residuals_diagnostic.png")
    print("=" * 70)

