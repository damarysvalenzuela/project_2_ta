"""
sensitivity_analysis.py — ±delta sensitivity on all strategy parameters.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from backtest import run_single_backtest
from metrics import get_metrics
from indicators import add_indicators

INT_PARAMS   = {"ema_fast", "ema_slow", "rsi_window", "macd_fast", "macd_slow",
                "macd_sign", "slope_lookback", "use_regime_filter"}
FLOAT_PARAMS = {"stop_loss", "take_profit", "risk_per_trade", "rsi_high", "rsi_low"}


# ──────────────────────────────────────────────────────────────────────────────

def _vary_param(base_val, delta: float, param_name: str):
    base_val = float(base_val)
    new_val  = base_val * (1.0 + delta)

    if param_name in FLOAT_PARAMS:
        return float(new_val)
    if param_name in INT_PARAMS:
        return max(1, int(round(new_val)))
    # fallback: preserve original type
    if isinstance(base_val, int):
        return max(1, int(round(new_val)))
    return float(new_val)


def run_sensitivity_analysis(
    raw_data: pd.DataFrame,
    base_params: dict,
    delta: float = 0.20,
) -> pd.DataFrame:
    """
    Vary each parameter by ±delta and record performance.
    raw_data must be raw OHLC (no indicators pre-computed).
    """
    results = []

    # ── Baseline ────────────────────────────────────────────────────────────
    try:
        base_prep   = add_indicators(raw_data.copy(), base_params)
        eq, _       = run_single_backtest(base_prep, base_params)
        s           = get_metrics(eq)
    except Exception as e:
        print(f"  [!] Error baseline: {e}")
        s = {"CalmarSimple": 0.0, "Calmar": 0.0, "Sharpe": 0.0,
             "MaxDD": 0.0, "Return": 0.0}

    results.append({
        "param": "BASELINE", "variation": "0%", "value": np.nan,
        "calmar_simple": s["CalmarSimple"],
        "sharpe":        s["Sharpe"],
        "max_dd":        s["MaxDD"],
        "return_pct":    s["Return"],
    })
    print(
        f"  BASELINE → CalmarSimple: {s['CalmarSimple']:.4f} | "
        f"Sharpe: {s['Sharpe']:.4f} | MaxDD: {s['MaxDD']:.2f}%"
    )

    # ── Per-parameter sweeps ─────────────────────────────────────────────────
    for param, base_val in base_params.items():
        if not isinstance(base_val, (int, float)):
            continue

        for sign, label in [(-1, f"-{int(delta*100)}%"), (+1, f"+{int(delta*100)}%")]:
            new_val     = _vary_param(base_val, sign * delta, param)
            test_params = {**base_params, param: new_val}

            # Logical guard-rails
            if test_params.get("ema_slow", 999) <= test_params.get("ema_fast", 0):
                results.append(_nan_row(param, label, new_val))
                continue
            if test_params.get("macd_slow", 999) <= test_params.get("macd_fast", 0):
                results.append(_nan_row(param, label, new_val))
                continue

            try:
                # Re-compute indicators when an indicator param is varied
                if param in INT_PARAMS:
                    prepared = add_indicators(raw_data.copy(), test_params)
                else:
                    prepared = add_indicators(raw_data.copy(), test_params)

                eq, _ = run_single_backtest(prepared, test_params)
                s     = get_metrics(eq)

                print(
                    f"  {param:18s} {label:5s} ({new_val:.5g}) → "
                    f"CalmarSimple: {s['CalmarSimple']:.4f} | MaxDD: {s['MaxDD']:.2f}%"
                )
                results.append({
                    "param":       param,
                    "variation":   label,
                    "value":       float(new_val),
                    "calmar_simple": s["CalmarSimple"],
                    "sharpe":      s["Sharpe"],
                    "max_dd":      s["MaxDD"],
                    "return_pct":  s["Return"],
                })

            except Exception as e:
                print(f"  [!] Error {param} {label}: {e}")
                results.append(_nan_row(param, label, new_val))

    return pd.DataFrame(results)


def _nan_row(param, label, val):
    return {
        "param": param, "variation": label, "value": float(val),
        "calmar_simple": np.nan, "sharpe": np.nan,
        "max_dd": np.nan, "return_pct": np.nan,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────────

def plot_sensitivity(df: pd.DataFrame):
    df_plot = df[df["param"] != "BASELINE"].copy()
    if df_plot.empty:
        return

    b_calmar = float(df.loc[df["param"] == "BASELINE", "calmar_simple"].values[0])
    b_maxdd  = float(df.loc[df["param"] == "BASELINE", "max_dd"].values[0])

    df_plot["label"] = df_plot["param"] + " " + df_plot["variation"]
    _bar(df_plot, "calmar_simple", b_calmar,
         "CalmarSimple — Sensibilidad ±20%", True,  "output_Sensitivity_Calmar.png")
    _bar(df_plot, "max_dd",        b_maxdd,
         "Max Drawdown (%) — Sensibilidad ±20%", False, "output_Sensitivity_MaxDD.png")


def _bar(
    df: pd.DataFrame,
    metric: str,
    baseline: float,
    title: str,
    higher_is_better: bool,
    filename: str,
):
    df = df.dropna(subset=[metric]).copy()
    if df.empty:
        return

    vals   = df[metric].values
    labels = df["label"].values
    colors = [
        "#22c55e"
        if (v >= baseline if higher_is_better else v <= baseline)
        else "#ef4444"
        for v in vals
    ]

    fig, ax = plt.subplots(figsize=(10, max(6, len(labels) * 0.4)))
    ax.barh(labels, vals, color=colors, alpha=0.85, edgecolor="white")
    ax.axvline(
        baseline, color="#1e40af", linestyle="--", linewidth=1.5,
        label=f"Baseline: {baseline:.4f}",
    )
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel(metric.replace("_", " ").title())
    ax.legend(fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Guardado: {filename}")