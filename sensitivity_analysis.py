import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from backtest import run_single_backtest
from indicators import add_indicators
from metrics import get_metrics


def run_sensitivity_analysis(
    df_test: pd.DataFrame,
    best_params: dict,
    delta: float = 0.20,
) -> pd.DataFrame:
    """
    Análisis de sensibilidad de parámetros (±delta%).

    Para cada parámetro en best_params:
        - Prueba con valor × (1 - delta)  →  -20%
        - Prueba con valor base            →   0% (baseline)
        - Prueba con valor × (1 + delta)  →  +20%

    Reporta Calmar, Sharpe, MaxDD y Return para cada combinación.

    Args:
        df_test:     DataFrame con datos de precio (sin indicadores aún)
        best_params: Diccionario de parámetros óptimos
        delta:       Variación fraccional (default 0.20 = ±20%)

    Returns:
        pd.DataFrame con resultados de sensibilidad
    """
    results = []

    # Baseline primero
    base_prep = add_indicators(df_test, best_params)
    equity_base, _ = run_single_backtest(base_prep, best_params)
    stats_base = get_metrics(equity_base)

    results.append({
        "param":      "BASELINE",
        "variation":  "0%",
        "value":      None,
        "calmar":     stats_base["Calmar"],
        "sharpe":     stats_base["Sharpe"],
        "max_dd":     stats_base["MaxDD"],
        "return_pct": stats_base["Return"],
    })

    print(f"  BASELINE → Calmar: {stats_base['Calmar']:.4f} | Sharpe: {stats_base['Sharpe']:.4f} | MaxDD: {stats_base['MaxDD']:.2f}%")

    # Variar cada parámetro individualmente
    for param_name, base_val in best_params.items():
        for factor, label in [(1 - delta, f"-{int(delta*100)}%"), (1 + delta, f"+{int(delta*100)}%")]:
            new_params = best_params.copy()
            raw_new_val = base_val * factor

            # Mantener enteros para parámetros que deben serlo
            int_params = {"macd_fast", "macd_slow", "macd_sign", "adx_window", "adx_min", "sma_f", "sma_s"}
            new_val = round(raw_new_val) if param_name in int_params else raw_new_val
            new_val = max(new_val, 1)  # No puede ser < 1
            new_params[param_name] = new_val

            # Validar restricciones básicas
            if new_params.get("macd_slow", 999) <= new_params.get("macd_fast", 0):
                results.append({
                    "param": param_name, "variation": label, "value": new_val,
                    "calmar": None, "sharpe": None, "max_dd": None, "return_pct": None,
                })
                continue
            if new_params.get("sma_s", 999) <= new_params.get("sma_f", 0):
                results.append({
                    "param": param_name, "variation": label, "value": new_val,
                    "calmar": None, "sharpe": None, "max_dd": None, "return_pct": None,
                })
                continue

            try:
                test_prep = add_indicators(df_test, new_params)
                equity, _ = run_single_backtest(test_prep, new_params)
                stats = get_metrics(equity)
                results.append({
                    "param":      param_name,
                    "variation":  label,
                    "value":      new_val,
                    "calmar":     stats["Calmar"],
                    "sharpe":     stats["Sharpe"],
                    "max_dd":     stats["MaxDD"],
                    "return_pct": stats["Return"],
                })
                print(f"  {param_name:12s} {label:5s} ({new_val:.4f}) → Calmar: {stats['Calmar']:.4f} | MaxDD: {stats['MaxDD']:.2f}%")
            except Exception as e:
                print(f"  [!] Error en {param_name} {label}: {e}")
                results.append({
                    "param": param_name, "variation": label, "value": new_val,
                    "calmar": None, "sharpe": None, "max_dd": None, "return_pct": None,
                })

    return pd.DataFrame(results)


def plot_sensitivity(df_sensitivity: pd.DataFrame):
    """
    Genera dos figuras:
        1. Impacto en Calmar Ratio por parámetro
        2. Impacto en MaxDD por parámetro
    """
    params = [p for p in df_sensitivity["param"].unique() if p != "BASELINE"]
    n = len(params)
    cols = 3
    rows = (n + cols - 1) // cols

    baseline_calmar = df_sensitivity.loc[df_sensitivity["param"] == "BASELINE", "calmar"].values[0]
    baseline_maxdd  = df_sensitivity.loc[df_sensitivity["param"] == "BASELINE", "max_dd"].values[0]

    fig, axes = plt.subplots(rows, cols, figsize=(14, rows * 3.5))
    axes_flat = axes.flatten() if n > 1 else [axes]

    for i, param in enumerate(params):
        ax = axes_flat[i]
        sub = df_sensitivity[df_sensitivity["param"] == param].dropna(subset=["calmar"])

        variations = sub["variation"].tolist()
        calmars    = sub["calmar"].tolist()
        colors     = ["#ef4444" if c < baseline_calmar else "#22c55e" for c in calmars]

        ax.bar(variations, calmars, color=colors, alpha=0.8, edgecolor="white", linewidth=0.5)
        ax.axhline(baseline_calmar, color="cyan", linestyle="--", linewidth=1.0, label="Baseline")
        ax.set_title(param, fontsize=9, fontweight="bold")
        ax.set_ylabel("Calmar Ratio")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3, axis="y")

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("Análisis de Sensibilidad — Impacto en Calmar Ratio (±20%)", fontsize=13)
    plt.tight_layout()
    plt.savefig("output_Sensitivity_Calmar.png", dpi=150, bbox_inches="tight")
    plt.show()

    # --- Segunda figura: MaxDD ---
    fig2, axes2 = plt.subplots(rows, cols, figsize=(14, rows * 3.5))
    axes_flat2 = axes2.flatten() if n > 1 else [axes2]

    for i, param in enumerate(params):
        ax = axes_flat2[i]
        sub = df_sensitivity[df_sensitivity["param"] == param].dropna(subset=["max_dd"])

        variations = sub["variation"].tolist()
        maxdds     = sub["max_dd"].tolist()
        colors     = ["#ef4444" if m > baseline_maxdd else "#22c55e" for m in maxdds]

        ax.bar(variations, maxdds, color=colors, alpha=0.8, edgecolor="white", linewidth=0.5)
        ax.axhline(baseline_maxdd, color="cyan", linestyle="--", linewidth=1.0, label="Baseline")
        ax.set_title(param, fontsize=9, fontweight="bold")
        ax.set_ylabel("Max Drawdown (%)")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3, axis="y")

    for j in range(i + 1, len(axes_flat2)):
        axes_flat2[j].set_visible(False)

    fig2.suptitle("Análisis de Sensibilidad — Impacto en Max Drawdown (±20%)", fontsize=13)
    plt.tight_layout()
    plt.savefig("output_Sensitivity_MaxDD.png", dpi=150, bbox_inches="tight")
    plt.show()