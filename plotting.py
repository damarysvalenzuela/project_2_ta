"""
plotting.py — All charting helpers for the trading project.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────

def _pick_first_existing_col(df: pd.DataFrame, candidates: list) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def plot_equity_curve(equity: pd.Series, title: str, filename: str):
    if equity is None or len(equity) < 2:
        print(f"[plot_equity_curve] Equity vacío, no se genera {filename}")
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(equity.index, equity.values)
    ax.set_title(title)
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Equity (USD)")
    ax.grid(alpha=0.3)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"${x:,.0f}")
    )
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Guardado: {filename}")


# ──────────────────────────────────────────────────────────────────────────────
# Wrappers (called from main.py)
# ──────────────────────────────────────────────────────────────────────────────

def plot_trading_results(
    equity,
    trades=None,
    title="Trading Results",
    filename="output_trading_results.png",
):
    try:
        plot_equity_curve(equity, title=title, filename=filename)
    except Exception as e:
        print(f"[plot_trading_results] No se pudo graficar: {e}")


def plot_walk_forward_equity(
    wf_equity,
    title="Walk-Forward Equity (Train OOS)",
    filename="output_WalkForward.png",
):
    try:
        plot_equity_curve(wf_equity, title=title, filename=filename)
    except Exception as e:
        print(f"[plot_walk_forward_equity] No se pudo graficar: {e}")


def plot_walk_forward_test(
    wf_equity_test,
    title="Walk-Forward TEST Equity (OOS)",
    filename="output_WalkForward_Test.png",
):
    try:
        plot_equity_curve(wf_equity_test, title=title, filename=filename)
    except Exception as e:
        print(f"[plot_walk_forward_test] No se pudo graficar: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# Per-window metrics chart
# ──────────────────────────────────────────────────────────────────────────────

def plot_weekly_returns(metrics_list, title=""):
    """
    Bar chart of OOS metrics per walk-forward window.
    Accepts column names in multiple spellings for robustness.
    """
    if not metrics_list:
        print("[plot_weekly_returns] metrics_list vacío")
        return

    df = pd.DataFrame(metrics_list).copy()
    if df.empty:
        print("[plot_weekly_returns] df vacío")
        return

    col_calmar = _pick_first_existing_col(df, ["calmar", "Calmar", "CalmarSimple"])
    col_ret    = _pick_first_existing_col(df, ["ret", "return_pct", "Return"])
    col_dd     = _pick_first_existing_col(df, ["max_dd", "MaxDD"])

    if col_calmar is None or col_ret is None or col_dd is None:
        print("[plot_weekly_returns] Faltan columnas de métricas")
        return

    if "window" not in df.columns:
        df["window"] = range(1, len(df) + 1)

    x       = df["window"].values
    calmars = df[col_calmar].astype(float).values
    rets    = df[col_ret].astype(float).values
    dds     = df[col_dd].astype(float).values

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(x, calmars, marker="o", label="Calmar")
    ax.plot(x, rets,    marker="s", label="Retorno (%)")
    ax.plot(x, dds,     marker="^", label="MaxDD (%)")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title(f"Métricas OOS por ventana — {title}".strip())
    ax.set_xlabel("Ventana")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    out = f"output_WeeklyReturns_{title}.png".replace(" ", "_")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Guardado: {out}")


# ──────────────────────────────────────────────────────────────────────────────
# Parameter stability
# ──────────────────────────────────────────────────────────────────────────────

def plot_parameter_stability(
    stability_df,
    title: str = "Parameter Stability",
    filename: str = "output_parameter_stability.png",
):
    try:
        df = stability_df.copy()

        if set(["window", "param", "value"]).issubset(df.columns):
            plt.figure(figsize=(12, 5))
            for p, g in df.groupby("param"):
                plt.plot(g["window"].values, g["value"].values, label=str(p))
            plt.title(title)
            plt.xlabel("Window")
            plt.ylabel("Value")
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(filename, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"    Guardado: {filename}")
            return

        x       = df["window"].values if "window" in df.columns else df.index.values
        df_plot = df.drop(columns=["window"], errors="ignore")
        df_plot = df_plot.select_dtypes(include=["number"])

        if df_plot.shape[1] == 0:
            out_csv = filename.replace(".png", ".csv")
            df.to_csv(out_csv, index=False)
            print(f"[plot_parameter_stability] Guardado CSV: {out_csv}")
            return

        plt.figure(figsize=(12, 5))
        for col in df_plot.columns:
            plt.plot(x, df_plot[col].values, label=str(col))
        plt.title(title)
        plt.xlabel("Window")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"    Guardado: {filename}")

    except Exception as e:
        print(f"[plot_parameter_stability] Falló gráfico: {e}")
        try:
            out_csv = filename.replace(".png", ".csv")
            stability_df.to_csv(out_csv, index=False)
            print(f"    Guardado CSV: {out_csv}")
        except Exception:
            pass