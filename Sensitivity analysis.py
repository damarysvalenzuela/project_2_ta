"""
plotting.py
===========
Funciones de visualización para el proyecto de trading algorítmico.

Gráficas:
    plot_trading_results()      -> Equity curve + Drawdown
    plot_walk_forward_equity()  -> Equity WF out-of-sample (train)
    plot_walk_forward_test()    -> Equity WF out-of-sample (test)
    plot_weekly_returns()       -> Retorno por semana OOS (barras)
    plot_parameter_stability()  -> Evolucion de parametros entre ventanas WF

Salida: PNG a 150 DPI.
Backend: Agg (sin GUI, compatible con servidores y notebooks).
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Equity curve + Drawdown
# ---------------------------------------------------------------------------

def plot_trading_results(equity, trades, initial_cash, title="Backtest"):
    net_profit   = equity.iloc[-1] - initial_cash
    pct          = (net_profit / initial_cash) * 100
    color_equity = "#2b6cb0" if pct >= 0 else "#c53030"

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 8), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]}
    )

    ax1.plot(equity.index, equity.values,
             color=color_equity, linewidth=1.0, label="Portfolio Value")
    ax1.axhline(y=initial_cash, color="black",
                linestyle="--", linewidth=0.8, alpha=0.5, label="Capital inicial")
    ax1.set_title(
        f"[{title}]  PnL: ${net_profit:,.2f} ({pct:.2f}%)  |  Trades: {len(trades)}",
        fontsize=12
    )
    ax1.set_ylabel("Valor ($)")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=9)
    ax1.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _: f"${x:,.0f}")
    )

    dd = (equity / equity.cummax()) - 1
    ax2.fill_between(dd.index, dd.values * 100, 0,
                     color="#ef4444", alpha=0.4, label="Drawdown")
    ax2.plot(dd.index, dd.values * 100, color="#ef4444", linewidth=0.5)
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("Fecha")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=9)

    plt.tight_layout()
    fname = f"output_{title.replace(' ', '_')}.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"    Guardado: {fname}")


# ---------------------------------------------------------------------------
# Walk-Forward equity — TRAIN
# ---------------------------------------------------------------------------

def plot_walk_forward_equity(wf_equity):
    if wf_equity.empty:
        print("[!] Walk-Forward equity vacia.")
        return
    _plot_wf_core(
        wf_equity,
        "output_WalkForward.png",
        "Walk-Forward Equity TRAIN (Out-of-Sample real)"
    )


# ---------------------------------------------------------------------------
# Walk-Forward equity — TEST
# ---------------------------------------------------------------------------

def plot_walk_forward_test(wf_equity):
    if wf_equity.empty:
        print("[!] Walk-Forward test equity vacia.")
        return
    _plot_wf_core(
        wf_equity,
        "output_WalkForward_Test.png",
        "Walk-Forward Equity TEST (Out-of-Sample - 1 semana)"
    )


# ---------------------------------------------------------------------------
# Nucleo compartido para graficas WF
# ---------------------------------------------------------------------------

def _plot_wf_core(wf_equity, filename, title_prefix):
    initial = wf_equity.iloc[0]
    final   = wf_equity.iloc[-1]
    ret_pct = (final / initial - 1) * 100
    color   = "#22c55e" if ret_pct >= 0 else "#ef4444"

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 8), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]}
    )

    ax1.plot(wf_equity.index, wf_equity.values,
             color=color, linewidth=1.0, label="WF OOS Equity")
    ax1.axhline(y=initial, color="gray",
                linestyle="--", linewidth=0.8, alpha=0.6, label="Capital base")
    ax1.set_title(f"{title_prefix}  |  Retorno: {ret_pct:.2f}%", fontsize=12)
    ax1.set_ylabel("Valor del Portafolio ($)")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=9)
    ax1.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _: f"${x:,.0f}")
    )

    dd = (wf_equity / wf_equity.cummax()) - 1
    ax2.fill_between(dd.index, dd.values * 100, 0,
                     color="#ef4444", alpha=0.4)
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("Fecha")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"    Guardado: {filename}")


# ---------------------------------------------------------------------------
# Retorno semanal OOS por ventana
# ---------------------------------------------------------------------------

def plot_weekly_returns(window_metrics, title="Train"):
    """
    Grafica de barras del retorno (%) de cada semana OOS del Walk-Forward.

    Panel superior: retorno % por semana
        Verde  -> semana ganadora
        Rojo   -> semana perdedora
        Texto  -> Calmar Ratio de esa semana encima de cada barra
        Linea naranja punteada -> retorno promedio

    Panel inferior: numero de trades ejecutados esa semana
    """
    if not window_metrics:
        print(f"[!] Sin metricas de ventanas para graficar ({title}).")
        return

    df       = pd.DataFrame(window_metrics)
    windows  = df["window"].values
    rets     = df["ret"].values
    calmars  = df["calmar"].values
    n_trades = df["n_trades"].values
    avg_ret  = rets.mean()

    colors = ["#22c55e" if r >= 0 else "#ef4444" for r in rets]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(max(14, len(windows) * 0.5), 10),
        gridspec_kw={"height_ratios": [3, 1]}
    )

    # Panel superior: retorno por semana
    bars = ax1.bar(
        windows, rets,
        color=colors, alpha=0.85,
        edgecolor="white", linewidth=0.5
    )

    ax1.axhline(avg_ret, color="#f97316", linestyle="--",
                linewidth=1.5, label=f"Retorno promedio: {avg_ret:.2f}%")
    ax1.axhline(0, color="black", linewidth=0.8, alpha=0.5)

    # Etiqueta Calmar encima/debajo de cada barra
    for bar, calmar in zip(bars, calmars):
        height = bar.get_height()
        y_pos  = height + 0.05 if height >= 0 else height - 0.3
        va     = "bottom" if height >= 0 else "top"
        if not np.isnan(calmar):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                y_pos,
                f"C:{calmar:.1f}",
                ha="center", va=va,
                fontsize=6.5, color="#1e3a5f"
            )

    semanas_ganadoras = sum(r >= 0 for r in rets)
    win_rate_semanal  = semanas_ganadoras / len(rets) * 100

    ax1.set_title(
        f"Retorno OOS por Semana - Walk-Forward {title}\n"
        f"Semanas ganadoras: {semanas_ganadoras}/{len(rets)}  |  "
        f"Win Rate semanal: {win_rate_semanal:.1f}%  |  "
        f"Retorno promedio: {avg_ret:.2f}%",
        fontsize=11
    )
    ax1.set_ylabel("Retorno OOS (%)")
    ax1.set_xlabel("Numero de Ventana (semana)")
    ax1.legend(fontsize=9)
    ax1.grid(axis="y", alpha=0.3)
    ax1.set_xticks(windows)

    # Panel inferior: trades por semana
    ax2.bar(windows, n_trades, color="#64748b", alpha=0.7, edgecolor="white")
    ax2.axhline(n_trades.mean(), color="#f97316", linestyle="--",
                linewidth=1.2, label=f"Media: {n_trades.mean():.1f} trades/semana")
    ax2.set_ylabel("Trades / semana")
    ax2.set_xlabel("Numero de Ventana")
    ax2.legend(fontsize=8)
    ax2.grid(axis="y", alpha=0.3)
    ax2.set_xticks(windows)

    plt.tight_layout()
    fname = f"output_WeeklyReturns_{title}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Guardado: {fname}")


# ---------------------------------------------------------------------------
# Estabilidad de parametros entre ventanas WF
# ---------------------------------------------------------------------------

def plot_parameter_stability(all_params, suffix="Train"):
    """
    Evolucion de los parametros optimos entre ventanas WF.

    Parametros estables entre ventanas -> estrategia robusta.
    Parametros muy variables           -> posible overfit por regimen.
    La linea roja punteada = promedio historico del parametro.
    """
    if not all_params:
        print(f"[!] Lista de parametros vacia ({suffix}).")
        return

    df_params    = pd.DataFrame(all_params)
    numeric_cols = df_params.select_dtypes(include=[float, int]).columns.tolist()

    if not numeric_cols:
        return

    n         = len(numeric_cols)
    cols_plot = 3
    rows_plot = (n + cols_plot - 1) // cols_plot

    fig, axes = plt.subplots(rows_plot, cols_plot,
                             figsize=(14, rows_plot * 3))
    axes_flat = np.array(axes).flatten() if n > 1 else [axes]

    for i, col in enumerate(numeric_cols):
        ax   = axes_flat[i]
        vals = df_params[col].values

        ax.plot(range(1, len(vals) + 1), vals,
                marker="o", markersize=4, linewidth=1.2, color="#3b82f6")
        ax.axhline(np.mean(vals), color="red", linestyle="--",
                   linewidth=0.8, alpha=0.7,
                   label=f"Media: {np.mean(vals):.3g}")
        ax.set_title(col, fontsize=9, fontweight="bold")
        ax.set_xlabel("Ventana WF", fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        f"Estabilidad de Parametros - Walk-Forward {suffix}",
        fontsize=13, y=1.01
    )
    plt.tight_layout()
    fname = f"output_ParameterStability_{suffix}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Guardado: {fname}")