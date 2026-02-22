"""
optimization.py — Walk-forward optimisation using Optuna.

Bug fix: objective() and all metric lookups now use 'Calmar' (the key
that metrics.get_metrics() returns) instead of 'CalmarSimple'.
"""

import optuna
import pandas as pd
import numpy as np

from backtest import run_single_backtest
from indicators import add_indicators
from metrics import get_metrics

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ──────────────────────────────────────────────────────────────────────────────
# Objective
# ──────────────────────────────────────────────────────────────────────────────

def objective(trial: optuna.Trial, train_chunk: pd.DataFrame) -> float:
    p = {
        "ema_slow": trial.suggest_int("ema_slow", 150, 400),  # Tendencia más lenta = más estable
        "rsi_window": trial.suggest_int("rsi_window", 14, 30),
        "macd_fast": trial.suggest_int("macd_fast", 12, 26),
        "macd_slow": trial.suggest_int("macd_slow", 26, 60),
        "macd_sign": trial.suggest_int("macd_sign", 9, 20),

        # CAMBIO PARA WIN RATE: Stop Loss más grande (más aire)
        "stop_loss": trial.suggest_float("stop_loss", 0.025, 0.04),  # 2.5% a 4%
        "take_profit": trial.suggest_float("take_profit", 0.05, 0.12),  # 5% a 12%
        "risk_per_trade": trial.suggest_float("risk_per_trade", 0.01, 0.02),
    }

    # Restricción 2:1 (Ya no 4:1). En 5min es más realista buscar ganar el doble de lo que arriesgas.
    if p["take_profit"] < p["stop_loss"] * 2.0:
        return -1e6

    if p["macd_slow"] <= p["macd_fast"]:
        return -1e6


    split = int(len(train_chunk) * 0.80)
    inner_val = train_chunk.iloc[split:].copy()

    try:
        val_prep = add_indicators(inner_val, p)
        equity_val, trades = run_single_backtest(val_prep, p)

        # CAMBIO: Con 2 trades muy buenos es suficiente para validar
        if len(trades) < 2:
            return -1e6

        stats = get_metrics(equity_val)
        calmar = float(stats["Calmar"])
        maxdd = float(stats["MaxDD"])
        ret = float(stats["Return"])

        # Control de riesgo sistémico
        if maxdd > 25.0:
            return -1e6

        score = calmar
        if ret < 0:
            score *= 0.1  # Penalización más fuerte si el retorno es negativo

        # Penalización por exceso de trading (evitar quemar cuenta en comisiones)
        score -= 0.001 * len(trades)

        if np.isnan(score) or np.isinf(score):
            return -1e6
        return float(np.clip(score, -500.0, 500.0))

    except Exception:
        return -1e6


# ──────────────────────────────────────────────────────────────────────────────
# Walk-forward — TRAIN set
# ──────────────────────────────────────────────────────────────────────────────

def run_walk_forward(data: pd.DataFrame, n_trials: int = 150):
    """
    Rolling walk-forward on train data.
    Window: 1 month train (8 640 bars) → 1 week OOS (2 016 bars), step weekly.
    """
    TRAIN_SIZE = 8_640   # ~1 month of 5-min bars
    TEST_SIZE  = 2_016   # ~1 week
    STEP       = 2_016

    all_params       = []
    wf_equity_chunks = []
    window_metrics   = []
    all_trades       = []
    window_num       = 0

    total_windows = (len(data) - TRAIN_SIZE - TEST_SIZE) // STEP + 1
    print(f"    Total ventanas estimadas: {total_windows}")

    for i in range(0, len(data) - TRAIN_SIZE - TEST_SIZE + 1, STEP):
        window_num  += 1
        train_chunk  = data.iloc[i : i + TRAIN_SIZE]
        test_chunk   = data.iloc[i + TRAIN_SIZE : i + TRAIN_SIZE + TEST_SIZE]

        print(f"\n  >>> Ventana {window_num}/{total_windows}")
        print(f"      Train: {train_chunk.index[0]} → {train_chunk.index[-1]}")
        print(f"      OOS:   {test_chunk.index[0]}  → {test_chunk.index[-1]}")

        sampler = optuna.samplers.TPESampler()
        pruner  = optuna.pruners.MedianPruner(n_startup_trials=30, n_warmup_steps=10)
        study   = optuna.create_study(
            direction="maximize", sampler=sampler, pruner=pruner
        )
        study.optimize(
            lambda t, tc=train_chunk: objective(t, tc),
            n_trials=n_trials,
            show_progress_bar=False,
        )

        best_p     = study.best_params
        best_score = study.best_value
        print(f"      Mejor Score (Calmar, val. interna): {best_score:.4f}")
        all_params.append(best_p)

        try:
            test_prep          = add_indicators(test_chunk, best_p)
            equity_oos, trades = run_single_backtest(test_prep, best_p)

            wf_equity_chunks.append(equity_oos)
            all_trades.extend(trades)

            oos_stats = get_metrics(equity_oos)

            window_metrics.append({
                "window":  window_num,
                "calmar":  float(oos_stats["Calmar"]),
                "sharpe":  float(oos_stats["Sharpe"]),
                "sortino": float(oos_stats["Sortino"]),
                "max_dd":  float(oos_stats["MaxDD"]),
                "ret":     float(oos_stats["Return"]),
                "n_trades": int(len(trades)),
            })

            print(
                f"      OOS — Calmar: {oos_stats['Calmar']:.4f} | "
                f"Sharpe: {oos_stats['Sharpe']:.4f} | "
                f"MaxDD: {oos_stats['MaxDD']:.2f}% | "
                f"Trades: {len(trades)}"
            )

        except Exception as e:
            print(f"      [!] Error en OOS ventana {window_num}: {e}")

    wf_equity = _concatenate_equity(wf_equity_chunks)

    if window_metrics:
        df_wm = pd.DataFrame(window_metrics)
        print(f"\n  ── Resumen Walk-Forward TRAIN (promedio OOS) ──")
        print(f"     Calmar  medio : {df_wm['calmar'].mean():.4f}")
        print(f"     Sharpe  medio : {df_wm['sharpe'].mean():.4f}")
        print(f"     Sortino medio : {df_wm['sortino'].mean():.4f}")
        print(f"     MaxDD   medio : {df_wm['max_dd'].mean():.2f}%")
        print(f"     Retorno medio : {df_wm['ret'].mean():.2f}%")
        print(f"     Trades  total : {df_wm['n_trades'].sum()}")

    return all_params, wf_equity, window_metrics, all_trades


# ──────────────────────────────────────────────────────────────────────────────
# Walk-forward — TEST set (single window: 3 weeks train → 1 week OOS)
# ──────────────────────────────────────────────────────────────────────────────

def run_walk_forward_test(data: pd.DataFrame, n_trials: int = 100):
    TRAIN_SIZE = 6_048   # 3 weeks
    TEST_SIZE  = 2_016   # 1 week

    if len(data) < TRAIN_SIZE + TEST_SIZE:
        print(
            f"  [!] Test set insuficiente: {len(data)} barras "
            f"< {TRAIN_SIZE + TEST_SIZE}"
        )
        return [], pd.Series(dtype=float), [], []

    train_chunk = data.iloc[:TRAIN_SIZE]
    test_chunk  = data.iloc[TRAIN_SIZE : TRAIN_SIZE + TEST_SIZE]
    leftover    = len(data) - TRAIN_SIZE - TEST_SIZE

    print(f"\n  >>> Ventana TEST (única)")
    print(
        f"      Train: {train_chunk.index[0]} → {train_chunk.index[-1]} "
        f"({len(train_chunk)} barras)"
    )
    print(
        f"      OOS:   {test_chunk.index[0]}  → {test_chunk.index[-1]}  "
        f"({len(test_chunk)} barras)"
    )
    if leftover > 0:
        print(f"      Nota:  {leftover} barras descartadas al final")

    sampler = optuna.samplers.TPESampler(seed=42)
    pruner  = optuna.pruners.MedianPruner(n_startup_trials=20, n_warmup_steps=5)
    study   = optuna.create_study(
        direction="maximize", sampler=sampler, pruner=pruner
    )
    study.optimize(
        lambda t, tc=train_chunk: objective(t, tc),
        n_trials=n_trials,
        show_progress_bar=False,
    )

    best_p     = study.best_params
    best_score = study.best_value
    print(f"      Mejor Score (Calmar, val. interna): {best_score:.4f}")

    try:
        test_prep          = add_indicators(test_chunk, best_p)
        equity_oos, trades = run_single_backtest(test_prep, best_p)
        oos_stats          = get_metrics(equity_oos)

        window_metrics = [{
            "window":   1,
            "calmar":   float(oos_stats["Calmar"]),
            "sharpe":   float(oos_stats["Sharpe"]),
            "sortino":  float(oos_stats["Sortino"]),
            "max_dd":   float(oos_stats["MaxDD"]),
            "ret":      float(oos_stats["Return"]),
            "n_trades": int(len(trades)),
        }]

        print(
            f"      OOS — Calmar: {oos_stats['Calmar']:.4f} | "
            f"Sharpe: {oos_stats['Sharpe']:.4f} | "
            f"MaxDD: {oos_stats['MaxDD']:.2f}% | "
            f"Trades: {len(trades)}"
        )

        return [best_p], equity_oos, window_metrics, trades

    except Exception as e:
        print(f"      [!] Error en OOS test: {e}")
        return [best_p], pd.Series(dtype=float), [], []


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _concatenate_equity(chunks: list[pd.Series]) -> pd.Series:
    """Chain equity chunks by re-basing each to the last value of the previous."""
    if not chunks:
        return pd.Series(dtype=float)

    base       = 1_000_000.0
    normalized = []

    for chunk in chunks:
        if chunk is None or len(chunk) == 0:
            continue
        scale  = base / float(chunk.iloc[0])
        scaled = chunk * scale
        base   = float(scaled.iloc[-1])
        normalized.append(scaled)

    return pd.concat(normalized) if normalized else pd.Series(dtype=float)