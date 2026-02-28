"""
optimization.py — Walk-forward optimisation.
Estilo: Lambda trial: objective(data, trial)
"""

import optuna
import pandas as pd
import numpy as np

from backtest import run_single_backtest
from indicators import add_indicators
from metrics import get_metrics

# Silenciar logs para limpieza
optuna.logging.set_verbosity(optuna.logging.WARNING)


def objective(train_chunk: pd.DataFrame, trial: optuna.Trial) -> float:
    # Definición del espacio de búsqueda
    p = {
        "ema_slow": trial.suggest_int("ema_slow", 150, 400),
        "rsi_window": trial.suggest_int("rsi_window", 14, 30),
        "macd_fast": trial.suggest_int("macd_fast", 12, 26),
        "macd_slow": trial.suggest_int("macd_slow", 26, 60),
        "macd_sign": trial.suggest_int("macd_sign", 9, 20),
        "stop_loss": trial.suggest_float("stop_loss", 0.025, 0.04),
        "take_profit": trial.suggest_float("take_profit", 0.05, 0.12),
        "risk_per_trade": trial.suggest_float("risk_per_trade", 0.01, 0.02),
    }

    # Restricciones lógicas
    if p["take_profit"] < p["stop_loss"] * 2.0 or p["macd_slow"] <= p["macd_fast"]:
        return -1e6

    val_prep = add_indicators(train_chunk.copy(), p)


    try:
        equity_val, trades = run_single_backtest(val_prep, p)

        # --- CAMBIO IMPORTANTE: ---
        # Si la estrategia no opera (menos de 5 trades), penalizamos.
        # Esto obliga a Optuna a buscar parámetros más activos y consume más tiempo.
        if len(trades) < 10:
            return -1e6

        stats = get_metrics(equity_val)
        return float(stats["Calmar"])

    except Exception:
        return -1e6


def run_walk_forward(data: pd.DataFrame, n_trials: int = 150):
    TRAIN_SIZE, TEST_SIZE, STEP = 8_640, 2_016, 2_016
    all_params, wf_equity_chunks, window_metrics, all_trades = [], [], [], []

    for i in range(0, len(data) - TRAIN_SIZE - TEST_SIZE + 1, STEP):
        train_chunk = data.iloc[i: i + TRAIN_SIZE]
        test_chunk = data.iloc[i + TRAIN_SIZE: i + TRAIN_SIZE + TEST_SIZE]

        # --- ESTILO DEL PROFESOR ---
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(train_chunk, trial), n_trials=n_trials)

        best_p = study.best_params
        all_params.append(best_p)

        # Backtest OOS
        test_prep = add_indicators(test_chunk, best_p)
        equity_oos, trades = run_single_backtest(test_prep, best_p)
        wf_equity_chunks.append(equity_oos)
        all_trades.extend(trades)

        oos_stats = get_metrics(equity_oos)
        window_metrics.append({
            "window": len(all_params),
            "calmar": float(oos_stats["Calmar"]),
            "n_trades": len(trades)
        })

    return all_params, _concatenate_equity(wf_equity_chunks), window_metrics, all_trades


def run_walk_forward_test(data: pd.DataFrame, n_trials: int = 150):  # Aumentado a 150
    TRAIN_SIZE, TEST_SIZE = 6_048, 2_016
    train_chunk = data.iloc[:TRAIN_SIZE]
    test_chunk = data.iloc[TRAIN_SIZE: TRAIN_SIZE + TEST_SIZE]

    # --- ESTILO DEL PROFESOR ---
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(train_chunk, trial), n_trials=n_trials)

    best_p = study.best_params
    test_prep = add_indicators(test_chunk, best_p)
    equity_oos, trades = run_single_backtest(test_prep, best_p)

    return [best_p], equity_oos, [], trades


def _concatenate_equity(chunks: list[pd.Series]) -> pd.Series:
    if not chunks: return pd.Series(dtype=float)
    base, normalized = 1_000_000.0, []
    for chunk in chunks:
        scale = base / float(chunk.iloc[0])
        scaled = chunk * scale
        base = float(scaled.iloc[-1])
        normalized.append(scaled)
    return pd.concat(normalized)