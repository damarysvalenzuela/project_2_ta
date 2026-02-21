"""
optimization.py
===============
Walk-Forward Analysis con optimización Bayesiana (Optuna / TPE).

ARQUITECTURA WALK-FORWARD — TRAIN (76 ventanas):
    TRAIN_SIZE = 8,640 barras  (~1 mes en velas de 5 min)
    TEST_SIZE  = 2,016 barras  (~1 semana en velas de 5 min)
    STEP       = 2,016 barras  (avanza 1 semana por iteración)

ARQUITECTURA WALK-FORWARD — TEST (1 ventana adaptada):
    El test set tiene 9,098 barras (~4.3 semanas). No alcanza para
    el WF estándar (10,656 barras mínimo). Solución:
        TRAIN = 3 semanas (6,048 barras)
        OOS   = 1 semana  (2,016 barras) — exacto como exige el profesor
        Sobran ~1,034 barras (~3.6 días) que se descartan.

FUNCIÓN OBJETIVO:
    Maximiza Calmar Ratio (no retorno simple como en el código base).
    Split interno 80/20 dentro del train de cada ventana.

CAMBIOS vs versión anterior:
    1. Penalización si trades_val < 3 en validación interna
    2. Penalización si MaxDD < 0.1% (estrategia que no opera)
    3. run_walk_forward() ahora retorna también all_trades (lista acumulada)
    4. run_walk_forward_test() ahora retorna también trades_oos

BUG CORREGIDO — Closure de Python:
    La lambda original capturaba train_chunk por REFERENCIA, haciendo que
    todas las ventanas optimizaran sobre el ÚLTIMO train_chunk del loop.
    → Corregido: lambda t, tc=train_chunk: objective(t, tc)
"""

import optuna
import pandas as pd
import numpy as np

from backtest   import run_single_backtest
from indicators import add_indicators
from metrics    import get_metrics

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ---------------------------------------------------------------------------
# Función objetivo
# ---------------------------------------------------------------------------

def objective(trial: optuna.Trial, train_chunk: pd.DataFrame) -> float:
    """
    Evalúa un conjunto de hiperparámetros sobre una ventana de entrenamiento.

    Proceso:
        1. Sugerir parámetros con TPE
        2. Validar restricciones de lógica financiera
        3. Split interno 80/20 del train_chunk
        4. Sanity check en train interno (mínimo 5 trades)
        5. Evaluar en validación interna (20%)
        6. Penalizar si < 3 trades en validación o MaxDD < 0.1%
        7. Retornar Calmar Ratio

    Returns
    -------
    float : Calmar Ratio (−1e6 si parámetros inválidos o excepción)
    """

    p = {
        # RSI
        "rsi_window": trial.suggest_int("rsi_window", 14, 28),
        "rsi_buy": trial.suggest_int("rsi_buy", 15, 30),
        "rsi_sell": trial.suggest_int("rsi_sell", 70, 85),

        # Bollinger Bands
        "bb_window": trial.suggest_int("bb_window", 20, 50),
        "bb_dev": trial.suggest_float("bb_dev", 2.0, 3.0),

        # MACD (ESTO ES LO QUE FALTA Y CAUSA EL ERROR)
        "macd_fast": trial.suggest_int("macd_fast", 8, 20),
        "macd_slow": trial.suggest_int("macd_slow", 21, 50),
        "macd_sign": trial.suggest_int("macd_sign", 7, 15),

        # Gestión de riesgo (Rangos más amplios para sobrevivir a comisiones)
        "stop_loss": trial.suggest_float("stop_loss", 0.015, 0.03),
        "take_profit": trial.suggest_float("take_profit", 0.04, 0.10),
    }

    # Restricciones de lógica financiera
    if p["take_profit"] < p["stop_loss"] * 1.2:   return -1e6
    if p["macd_slow"]   <= p["macd_fast"]:          return -1e6
    if p["rsi_buy"]     >= p["rsi_sell"]:            return -1e6

    # Split interno 80/20
    split       = int(len(train_chunk) * 0.80)
    inner_train = train_chunk.iloc[:split]
    inner_val   = train_chunk.iloc[split:]

    # Sanity check en train interno
    try:
        train_prep = add_indicators(inner_train, p)
        if len(train_prep) < 100:
            return -1e6
        _, trades_tr = run_single_backtest(train_prep, p)
        if len(trades_tr) < 5:
            return -1e6
    except Exception:
        return -1e6

    # Evaluar en validación interna
    try:
        val_prep = add_indicators(inner_val, p)
        if len(val_prep) < 50:
            return -1e6
        equity_val, trades_val = run_single_backtest(val_prep, p)
    except Exception:
        return -1e6

    # CAMBIO 1: penalizar si pocos trades en validación
    if len(trades_val) < 5:
        return -10.0

    stats  = get_metrics(equity_val)
    calmar = stats["Calmar"]

    # CAMBIO 2: penalizar si MaxDD casi cero (estrategia que no opera)
    if stats["MaxDD"] < 0.1:
        return 0.0

    return calmar if not np.isnan(calmar) else -1e6


# ---------------------------------------------------------------------------
# Walk-Forward Analysis — TRAIN
# ---------------------------------------------------------------------------

def run_walk_forward(
    data: pd.DataFrame,
    n_trials: int = 150
) -> tuple[list, pd.Series, list, list]:
    """
    Ejecuta Walk-Forward Analysis sobre el dataset de ENTRENAMIENTO.

    Para cada ventana:
        1. train_chunk = data[i : i + 8640]
        2. Optimizar con Optuna (TPE, seed=42)
        3. Evaluar best_params en test_chunk (OOS semanal)
        4. Guardar equity OOS, parámetros y trades

    Parameters
    ----------
    data     : pd.DataFrame  Train set con DatetimeIndex
    n_trials : int           Trials Optuna por ventana (100-200 recomendado)

    Returns
    -------
    all_params     : list[dict]   Mejores parámetros por ventana
    wf_equity      : pd.Series    Equity OOS concatenada y normalizada
    window_metrics : list[dict]   Métricas OOS por ventana
    all_trades     : list[dict]   Todos los trades OOS acumulados   ← NUEVO
    """
    TRAIN_SIZE = 8_640
    TEST_SIZE  = 2_016
    STEP       = 2_016

    all_params       = []
    wf_equity_chunks = []
    window_metrics   = []
    all_trades       = []   # CAMBIO: acumular trades de todas las ventanas
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

        sampler = optuna.samplers.TPESampler(seed=42)
        pruner  = optuna.pruners.MedianPruner(n_startup_trials=20, n_warmup_steps=5)
        study   = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
        )

        # BUG FIX: capturar train_chunk por valor con argumento por defecto
        study.optimize(
            lambda t, tc=train_chunk: objective(t, tc),
            n_trials=n_trials,
            show_progress_bar=False,
        )

        best_p      = study.best_params
        best_calmar = study.best_value
        print(f"      Mejor Calmar (val. interna): {best_calmar:.4f}")

        try:
            test_prep              = add_indicators(test_chunk, best_p)
            equity_oos, trades_oos = run_single_backtest(test_prep, best_p)
            wf_equity_chunks.append(equity_oos)
            all_trades.extend(trades_oos)   # CAMBIO: acumular trades OOS

            oos_stats = get_metrics(equity_oos)
            window_metrics.append({
                "window":   window_num,
                "calmar":   oos_stats["Calmar"],
                "sharpe":   oos_stats["Sharpe"],
                "max_dd":   oos_stats["MaxDD"],
                "ret":      oos_stats["Return"],
                "n_trades": len(trades_oos),
            })
            print(f"      OOS — Calmar: {oos_stats['Calmar']:.4f} | "
                  f"Sharpe: {oos_stats['Sharpe']:.4f} | "
                  f"MaxDD: {oos_stats['MaxDD']:.2f}% | "
                  f"Trades: {len(trades_oos)}")
        except Exception as e:
            print(f"      [!] Error en OOS ventana {window_num}: {e}")

        all_params.append(best_p)

    # Concatenar equity WF normalizando en cadena
    wf_equity = _concatenate_equity(wf_equity_chunks)

    # Resumen promedio OOS
    if window_metrics:
        df_wm = pd.DataFrame(window_metrics)
        print(f"\n  ── Resumen Walk-Forward TRAIN (promedio OOS) ──")
        print(f"     Calmar  medio : {df_wm['calmar'].mean():.4f}")
        print(f"     Sharpe  medio : {df_wm['sharpe'].mean():.4f}")
        print(f"     MaxDD   medio : {df_wm['max_dd'].mean():.2f}%")
        print(f"     Retorno medio : {df_wm['ret'].mean():.2f}%")
        print(f"     Trades  total : {df_wm['n_trades'].sum()}")

    return all_params, wf_equity, window_metrics, all_trades   # CAMBIO: +all_trades


# ---------------------------------------------------------------------------
# Walk-Forward Analysis — TEST (ventana adaptada)
# ---------------------------------------------------------------------------

def run_walk_forward_test(
    data: pd.DataFrame,
    n_trials: int = 150
) -> tuple[list, pd.Series, list, list]:
    """
    Walk-Forward especial para el TEST SET (1 ventana adaptada).

    El test set tiene 9,098 barras (~4.3 semanas). No alcanza para
    1 mes train + 1 semana test (10,656 barras mínimo).

    Ventana adoptada:
        TRAIN = 3 semanas = 6,048 barras  (2,016 × 3)
        OOS   = 1 semana  = 2,016 barras  (exacto, igual que en train WF)
        Total = 8,064 barras — entra en los 9,098 disponibles.
        Sobran ~1,034 barras (~3.6 días) que se descartan.

    Parameters
    ----------
    data     : pd.DataFrame  Test set completo con DatetimeIndex
    n_trials : int           Trials de Optuna

    Returns
    -------
    all_params     : list[dict]   Parámetros de la ventana única
    wf_equity      : pd.Series    Equity OOS de la semana evaluada
    window_metrics : list[dict]   Métricas OOS
    trades_oos     : list[dict]   Trades ejecutados en OOS           ← NUEVO
    """
    TRAIN_SIZE = 3 * 2016   # 6,048 barras = 3 semanas
    TEST_SIZE  = 2_016       # 2,016 barras = 1 semana exacta

    if len(data) < TRAIN_SIZE + TEST_SIZE:
        print(f"  [!] Test set insuficiente: {len(data)} barras < {TRAIN_SIZE+TEST_SIZE}")
        return [], pd.Series(dtype=float), [], []

    train_chunk = data.iloc[:TRAIN_SIZE]
    test_chunk  = data.iloc[TRAIN_SIZE : TRAIN_SIZE + TEST_SIZE]
    leftover    = len(data) - TRAIN_SIZE - TEST_SIZE

    print(f"\n  >>> Ventana TEST (única — adaptada al tamaño del dataset)")
    print(f"      Train: {train_chunk.index[0]} → {train_chunk.index[-1]} "
          f"({len(train_chunk)} barras = 3 semanas)")
    print(f"      OOS:   {test_chunk.index[0]}  → {test_chunk.index[-1]}  "
          f"({len(test_chunk)} barras = 1 semana exacta)")
    print(f"      Nota:  {leftover} barras finales (~3.6 días) descartadas")

    sampler = optuna.samplers.TPESampler(seed=42)
    pruner  = optuna.pruners.MedianPruner(n_startup_trials=20, n_warmup_steps=5)
    study   = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
    )

    study.optimize(
        lambda t, tc=train_chunk: objective(t, tc),
        n_trials=n_trials,
        show_progress_bar=False,
    )

    best_p      = study.best_params
    best_calmar = study.best_value
    print(f"      Mejor Calmar (val. interna): {best_calmar:.4f}")

    all_params     = [best_p]
    wf_equity_out  = pd.Series(dtype=float)
    window_metrics = []
    trades_oos     = []   # CAMBIO: inicializar lista de trades

    try:
        test_prep              = add_indicators(test_chunk, best_p)
        equity_oos, trades_oos = run_single_backtest(test_prep, best_p)
        wf_equity_out          = equity_oos

        oos_stats = get_metrics(equity_oos)
        window_metrics.append({
            "window":   1,
            "calmar":   oos_stats["Calmar"],
            "sharpe":   oos_stats["Sharpe"],
            "max_dd":   oos_stats["MaxDD"],
            "ret":      oos_stats["Return"],
            "n_trades": len(trades_oos),
        })
        print(f"      OOS — Calmar: {oos_stats['Calmar']:.4f} | "
              f"Sharpe: {oos_stats['Sharpe']:.4f} | "
              f"MaxDD: {oos_stats['MaxDD']:.2f}% | "
              f"Trades: {len(trades_oos)}")
    except Exception as e:
        print(f"      [!] Error en OOS test: {e}")

    return all_params, wf_equity_out, window_metrics, trades_oos  # CAMBIO: +trades_oos


# ---------------------------------------------------------------------------
# Helper interno
# ---------------------------------------------------------------------------

def _concatenate_equity(chunks: list) -> pd.Series:
    """
    Concatena chunks de equity normalizando en cadena.
    Cada chunk escala para empezar donde terminó el anterior.
    Simula un portafolio continuo con rebalanceo entre ventanas.
    """
    if not chunks:
        return pd.Series(dtype=float)

    base       = 1_000_000.0
    normalized = []

    for chunk in chunks:
        if len(chunk) == 0:
            continue
        scale  = base / chunk.iloc[0]
        scaled = chunk * scale
        base   = scaled.iloc[-1]
        normalized.append(scaled)

    return pd.concat(normalized) if normalized else pd.Series(dtype=float)