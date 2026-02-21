"""
main.py
=======
Punto de entrada principal del proyecto de trading algorítmico.

Flujo completo:
    1. Carga y preprocesamiento de datos (train + test)
    2. Walk-Forward sobre TRAIN (76 ventanas: 1 mes train → 1 semana OOS)
    3. Walk-Forward sobre TEST  (1 ventana adaptada: 3 sem train → 1 sem OOS)
    4. Reportes de métricas y gráficas para ambos WF
    5. Análisis de sensibilidad ±20%
    6. Análisis de impacto de comisiones

DECISIÓN SOBRE EL TEST SET:
    El test set tiene 9,098 barras (~4.3 semanas). No alcanza para el WF
    estándar (10,656 barras mínimo). Se aplica 1 ventana WF adaptada:
        Train = 3 semanas (6,048 barras) → optimización con datos del test
        OOS   = 1 semana  (2,016 barras) → evaluación out-of-sample exacta
    Los parámetros son 100% independientes del train set.

CAMBIOS vs versión anterior:
    - run_walk_forward()      ahora retorna 4 valores (+ all_trades)
    - run_walk_forward_test() ahora retorna 4 valores (+ trades_oos)
    - generar_reporte() recibe trades reales en vez de lista vacía []
"""

import pandas as pd
import numpy as np
import time
import json

from data                 import load_data, preprocess_data
from optimization         import run_walk_forward, run_walk_forward_test
from indicators           import add_indicators
from metrics              import get_metrics
from plotting             import (plot_trading_results,
                                  plot_walk_forward_equity,
                                  plot_walk_forward_test,
                                  plot_parameter_stability,
                                  plot_weekly_returns)
from sensitivity_analysis import run_sensitivity_analysis, plot_sensitivity


# ─────────────────────────────────────────────────────── #
#  CONFIGURACIÓN                                          #
# ─────────────────────────────────────────────────────── #

N_TRIALS          = 150   # Trials Optuna por ventana (100-200 recomendado)
                           # Usar N_TRIALS = 5 para prueba rápida
DELTA_SENSITIVITY = 0.20  # Variación ±20% para análisis de sensibilidad


# ─────────────────────────────────────────────────────── #
#  REPORTE DE MÉTRICAS                                    #
# ─────────────────────────────────────────────────────── #

def generar_reporte(equity: pd.Series, trades: list, nombre_set: str):
    """
    Imprime reporte completo de métricas.

    Incluye: capital inicial/final, PnL, Win Rate, conteo Long/Short,
    Calmar, Sharpe, Sortino, MaxDD, volatilidad anualizada,
    y tablas de retornos mensual, trimestral y anual.
    """
    stats     = get_metrics(equity)
    df_trades = pd.DataFrame(trades)

    win_rate = 0.0
    n_longs  = 0
    n_shorts = 0

    if not df_trades.empty and "pnl" in df_trades.columns:
        win_rate = (df_trades["pnl"] > 0).mean() * 100
        if "side" in df_trades.columns:
            n_longs  = len(df_trades[df_trades["side"] == "LONG"])
            n_shorts = len(df_trades[df_trades["side"] == "SHORT"])

    vol_anual = equity.pct_change().std() * np.sqrt(105_120) * 100

    print(f"\n{'='*20} REPORTE: {nombre_set} {'='*20}")
    print(f"Capital Inicial : ${equity.iloc[0]:>15,.2f}")
    print(f"Capital Final   : ${equity.iloc[-1]:>15,.2f}")
    print(f"Ganancia Neta   : ${equity.iloc[-1]-equity.iloc[0]:>15,.2f}  ({stats['Return']:.2f}%)")
    print(f"Win Rate        : {win_rate:.2f}%")
    print(f"Total Trades    : {len(trades)}  (Longs: {n_longs} | Shorts: {n_shorts})")
    print("-" * 55)
    print(f"Calmar Ratio    : {stats['Calmar']:>10.4f}")
    print(f"Sharpe Ratio    : {stats['Sharpe']:>10.4f}")
    print(f"Sortino Ratio   : {stats['Sortino']:>10.4f}")
    print(f"Max Drawdown    : {stats['MaxDD']:>10.2f}%")
    print(f"Volatilidad An. : {vol_anual:>10.2f}%")
    print("=" * 55)

    # Tablas de retorno periódico (requiere DatetimeIndex)
    try:
        for freq, label in [("ME", "MENSUAL"), ("QE", "TRIMESTRAL"), ("YE", "ANUAL")]:
            ret_table = equity.resample(freq).last().pct_change(fill_method=None) * 100
            ret_clean = ret_table.dropna()
            if not ret_clean.empty:
                print(f"\nRETORNOS {label} ({nombre_set}):")
                print(ret_clean.to_frame(name="Ret (%)").to_string())
    except Exception as e:
        print(f"\n[!] No se pudieron calcular tablas periódicas: {e}")


# ─────────────────────────────────────────────────────── #
#  MAIN                                                   #
# ─────────────────────────────────────────────────────── #

def main():
    print("=" * 65)
    print("  PROYECTO TRADING: RSI + Bollinger Bands + MACD — BTC 5min")
    print("=" * 65)

    # ─────────────────────────────────────────────────── #
    # 1. CARGA Y PREPROCESAMIENTO                        #
    # ─────────────────────────────────────────────────── #
    print("\n[1/6] Cargando y preprocesando datos...")
    df_train_raw, df_test_raw = load_data()
    df_train = preprocess_data(df_train_raw)
    df_test  = preprocess_data(df_test_raw)

    print(f"  Train: {len(df_train):,} barras  "
          f"({df_train.index[0]} → {df_train.index[-1]})")
    print(f"  Test : {len(df_test):,} barras  "
          f"({df_test.index[0]} → {df_test.index[-1]})")

    # ─────────────────────────────────────────────────── #
    # 2. WALK-FORWARD — TRAIN                            #
    #    76 ventanas: 1 mes train → 1 semana OOS         #
    # ─────────────────────────────────────────────────── #
    print(f"\n[2/6] Walk-Forward Analysis — TRAIN SET "
          f"({N_TRIALS} trials/ventana)...")
    start_train = time.time()

    # CAMBIO: ahora recibe 4 valores (+ all_trades)
    params_train, wf_equity_train, metrics_train, trades_train = run_walk_forward(
        df_train, n_trials=N_TRIALS
    )

    time_train_min = (time.time() - start_train) / 60
    print(f"\n  Completado en {time_train_min:.2f} min. "
          f"Ventanas: {len(params_train)}")

    with open("mejores_parametros_wf_train.json", "w") as f:
        json.dump(params_train, f, indent=4)
    print("  Guardado: mejores_parametros_wf_train.json")

    # ─────────────────────────────────────────────────── #
    # 3. WALK-FORWARD — TEST                             #
    #    1 ventana: 3 sem train → 1 semana OOS exacta   #
    # ─────────────────────────────────────────────────── #
    print(f"\n[3/6] Walk-Forward Analysis — TEST SET "
          f"(ventana adaptada, {N_TRIALS} trials)...")
    start_test = time.time()

    # CAMBIO: ahora recibe 4 valores (+ trades_test)
    params_test, wf_equity_test, metrics_test, trades_test = run_walk_forward_test(
        df_test, n_trials=N_TRIALS
    )

    time_test_min = (time.time() - start_test) / 60
    print(f"\n  Completado en {time_test_min:.2f} min.")

    with open("mejores_parametros_wf_test.json", "w") as f:
        json.dump(params_test, f, indent=4)
    print("  Guardado: mejores_parametros_wf_test.json")

    print(f"\n  Tiempo total de optimización: "
          f"{time_train_min + time_test_min:.2f} min.")

    # ─────────────────────────────────────────────────── #
    # 4. REPORTES Y GRÁFICAS                             #
    # ─────────────────────────────────────────────────── #
    print("\n[4/6] Reportes y gráficas...")

    # Walk-Forward TRAIN
    if not wf_equity_train.empty:
        print("\n--- WALK-FORWARD TRAIN ---")
        # CAMBIO: pasar trades_train reales en vez de []
        generar_reporte(wf_equity_train, trades_train, "WALK-FORWARD TRAIN (OOS)")
        plot_walk_forward_equity(wf_equity_train)
        plot_weekly_returns(metrics_train, title="Train")
        plot_parameter_stability(params_train, suffix="Train")
        plot_trading_results(
            wf_equity_train, trades_train,
            wf_equity_train.iloc[0],
            title="WalkForward_Train"
        )

    # Walk-Forward TEST
    if not wf_equity_test.empty:
        print("\n--- WALK-FORWARD TEST ---")
        # CAMBIO: pasar trades_test reales en vez de []
        generar_reporte(wf_equity_test, trades_test, "WALK-FORWARD TEST (OOS — 1 semana)")
        plot_walk_forward_test(wf_equity_test)
        plot_weekly_returns(metrics_test, title="Test")
        plot_parameter_stability(params_test, suffix="Test")
        plot_trading_results(
            wf_equity_test, trades_test,
            wf_equity_test.iloc[0],
            title="WalkForward_Test"
        )

    # ─────────────────────────────────────────────────── #
    # 5. ANÁLISIS DE SENSIBILIDAD ±20%                   #
    # ─────────────────────────────────────────────────── #
    print(f"\n[5/6] Análisis de sensibilidad (±{int(DELTA_SENSITIVITY*100)}%)...")

    # Usar parámetros del test WF (si existen), si no los del train
    final_params = (params_test[-1]  if params_test  else
                    params_train[-1] if params_train else None)

    if final_params is not None:
        print("\n  Parámetros de referencia:")
        for k, v in final_params.items():
            print(f"    {k:15s}: {v}")

        try:
            # Evaluar sobre el OOS del test (última semana del test set)
            TRAIN_TEST = 3 * 2016
            TEST_TEST  = 2016
            sens_chunk = df_test.iloc[TRAIN_TEST : TRAIN_TEST + TEST_TEST]
            sens_prep  = add_indicators(sens_chunk, final_params)

            df_sensitivity = run_sensitivity_analysis(
                sens_prep, final_params, delta=DELTA_SENSITIVITY
            )
            df_sensitivity.to_csv("output_sensitivity_results.csv", index=False)
            print("\n  Resumen de Sensibilidad:")
            print(df_sensitivity.to_string(index=False))
            plot_sensitivity(df_sensitivity)

        except Exception as e:
            print(f"  [!] Error en sensibilidad: {e}")
    else:
        print("  [!] Sin parámetros disponibles.")

    # ─────────────────────────────────────────────────── #
    # 6. IMPACTO DE COMISIONES                           #
    # ─────────────────────────────────────────────────── #
    print(f"\n[6/6] Análisis de impacto de comisiones...")
    COM = 0.00125

    print("=" * 55)
    print("ANÁLISIS DE IMPACTO DE COMISIONES")
    print("=" * 55)

    if metrics_train:
        total_trades_tr = sum(m["n_trades"] for m in metrics_train)
        if not wf_equity_train.empty and total_trades_tr > 0:
            est_fees = wf_equity_train.iloc[0] * 0.01 * total_trades_tr * 2 * COM
            print(f"  TRAIN WF — Trades: {total_trades_tr:,} | "
                  f"Comisiones est.: ~${est_fees:,.0f}")

    if metrics_test:
        total_trades_te = sum(m["n_trades"] for m in metrics_test)
        if not wf_equity_test.empty and total_trades_te > 0:
            est_fees = wf_equity_test.iloc[0] * 0.01 * total_trades_te * 2 * COM
            print(f"  TEST  WF — Trades: {total_trades_te:,} | "
                  f"Comisiones est.: ~${est_fees:,.0f}")
    print("=" * 55)

    # ─────────────────────────────────────────────────── #
    # RESUMEN ARCHIVOS GENERADOS                         #
    # ─────────────────────────────────────────────────── #
    print("\n✅ Análisis completo. Archivos generados:")
    for f in [
        "mejores_parametros_wf_train.json",
        "mejores_parametros_wf_test.json",
        "output_sensitivity_results.csv",
        "output_WalkForward.png",
        "output_WalkForward_Test.png",
        "output_WalkForward_Train.png",
        "output_WeeklyReturns_Train.png",
        "output_WeeklyReturns_Test.png",
        "output_ParameterStability_Train.png",
        "output_ParameterStability_Test.png",
        "output_Sensitivity_Calmar.png",
        "output_Sensitivity_MaxDD.png",
    ]:
        print(f"   {f}")


if __name__ == "__main__":
    main()