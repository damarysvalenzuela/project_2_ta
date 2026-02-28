import pandas as pd
import numpy as np
import time
import json

from data import load_data, preprocess_data
from optimization import run_walk_forward, run_walk_forward_test
from indicators import add_indicators
from metrics import get_metrics
from plotting import (
    plot_trading_results,
    plot_walk_forward_equity,
    plot_walk_forward_test,
    plot_parameter_stability,
    plot_weekly_returns
)
from sensitivity_analysis import run_sensitivity_analysis, plot_sensitivity

import sensitivity_analysis as sa
print("\n[DEBUG] sensitivity_analysis importado desde:", sa.__file__)
print("[DEBUG] FLOAT_PARAMS:", getattr(sa, "FLOAT_PARAMS", None))
# ─────────────────────────────────────────────────────── #
#  CONFIGURACIÓN
# ─────────────────────────────────────────────────────── #

N_TRIALS          = 200
DELTA_SENSITIVITY = 0.20


# ─────────────────────────────────────────────────────── #
#  REPORTE DE MÉTRICAS
# ─────────────────────────────────────────────────────── #

def generar_reporte(equity: pd.Series, trades: list, nombre_set: str):
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
    print(f"Calmar (Simple) : {stats['CalmarSimple']:>10.4f}")
    print(f"Sharpe Ratio    : {stats['Sharpe']:>10.4f}")
    print(f"Sortino Ratio   : {stats['Sortino']:>10.4f}")
    print(f"Max Drawdown    : {stats['MaxDD']:>10.2f}%")
    print(f"Volatilidad An. : {vol_anual:>10.2f}%")
    print("=" * 55)

    # Tablas periódicas
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
#  MAIN
# ─────────────────────────────────────────────────────── #

def main():
    print("=" * 65)
    print("  PROYECTO TRADING")
    print("=" * 65)

    # 1) Carga y preprocesamiento
    print("\n[1/6] Cargando y preprocesando datos...")
    df_train_raw, df_test_raw = load_data()
    df_train = preprocess_data(df_train_raw)
    df_test  = preprocess_data(df_test_raw)

    print(f"  Train: {len(df_train):,} barras  ({df_train.index[0]} → {df_train.index[-1]})")
    print(f"  Test : {len(df_test):,} barras  ({df_test.index[0]} → {df_test.index[-1]})")

    # 2) Walk-forward TRAIN
    print(f"\n[2/6] Walk-Forward Analysis — TRAIN SET ({N_TRIALS} trials/ventana)...")
    start_train = time.time()

    params_train, wf_equity_train, metrics_train, trades_train = run_walk_forward(
        df_train, n_trials=N_TRIALS
    )

    time_train_min = (time.time() - start_train) / 60
    print(f"\n  Completado en {time_train_min:.2f} min. Ventanas: {len(params_train)}")

    with open("mejores_parametros_wf_train.json", "w") as f:
        json.dump(params_train, f, indent=4)
    print("  Guardado: mejores_parametros_wf_train.json")

    # 3) Walk-forward TEST (3 semanas train -> 1 semana OOS)
    print(f"\n[3/6] Walk-Forward Analysis — TEST SET (ventana adaptada, {N_TRIALS} trials)...")
    start_test = time.time()

    params_test, wf_equity_test, metrics_test, trades_test = run_walk_forward_test(
        df_test, n_trials=N_TRIALS
    )

    time_test_min = (time.time() - start_test) / 60
    print(f"\n  Completado en {time_test_min:.2f} min.")

    with open("mejores_parametros_wf_test.json", "w") as f:
        json.dump(params_test, f, indent=4)
    print("  Guardado: mejores_parametros_wf_test.json")

    print(f"\n  Tiempo total de optimización: {time_train_min + time_test_min:.2f} min.")

    # 4) Reportes y gráficas
    print("\n[4/6] Reportes y gráficas...")

    if not wf_equity_train.empty:
        print("\n--- WALK-FORWARD TRAIN ---")
        generar_reporte(wf_equity_train, trades_train, "WALK-FORWARD TRAIN (OOS)")
        plot_walk_forward_equity(wf_equity_train)
        plot_weekly_returns(metrics_train, title="Train")

    if not wf_equity_test.empty:
        print("\n--- WALK-FORWARD TEST ---")
        generar_reporte(wf_equity_test, trades_test, "WALK-FORWARD TEST (OOS — 1 semana)")
        plot_walk_forward_test(wf_equity_test)

    # 5) Sensitivity ±20%
    print(f"\n[5/6] Análisis de sensibilidad (±{int(DELTA_SENSITIVITY*100)}%)...")

    final_params = (params_test[-1]  if params_test  else
                    params_train[-1] if params_train else None)

    if final_params is not None:
        print("\n  Parámetros de referencia:")
        for k, v in final_params.items():
            print(f"    {k:15s}: {v}")

        try:
            TRAIN_TEST = 3 * 2016
            TEST_TEST  = 2016
            sens_chunk = df_test.iloc[TRAIN_TEST: TRAIN_TEST + TEST_TEST]  # ✅ raw OHLC
            df_sensitivity = run_sensitivity_analysis(
                sens_chunk, final_params, delta=DELTA_SENSITIVITY
            )
            df_sensitivity.to_csv("output_sensitivity_results.csv", index=False)
            print("\n  Resumen de Sensibilidad:")
            print(df_sensitivity.to_string(index=False))
            plot_sensitivity(df_sensitivity)

        except Exception as e:
            print(f"  [!] Error en sensibilidad: {e}")
    else:
        print("  [!] Sin parámetros disponibles.")

    # 6) Impacto comisiones (estimación simple)
    print(f"\n[6/6] Análisis de impacto de comisiones...")
    COM = 0.00125

    print("=" * 55)
    print("ANÁLISIS DE IMPACTO DE COMISIONES")
    print("=" * 55)

    # NOTA: esto es estimación; si quieres exacto, luego guardamos fee_entry/fee_exit por trade
    if metrics_train:
        total_trades_tr = sum(m["n_trades"] for m in metrics_train)
        if not wf_equity_train.empty and total_trades_tr > 0:
            # usa riesgo promedio aproximado (si existe en params) solo para “orden de magnitud”
            r = float(final_params.get("risk_per_trade", 0.002)) if final_params else 0.002
            est_fees = wf_equity_train.iloc[0] * r * total_trades_tr * 2 * COM
            print(f"  TRAIN WF — Trades: {total_trades_tr:,} | Comisiones est.: ~${est_fees:,.0f}")

    if metrics_test:
        total_trades_te = sum(m["n_trades"] for m in metrics_test)
        if not wf_equity_test.empty and total_trades_te > 0:
            r = float(final_params.get("risk_per_trade", 0.002)) if final_params else 0.002
            est_fees = wf_equity_test.iloc[0] * r * total_trades_te * 2 * COM
            print(f"  TEST  WF — Trades: {total_trades_te:,} | Comisiones est.: ~${est_fees:,.0f}")

    print("=" * 55)

    print("\n✅ Análisis completo. Archivos generados:")
    archivos_finales = [
        "mejores_parametros_wf_train.json",
        "mejores_parametros_wf_test.json",
        "output_sensitivity_results.csv",
        "output_WalkForward.png",
        "output_WalkForward_Test.png",
        "output_WeeklyReturns_Train.png",
        "output_Sensitivity_Calmar.png",
        "output_Sensitivity_MaxDD.png",
    ]
    for f in archivos_finales:
        print(f"   {f}")


if __name__ == "__main__":
    main()