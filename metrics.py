"""
metrics.py
==========
Cálculo de métricas de rendimiento estándar para estrategias de trading.

Métricas implementadas:
    Return   → Retorno total del periodo (%)
    MaxDD    → Máximo Drawdown (%)
    Calmar   → Calmar Ratio = CAGR / Max Drawdown
    Sharpe   → Sharpe Ratio anualizado (rf = 0)
    Sortino  → Sortino Ratio anualizado (solo downside volatility)

Factor de anualización:
    BTC 5min → 12 velas/hora × 24h × 365d = 105,120 barras/año
    ann_factor para Sharpe/Sortino = √105,120 ≈ 324.22

NOTAS:
    - Sharpe usa rf = 0 (estándar para activos crypto de alta volatilidad)
    - Calmar usa CAGR geométrico, no retorno simple escalado
    - MaxDD < 0.5% → Calmar = 0 para evitar valores artificialmente altos
      cuando la estrategia no opera (casi sin drawdown = casi sin trades)
"""

import numpy as np
import pandas as pd

PERIODS_PER_YEAR = 105_120   # 12 × 24 × 365


def get_metrics(equity_series: pd.Series) -> dict:
    """
    Calcula métricas de rendimiento sobre una serie de equity.

    Parameters
    ----------
    equity_series : pd.Series
        Valor del portafolio por barra. Mínimo 2 puntos.

    Returns
    -------
    dict con keys: Return, MaxDD, Calmar, Sharpe, Sortino
    """
    if len(equity_series) < 2:
        return {"Return": 0.0, "MaxDD": 0.0, "Calmar": 0.0,
                "Sharpe": 0.0, "Sortino": 0.0}

    rets = equity_series.pct_change().dropna()

    # Retorno total
    total_ret = (equity_series.iloc[-1] / equity_series.iloc[0]) - 1

    # Máximo Drawdown
    peak   = equity_series.cummax()
    dd     = (equity_series - peak) / peak
    max_dd = abs(dd.min())

    # Calmar Ratio — usa CAGR geométrico
    # CAMBIO: umbral subido de 0.001 a 0.5% para evitar Calmar inflado
    # cuando la estrategia casi no opera (MaxDD cercano a cero)
    n_periods  = len(equity_series)
    years      = n_periods / PERIODS_PER_YEAR
    annual_ret = (1 + total_ret) ** (1 / years) - 1 if years > 0 else 0.0
    calmar     = annual_ret / max_dd if max_dd > 0.005 else 0.0  # 0.5% mínimo

    # Sharpe Ratio anualizado
    ann_factor = np.sqrt(PERIODS_PER_YEAR)
    sharpe = (
        (rets.mean() / rets.std()) * ann_factor
        if rets.std() > 0 else 0.0
    )

    # Sortino Ratio anualizado (solo retornos negativos)
    downside = rets[rets < 0]
    sortino  = (
        (rets.mean() / downside.std()) * ann_factor
        if len(downside) > 1 and downside.std() > 0 else 0.0
    )

    return {
        "Return":  total_ret * 100,
        "MaxDD":   max_dd   * 100,
        "Calmar":  calmar,
        "Sharpe":  sharpe,
        "Sortino": sortino,
    }