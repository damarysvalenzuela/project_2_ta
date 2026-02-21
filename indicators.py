"""
indicators.py
=============
Cálculo de indicadores técnicos y generación de señales 2-de-3.

Estrategia: RSI + Bollinger Bands + MACD
    RSI        → Familia: Momentum / sobrecompra-sobreventa
    BB         → Familia: Volatilidad / reversión a la media
    MACD       → Familia: Tendencia / impulso de precio

Las 3 familias son intencionalmente DISTINTAS para evitar redundancia
de señal (no usar RSI + Stoch que son ambos momentum).

Regla de confirmación 2-de-3:
    LONG  si ≥ 2 votos alcistas
    SHORT si ≥ 2 votos bajistas
    FLAT  si no hay mayoría
"""

import ta
import pandas as pd


def add_indicators(df: pd.DataFrame, p: dict) -> pd.DataFrame:
    """
    Agrega RSI, Bollinger Bands y MACD al DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        OHLC con DatetimeIndex. Debe contener columna 'Close'.
    p  : dict
        Hiperparámetros:
            rsi_window  : int   → período del RSI
            rsi_buy     : int   → umbral de sobreventa  (ej. 30)
            rsi_sell    : int   → umbral de sobrecompra (ej. 70)
            bb_window   : int   → período de las BB
            bb_dev      : float → desviaciones estándar
            macd_fast   : int   → EMA rápida del MACD
            macd_slow   : int   → EMA lenta del MACD
            macd_sign   : int   → EMA de señal MACD

    Returns
    -------
    pd.DataFrame con columnas adicionales:
        rsi, bb_upper, bb_lower, bb_mid, macd_diff

    """
    df = df.copy()

    # 1. RSI — Momentum
    df["rsi"] = ta.momentum.RSIIndicator(
        df["Close"], window=p["rsi_window"]
    ).rsi()

    # 2. Bollinger Bands — Volatilidad
    bb = ta.volatility.BollingerBands(
        df["Close"], window=p["bb_window"], window_dev=p["bb_dev"]
    )
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_mid"]   = bb.bollinger_mavg()

    # 3. MACD — Tendencia
    macd = ta.trend.MACD(
        df["Close"],
        window_fast=p["macd_fast"],
        window_slow=p["macd_slow"],
        window_sign=p["macd_sign"]
    )
    df["macd_diff"] = macd.macd_diff()

    return df.dropna()


def get_signals(row: pd.Series, p: dict) -> int:
    v_long = 0
    v_short = 0

    # Lógica de IMPULSO (Trend Following)
    # 1. RSI: No buscamos sobreventa, buscamos fuerza.
    # Si el RSI > 50, hay impulso alcista.
    if row["rsi"] > 50: v_long += 1
    if row["rsi"] < 50: v_short += 1

    # 2. BB: Ruptura de bandas (Volatility Breakout)
    if row["Close"] > row["bb_upper"]: v_long += 1
    if row["Close"] < row["bb_lower"]: v_short += 1

    # 3. MACD: Confirmación de tendencia
    if row["macd_diff"] > 0: v_long += 1
    if row["macd_diff"] < 0: v_short += 1

    # EXIGENCIA 3 de 3: Solo entramos cuando todo confirma que hay una tendencia fuerte
    if v_long == 3: return 1
    if v_short == 3: return -1
    return 0