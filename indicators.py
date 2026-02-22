import ta
import pandas as pd


def add_indicators(df: pd.DataFrame, p: dict) -> pd.DataFrame:
    df = df.copy()

    # 1. EMA de Tendencia (Para saber si el mercado sube o baja)
    df["ema_slow"] = ta.trend.ema_indicator(df["Close"], window=p["ema_slow"])

    # 2. RSI para medir la fuerza del movimiento
    df["rsi"] = ta.momentum.RSIIndicator(df["Close"], window=p["rsi_window"]).rsi()

    # 3. MACD para confirmar el impulso
    macd = ta.trend.MACD(df["Close"], window_fast=p["macd_fast"],
                         window_slow=p["macd_slow"], window_sign=p["macd_sign"])
    df["macd_diff"] = macd.macd_diff()

    # 4. Canales de Ruptura (En lugar de Bollinger para evitar errores de nombres)
    # Calculamos el máximo y mínimo de las últimas N barras
    df["high_channel"] = df["Close"].rolling(window=p.get("lookback", 20)).max()
    df["low_channel"] = df["Close"].rolling(window=p.get("lookback", 20)).min()

    return df.dropna()


def get_signals(row: pd.Series, p: dict) -> int:
    v_long = 0
    v_short = 0

    # CAMBIO: Usaremos el RSI no para comprar barato, sino para confirmar que el tren ya arrancó
    # Votos LONG
    if row["rsi"] > 55: v_long += 1                 # El precio tiene fuerza alcista real
    if row["macd_diff"] > 0: v_long += 1            # El impulso está creciendo
    if row["Close"] > row["ema_slow"]: v_long += 1  # Estamos por encima de la tendencia macro

    # Votos SHORT
    if row["rsi"] < 45: v_short += 1                # El precio tiene fuerza bajista real
    if row["macd_diff"] < 0: v_short += 1           # El impulso es negativo
    if row["Close"] < row["ema_slow"]: v_short += 1 # Estamos por debajo de la tendencia macro

    # REGLA 3 de 3: Solo entramos si TODO confirma la tendencia.
    # Esto bajará el número de trades pero subirá el Win Rate drásticamente.
    if v_long == 3: return 1
    if v_short == 3: return -1
    return 0