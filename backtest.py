"""
backtest.py
===========
Motor de backtesting realista para estrategia Long/Short sin apalancamiento.

REGLAS DE MARGEN (shorts):
    INIT_MARGIN  = 50%  → Broker bloquea 50% del nocional como garantía.
                          Ej: short $10,000 en BTC → $5,000 bloqueados.
    MAINT_MARGIN = 30%  → Margin call si equity_posición / valor_mercado < 30%.
    BORROW_RATE  = 1%/año → Interés sobre el nocional prestado, por barra 5min.
                          = 0.01 / 105,120 barras/año

COMISIÓN:
    COM = 0.125% aplicado al nocional en apertura y cierre de cada posición.

"""

import pandas as pd
from indicators import get_signals

# Constantes globales
COM          = 0.125 / 100
INIT_MARGIN  = 0.50
MAINT_MARGIN = 0.30
BORROW_RATE  = 0.01 / (365 * 24 * 12)   # 1% anual → por barra de 5min


def run_single_backtest(
    data: pd.DataFrame,
    params: dict
) -> tuple[pd.Series, list]:

    cash           = 1_000_000.0
    active_pos     = None
    equity_hist    = []
    trades_history = []

    data_records = data.to_dict("records")

    for row in data_records:
        price = row["Close"]

        # ── 1. GESTIÓN DE POSICIÓN ACTIVA ─────────────────────────── #
        if active_pos is not None:

            if active_pos["side"] == "LONG":
                pnl      = (price - active_pos["entry"]) * active_pos["shares"]
                is_exit  = (price <= active_pos["sl"]) or (price >= active_pos["tp"])

            else:  # SHORT
                # Cobrar borrow cada barra sobre el nocional al precio de entrada
                borrow_cost = (active_pos["shares"] * active_pos["entry"]) * BORROW_RATE
                active_pos["accumulated_borrow"] += borrow_cost

                pnl = (
                    (active_pos["entry"] - price) * active_pos["shares"]
                    - active_pos["accumulated_borrow"]
                )

                # Margin call: equity_posición / valor_mercado < 30%
                equity_in_pos = active_pos["margin_locked"] + pnl
                valor_mercado = price * active_pos["shares"]
                margin_level  = (
                    equity_in_pos / valor_mercado if valor_mercado > 0 else 1.0
                )

                is_exit = (
                    (price >= active_pos["sl"])
                    or (price <= active_pos["tp"])
                    or (margin_level <= MAINT_MARGIN)
                )

            # ── Cerrar posición ────────────────────────────────────── #
            if is_exit:
                fee = price * active_pos["shares"] * COM

                if active_pos["side"] == "LONG":
                    cash   += price * active_pos["shares"] - fee
                    net_pnl = pnl - fee
                else:
                    # Devolver margen bloqueado + PnL - comisión de cierre
                    cash   += active_pos["margin_locked"] + pnl - fee
                    net_pnl = pnl - fee

                trades_history.append({
                    "pnl":  net_pnl,
                    "side": active_pos["side"]
                })
                active_pos = None

        # ── 2. APERTURA DE NUEVA POSICIÓN ─────────────────────────── #
        if active_pos is None:
            sig = get_signals(row, params)

            if sig != 0:
                # Sizing: 1% de riesgo sobre el capital actual
                n_shares_ideal = (cash * 0.01) / (price * params["stop_loss"])
                # Techo: nocional ≤ 98% del cash (sin apalancamiento)
                max_shares     = (cash * 0.98) / (price * (1 + COM))
                n_shares       = min(n_shares_ideal, max_shares)

                if n_shares > 0.0001:
                    notional = price * n_shares
                    fee      = notional * COM

                    if sig == 1:  # LONG
                        cost = notional + fee
                        if cash >= cost:
                            cash -= cost
                            active_pos = {
                                "side":   "LONG",
                                "entry":  price,
                                "shares": n_shares,
                                "sl":     price * (1 - params["stop_loss"]),
                                "tp":     price * (1 + params["take_profit"]),
                            }

                    else:  # SHORT
                        margin_req = notional * INIT_MARGIN
                        cost_short = margin_req + fee
                        # CRÍTICO: verificar cash antes de abrir short
                        if cash >= cost_short:
                            cash -= cost_short
                            active_pos = {
                                "side":               "SHORT",
                                "entry":              price,
                                "shares":             n_shares,
                                "margin_locked":      margin_req,
                                "accumulated_borrow": 0.0,
                                "sl":  price * (1 + params["stop_loss"]),
                                "tp":  price * (1 - params["take_profit"]),
                            }

        # ── 3. MARK-TO-MARKET ─────────────────────────────────────── #
        current_val = cash

        if active_pos is not None:
            if active_pos["side"] == "LONG":
                current_val += price * active_pos["shares"]
            else:
                pnl_mtm = (
                    (active_pos["entry"] - price) * active_pos["shares"]
                    - active_pos.get("accumulated_borrow", 0.0)
                )
                current_val += active_pos["margin_locked"] + pnl_mtm

        equity_hist.append(current_val)

    return pd.Series(equity_hist, index=data.index), trades_history