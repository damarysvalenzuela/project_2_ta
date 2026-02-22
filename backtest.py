"""
backtest.py — Long/Short backtesting engine without leverage.

Rules:
  • Commission : 0.125 % per side
  • LONG       : pay full notional + fee; no leverage
  • SHORT      : require cash ≥ notional + fee (conservative); lock
                 50 % as initial margin; 30 % maintenance margin;
                 borrow cost 1 %/year pro-rated per 5-min bar
  • Position sizing by risk: size = (cash × risk_per_trade) / (price × sl_pct)
    capped at 98 % of available cash
"""

import pandas as pd

COM          = 0.00125          # 0.125 %
INIT_MARGIN  = 0.50
MAINT_MARGIN = 0.30
BORROW_RATE  = 0.01 / (365 * 24 * 12)  # 1 % p.a. ÷ 5-min bars per year


def run_single_backtest(
    data: pd.DataFrame,
    params: dict,
) -> tuple[pd.Series, list]:
    """
    Run one complete backtest on a pre-processed & indicator-enriched DataFrame.

    Parameters
    ----------
    data   : DataFrame with Close column and all indicator columns.
    params : dict of strategy parameters (stop_loss, take_profit,
             risk_per_trade, …).

    Returns
    -------
    equity_series : pd.Series of mark-to-market portfolio value.
    trades_history: list of dicts with keys 'pnl' and 'side'.
    """
    from indicators import get_signals

    cash           = 1_000_000.0
    active_pos     = None
    equity_hist    = []
    trades_history = []

    sl_pct = float(params["stop_loss"])
    tp_pct = float(params["take_profit"])
    risk   = float(params.get("risk_per_trade", 0.002))

    for row in data.to_dict("records"):
        price = float(row["Close"])

        # ──────────────────────────────────────────────────────────────
        # 1. Manage open position
        # ──────────────────────────────────────────────────────────────
        if active_pos is not None:
            side = active_pos["side"]

            if side == "LONG":
                pnl     = (price - active_pos["entry"]) * active_pos["shares"]
                is_exit = (price <= active_pos["sl"]) or (price >= active_pos["tp"])

            else:  # SHORT
                # Accrue borrow cost in USDT
                borrow_cost = active_pos["shares"] * BORROW_RATE * price
                active_pos["accumulated_borrow"] += borrow_cost

                pnl = ((active_pos["entry"] - price) * active_pos["shares"]
                       - active_pos["accumulated_borrow"])

                # Maintenance margin check
                equity_in_pos = active_pos["margin_locked"] + pnl
                notional_now  = price * active_pos["shares"]
                margin_level  = (equity_in_pos / notional_now
                                 if notional_now > 0 else 1.0)

                is_exit = (
                    (price >= active_pos["sl"])
                    or (price <= active_pos["tp"])
                    or (margin_level <= MAINT_MARGIN)
                )

            # Close position
            if is_exit:
                fee_exit = price * active_pos["shares"] * COM

                if side == "LONG":
                    cash   += price * active_pos["shares"] - fee_exit
                    net_pnl = pnl - fee_exit
                else:
                    cash   += active_pos["margin_locked"] + pnl - fee_exit
                    net_pnl = pnl - fee_exit

                trades_history.append({"pnl": net_pnl, "side": side})
                active_pos = None

        # ──────────────────────────────────────────────────────────────
        # 2. Open new position
        # ──────────────────────────────────────────────────────────────
        if active_pos is None:
            sig = get_signals(row, params)

            if sig != 0:
                # Size by risk; cap at 98 % of cash
                n_shares_ideal = (cash * risk) / (price * sl_pct)
                max_shares     = (cash * 0.98) / (price * (1 + COM))
                n_shares       = min(n_shares_ideal, max_shares)

                if n_shares > 0.0001:
                    notional  = price * n_shares
                    fee_entry = notional * COM

                    if sig == 1:  # LONG
                        cost = notional + fee_entry
                        if cash >= cost:
                            cash -= cost
                            active_pos = {
                                "side":  "LONG",
                                "entry": price,
                                "shares": n_shares,
                                "sl": price * (1 - sl_pct),
                                "tp": price * (1 + tp_pct),
                            }

                    else:  # SHORT — require full notional in cash (no leverage)
                        if cash >= notional + fee_entry:
                            margin_req = notional * INIT_MARGIN
                            cash -= margin_req + fee_entry
                            active_pos = {
                                "side":  "SHORT",
                                "entry": price,
                                "shares": n_shares,
                                "margin_locked": margin_req,
                                "accumulated_borrow": 0.0,
                                "sl": price * (1 + sl_pct),
                                "tp": price * (1 - tp_pct),
                            }

        # ──────────────────────────────────────────────────────────────
        # 3. Mark-to-market equity
        # ──────────────────────────────────────────────────────────────
        current_val = cash

        if active_pos is not None:
            if active_pos["side"] == "LONG":
                current_val += price * active_pos["shares"]
            else:
                pnl_mtm = ((active_pos["entry"] - price) * active_pos["shares"]
                           - active_pos.get("accumulated_borrow", 0.0))
                current_val += active_pos["margin_locked"] + pnl_mtm

        equity_hist.append(current_val)

    return pd.Series(equity_hist, index=data.index), trades_history