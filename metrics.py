import numpy as np
import pandas as pd

PERIODS_PER_YEAR = 105_120  # 12 × 24 × 365 (5-min bars)


def get_metrics(equity_series: pd.Series) -> dict:
    """
    Computes standard performance metrics for a backtest equity curve.
    Returns both 'Calmar' and 'CalmarSimple' so optimization.py and
    reporting code can both find what they expect.
    """
    if len(equity_series) < 2:
        return {
            "Return": 0.0,
            "MaxDD": 0.0,
            "Calmar": 0.0,       # <-- used by optimization.py objective()
            "CalmarSimple": 0.0, # <-- used by sensitivity_analysis & reporting
            "Sharpe": 0.0,
            "Sortino": 0.0,
        }

    rets = equity_series.pct_change().dropna()

    # ── Total return (not annualised; sensible for short windows) ──────────
    total_ret = (equity_series.iloc[-1] / equity_series.iloc[0]) - 1

    # ── Max Drawdown ────────────────────────────────────────────────────────
    peak   = equity_series.cummax()
    dd     = (equity_series - peak) / peak
    max_dd = abs(dd.min())

    # ── Calmar (window-safe, no CAGR explosion on tiny windows) ─────────────
    # Require at least 0.5 % drawdown to avoid division-noise
    calmar = (total_ret / max_dd) if max_dd > 0.005 else 0.0

    # ── Sharpe (annualised) ──────────────────────────────────────────────────
    ann_factor = np.sqrt(PERIODS_PER_YEAR)
    sharpe = (rets.mean() / rets.std() * ann_factor) if rets.std() > 0 else 0.0

    # ── Sortino (annualised) ─────────────────────────────────────────────────
    downside = rets[rets < 0]
    if len(downside) > 1 and downside.std() > 0:
        sortino = rets.mean() / downside.std() * ann_factor
    else:
        sortino = 0.0

    return {
        "Return":       total_ret * 100,
        "MaxDD":        max_dd * 100,
        "Calmar":       float(calmar),       # ← optimization.py uses this
        "CalmarSimple": float(calmar),       # ← sensitivity / reporting uses this
        "Sharpe":       float(sharpe),
        "Sortino":      float(sortino),
    }