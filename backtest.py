import pandas as pd
import ta


def run_single_backtest(data: pd.DataFrame) -> list[float]:
    """ Backtest a simple RSI strategy
    """
    data = data.copy()

    # params & constants
    cash = 1_000_000
    COM = 0.125 / 100
    active_long_positions = []
    active_short_positions = []
    strategy_value = [cash]

    num_operations = 0

    # hyperparams
    rsi_window = 4
    rsi_lower = 41
    # rsi_upper = 70

    n_shares = 2
    take_profit = 0.135
    stop_loss = 0.142

    rsi_ind = ta.momentum.RSIIndicator(data.Close, window=rsi_window)

    # Store indicators in the dataframe
    data["rsi"] = rsi_ind.rsi()

    data = data.dropna()

    for i, row in data.iterrows():

        # Risk Management
        for position in active_long_positions.copy():
            current_val = row.Close * n_shares
            if current_val < position["sl"] or current_val > position["tp"]:
                # Close position
                cash += current_val * (1 - COM)
                active_long_positions.remove(position)

        # Check if rsi < rsi_lower
        if row.rsi < rsi_lower:  # Long signal
            cost = row.Close * n_shares * (1 + COM)
            if cash > cost:
                cash -= cost
                active_long_positions.append({
                    "bought_at": row.Close,
                    "type": "LONG",
                    "sl": row.Close * n_shares * (1 - stop_loss),
                    "tp": row.Close * n_shares * (1 + take_profit),
                    "shares": n_shares
                })
                num_operations += 1

        # Strategy Value
        long_values = len(active_long_positions) * row.Close * n_shares
        current_strategy_value = cash + long_values
        strategy_value.append(current_strategy_value)

    # Returns...
    return strategy_value, num_operations

def objective(data: pd.DataFrame, trial) -> list[float]:
    """ Backtest a simple RSI strategy
    """
    data = data.copy()

    # params & constants
    cash = 1_000_000
    COM = 0.125 / 100
    active_long_positions = []
    active_short_positions = []
    strategy_value = [cash]

    num_operations = 0

    # hyperparams
    rsi_window = trial.suggest_int("rsi_window", 4, 40)
    rsi_lower = trial.suggest_int("rsi_lower", 5, 45)
    # rsi_upper = 70

    n_shares = trial.suggest_int("n_shares", 0.1, 10)
    take_profit = trial.suggest_float("take_profit", 0.02, 0.15)
    stop_loss = trial.suggest_float("stop_loss", 0.02, 0.15)

    rsi_ind = ta.momentum.RSIIndicator(data.Close, window=rsi_window)

    # Store indicators in the dataframe
    data["rsi"] = rsi_ind.rsi()

    data = data.dropna()

    for i, row in data.iterrows():

        # Risk Management
        for position in active_long_positions.copy():
            current_val = row.Close * n_shares
            if current_val < position["sl"] or current_val > position["tp"]:
                # Close position
                cash += current_val * (1 - COM)
                active_long_positions.remove(position)

        # Check if rsi < rsi_lower
        if row.rsi < rsi_lower:  # Long signal
            cost = row.Close * n_shares * (1 + COM)
            if cash > cost:
                cash -= cost
                active_long_positions.append({
                    "bought_at": row.Close,
                    "type": "LONG",
                    "sl": row.Close * n_shares * (1 - stop_loss),
                    "tp": row.Close * n_shares * (1 + take_profit),
                    "shares": n_shares
                })
                num_operations += 1

        # Strategy Value
        long_values = len(active_long_positions) * row.Close * n_shares
        current_strategy_value = cash + long_values
        strategy_value.append(current_strategy_value)

    # Returns...
    ret = strategy_value[-1] / strategy_value[0] - 1
    return ret