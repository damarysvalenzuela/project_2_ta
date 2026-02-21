import pandas as pd


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    data_train = pd.read_csv('data/btc_project_train.csv')
    data_test  = pd.read_csv('data/btc_project_test.csv')
    return data_train, data_test

def preprocess_data(raw_data: pd.DataFrame) -> pd.DataFrame:
    data = pd.DataFrame()

    data.index = pd.to_datetime(
        raw_data['Datetime'].values,
        format='mixed',
        dayfirst=True
    )
    data.index.name = 'Datetime'

    data['timestamp'] = raw_data['Timestamp'].values
    data['Open']      = raw_data['Open'].values
    data['High']      = raw_data['High'].values
    data['Low']       = raw_data['Low'].values
    data['Close']     = raw_data['Close'].values

    data = data.sort_index()
    data = data[~data.index.duplicated(keep='first')]
    data = data.dropna(subset=['Open', 'High', 'Low', 'Close'])

    return data