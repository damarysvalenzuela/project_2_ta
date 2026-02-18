import pandas as pd

def load_data():
    data_train = pd.read_csv('data/btc_project_train.csv').dropna()
    data_test = pd.read_csv('data/btc_project_test.csv').dropna()
    return data_train, data_test

def preprocess_data(raw_data: pd.DataFrame) -> pd.DataFrame:
    data = pd.DataFrame()
    data['timestamp'] = raw_data.Timestamp
    data['Datetime'] = raw_data.Datetime
    data['Open'] = raw_data.Open
    data['High'] = raw_data.High
    data['Low'] = raw_data.Low
    data['Close'] = raw_data.Close
    return data