import pandas as pd

DATA_PATH = "Steel_industry_data.csv"
TARGET = "Usage_kWh"
DATE_COL = "date"

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    dt = pd.to_datetime(df[DATE_COL], format="%d/%m/%Y %H:%M", errors="coerce")
    df["hour"] = dt.dt.hour
    df["minute"] = dt.dt.minute
    df["day"] = dt.dt.day
    df["month"] = dt.dt.month
    df["dayofweek_num"] = dt.dt.dayofweek
    return df.drop(columns=[DATE_COL])

def load_split(test_ratio: float = 0.2):
    df = pd.read_csv(DATA_PATH)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], format="%d/%m/%Y %H:%M", errors="coerce")
    df = df.sort_values(DATE_COL).reset_index(drop=True)

    cut = int(len(df) * (1 - test_ratio))
    train_df = add_time_features(df.iloc[:cut])
    test_df  = add_time_features(df.iloc[cut:])

    y_train = train_df[TARGET]
    y_test  = test_df[TARGET]
    X_train = train_df.drop(columns=[TARGET])
    X_test  = test_df.drop(columns=[TARGET])
    return X_train, y_train, X_test, y_test
