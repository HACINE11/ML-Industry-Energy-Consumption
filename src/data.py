import pandas as pd
from sklearn.preprocessing import StandardScaler


DATA_PATH = "Steel_industry_data.csv"
TARGET = "Usage_kWh"
DATE_COL = "date"
CATEGORICAL_COLS = ['WeekStatus', 'Day_of_week', 'Load_Type']

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    dt = pd.to_datetime(df[DATE_COL], format="%d/%m/%Y %H:%M", errors="coerce")
    df["hour"] = dt.dt.hour
    df["minute"] = dt.dt.minute
    df["day"] = dt.dt.day
    df["month"] = dt.dt.month
    df["dayofweek_num"] = dt.dt.dayofweek
    return df.drop(columns=[DATE_COL])

def load_split(test_ratio: float = 0.2, random_state: int = 42, use_time_features: bool = True):
    df = pd.read_csv(DATA_PATH)
    #df[DATE_COL] = pd.to_datetime(df[DATE_COL], format="%d/%m/%Y %H:%M", errors="coerce")
    #df = df.sort_values(DATE_COL).reset_index(drop=True)

    #cut = int(len(df) * (1 - test_ratio))
    #train_df = add_time_features(df.iloc[:cut])
    #test_df  = add_time_features(df.iloc[cut:])
    if use_time_features:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], format="%d/%m/%Y %H:%M", errors="coerce")
        df = df.sort_values(DATE_COL).reset_index(drop=True)
        
        cut = int(len(df) * (1 - test_ratio))
        train_df = add_time_features(df.iloc[:cut])
        test_df  = add_time_features(df.iloc[cut:])
        
        # One-hot encode categorical columns (like in your code)
        train_df = pd.get_dummies(train_df, columns=CATEGORICAL_COLS, drop_first=True)
        test_df = pd.get_dummies(test_df, columns=CATEGORICAL_COLS, drop_first=True)
        
        # Ensure both train and test have same columns
        train_cols = train_df.columns
        test_cols = test_df.columns
        all_cols = train_cols.union(test_cols)
        
        train_df = train_df.reindex(columns=all_cols, fill_value=0)
        test_df = test_df.reindex(columns=all_cols, fill_value=0)
        
    else:
        # Alternative: drop date column and encode categoricals (your original approach)
        df = df.drop(columns=[DATE_COL])
        df = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=True)
        
        cut = int(len(df) * (1 - test_ratio))
        train_df = df.iloc[:cut]
        test_df  = df.iloc[cut:]

    y_train = train_df[TARGET]
    y_test  = test_df[TARGET]
    X_train = train_df.drop(columns=[TARGET])
    X_test  = test_df.drop(columns=[TARGET])
    return X_train, y_train, X_test, y_test
