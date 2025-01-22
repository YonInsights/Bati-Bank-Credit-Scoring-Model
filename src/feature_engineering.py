import pandas as pd

def create_aggregate_features(df):
    """Create aggregate features like total transaction amount."""
    df["TotalTransactionAmount"] = df.groupby("CustomerId")["Amount"].transform("sum")
    return df

def extract_datetime_features(df, datetime_column):
    """Extract features like year, month, day from a datetime column."""
    df[datetime_column] = pd.to_datetime(df[datetime_column])
    df["TransactionYear"] = df[datetime_column].dt.year
    df["TransactionMonth"] = df[datetime_column].dt.month
    df["TransactionDay"] = df[datetime_column].dt.day
    return df

if __name__ == "__main__":
    data_path = "D:/Kifya_training/Week 6/Bati-Bank-Credit-Scoring-Model/data/raw/data.csv"
    data = pd.read_csv(data_path)
    engineered_data = create_aggregate_features(data)
    print("Feature engineering complete.")
