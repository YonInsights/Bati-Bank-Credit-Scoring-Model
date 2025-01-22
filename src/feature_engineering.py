import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler

def create_aggregate_features(df):
    """Create aggregate features like total transaction amount."""
    df["TotalTransactionAmount"] = df.groupby("CustomerId")["Amount"].transform("sum")
    df["AverageTransactionAmount"] = df.groupby("CustomerId")["Amount"].transform("mean")
    df["TransactionCount"] = df.groupby("CustomerId")["TransactionId"].transform("count")
    return df

def extract_datetime_features(df, datetime_column):
    """Extract features like year, month, day from a datetime column."""
    df[datetime_column] = pd.to_datetime(df[datetime_column])
    df["TransactionYear"] = df[datetime_column].dt.year
    df["TransactionMonth"] = df[datetime_column].dt.month
    df["TransactionDay"] = df[datetime_column].dt.day
    df["TransactionHour"] = df[datetime_column].dt.hour
    return df

def encode_categorical_variables(df, categorical_cols, method="label"):
    """Encode categorical variables using Label Encoding or One-Hot Encoding."""
    if method == "label":
        encoder = LabelEncoder()
        for col in categorical_cols:
            df[col] = encoder.fit_transform(df[col])
    elif method == "onehot":
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    return df

def handle_missing_values(df, method="mean"):
    """Impute missing values using the specified method."""
    if method == "mean":
        df.fillna(df.mean(), inplace=True)
    elif method == "median":
        df.fillna(df.median(), inplace=True)
    elif method == "mode":
        df.fillna(df.mode().iloc[0], inplace=True)
    return df

def scale_features(df, numerical_cols, method="standard"):
    """Scale numerical features using Standard or Min-Max Scaler."""
    scaler = StandardScaler() if method == "standard" else MinMaxScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    return df

if __name__ == "__main__":
    data_path = "D:/Kifya_training/Week 6/Bati-Bank-Credit-Scoring-Model/data/raw/data.csv"
    data = pd.read_csv(data_path)

