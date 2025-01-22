import pandas as pd

def load_data(file_path):
    """Load data from a CSV file."""
    return pd.read_csv(file_path)

def preprocess_data(df):
    """Basic data preprocessing steps."""
    # Handle missing values
    df = df.dropna()  # Example: drop rows with missing values
    return df

if __name__ == "__main__":
    data_path = "D:/Kifya_training/Week 6/Bati-Bank-Credit-Scoring-Model/data/raw/data.csv"
    data = load_data(data_path)
    processed_data = preprocess_data(data)
    print("Data preprocessing complete.")
