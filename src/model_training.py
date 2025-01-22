import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def train_model(df, target_column):
    """Train a Random Forest model."""
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    print("Model training complete.")
    return model

if __name__ == "__main__":
    processed_path = "D:/Kifya_training/Week 6/Bati-Bank-Credit-Scoring-Model/data/processed/processed_data.csv"
    data = pd.read_csv(processed_path)
    trained_model = train_model(data, target_column="FraudResult")
