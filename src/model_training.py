import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def preprocess_target_column(df, target_column):
    """Convert continuous target values to discrete categories."""
    df[target_column] = (df[target_column] > 0).astype(int)
    return df

def split_data(df, target_column, test_size=0.2, random_state=42):
    """Split the dataset into training and testing sets."""
    # Select only numeric features
    X = df.select_dtypes(include=["int64", "float64"])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def train_logistic_regression(X_train, y_train):
    """Train a Logistic Regression model with class weighting."""
    model = LogisticRegression(random_state=42, max_iter=1000, class_weight="balanced")
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train):
    """Train a Random Forest Classifier with class weighting."""
    model = RandomForestClassifier(random_state=42, n_estimators=100, class_weight="balanced")
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and return performance metrics."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
    }
    if y_prob is not None:
        metrics["roc_auc"] = roc_auc_score(y_test, y_prob)
    return metrics
