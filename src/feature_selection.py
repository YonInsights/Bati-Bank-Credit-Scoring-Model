import pandas as pd

def select_relevant_features(df, target_column, correlation_threshold=0.05):
    """
    Select features with correlation above the specified threshold with the target variable.
    """
    numeric_df = df.select_dtypes(include=["int64", "float64"])
    correlation = numeric_df.corr()[target_column]
    relevant_features = correlation[abs(correlation) > correlation_threshold].index.tolist()
    relevant_features.remove(target_column)
    return relevant_features
