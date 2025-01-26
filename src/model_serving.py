from fastapi import FastAPI
import joblib
import pandas as pd

# Initialize FastAPI app
app = FastAPI()

# Load the trained model
model_path = "D:/Kifya_training/Week 6/Bati-Bank-Credit-Scoring-Model/models/tuned_rf_model.pkl"
model = joblib.load(model_path)

# Define the prediction endpoint
@app.post("/predict/")
async def predict(features: dict):
    """
    Endpoint to make predictions with the trained model.
    Args:
        features (dict): A dictionary of input features.

    Returns:
        dict: Prediction and probability.
    """
    # Convert input features to DataFrame
    input_df = pd.DataFrame([features])
    
    # Make predictions
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)[:, 1]

    return {
        "prediction": int(prediction[0]),
        "probability": float(probability[0])
    }
