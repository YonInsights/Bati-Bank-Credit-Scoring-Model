from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

# Load trained model
model = joblib.load("D:/Kifya_training/Week 6/Bati-Bank-Credit-Scoring-Model/models/credit_model.pkl")

@app.post("/predict/")
async def predict(data: dict):
    """Endpoint for making predictions."""
    df = pd.DataFrame([data])
    prediction = model.predict(df)
    return {"prediction": int(prediction[0])}

# Run the API with uvicorn (e.g., `uvicorn model_serving:app --reload`)
