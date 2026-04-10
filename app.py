import uvicorn
from fastapi import FastAPI
import pandas as pd
import joblib

app = FastAPI()

model = joblib.load("./artifacts/model.pkl")

@app.get("/")
def test_api():
    return {"message": "Test API"}

@app.get("/predict")
def predict(data:dict):
    df = pd.DataFrame[data["features"]]
    prediction = model.predict(df)
    return {"prediction": float(prediction)}

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8001, reload=True)