import pandas as pd
import uvicorn
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from typing import Dict

from src.helpers.mlflow import MlflowModel
from src.server.schemas import BreastCancerData

model = MlflowModel("BreastCancerModel")
threshold = float(model.parameters["threshold"])
app = FastAPI()


@app.get("/")
def hello() -> str:
    """
    Endpoint to test if the API is running.

    Returns:
        str: A welcome message.
    """
    return "BreastCancerModel API"


@app.post("/predict")
def predict(data: BreastCancerData) -> Dict[str, float]:
    """
    Endpoint to predict breast cancer using provided features.

    Args:
        data (BreastCancerData): Input data containing the features required for
            the prediction.

    Returns:
        Dict[str, float]: Prediction result and threshold value.
    """
    df = pd.DataFrame(jsonable_encoder(data), index=[0])
    prediction = model.predict(df)
    return {"prediction": prediction[1], "threshold": threshold}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
