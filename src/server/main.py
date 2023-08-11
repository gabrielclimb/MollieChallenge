import pandas as pd
import uvicorn
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder

from src.helpers.mlflow import MlflowModel
from src.server.schemas import BreastCancerData

model = MlflowModel("BreastCancerModel")

app = FastAPI()


@app.get("/")
def hello() -> str:
    return "BreastCancerModel API"


@app.post("/predict")
def predict(data: BreastCancerData) -> dict:
    df = pd.DataFrame(jsonable_encoder(data), index=[0])
    prediction = model.predict(df)
    return {"prediction": prediction[0]}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
