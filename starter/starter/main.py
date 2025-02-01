from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from starter.ml.data import process_data
from train_model import cat_features, encoder, lb, loaded_model


class ModelInput(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str


app = FastAPI()


@app.get("/")
async def greeting():
    return {"greeting": "Greetings! The following app makes predictions on income range based on demographic and census-related factors"}


@app.post("/inference")
async def make_inference(data: ModelInput):
    data_dict = data.dict()
    data_dict = {key.replace("_", "-"): val for key, val in data_dict.items()}
    df = pd.DataFrame([data_dict])
    X, _, _, _ = process_data(df,
                              categorical_features=cat_features,
                              label=None,
                              training=False,
                              encoder=encoder,
                              lb=lb)

    output = loaded_model.predict(X)
    if output == 1:
        out_str = "Salary is over 50k"
    elif output == 0:
        out_str = "Salary is less than or equal to 50k"

    return {"output": out_str}
