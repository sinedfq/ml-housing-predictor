from fastapi import FastAPI
import joblib
import json
from pydantic import BaseModel
import pandas as pd

app = FastAPI()

model = joblib.load('models/housing_model.plk')
expected_columns = joblib.load('models/column_names.pkl')

with open('models/metrics.json', 'r') as f:
    metrics = json.load(f)

class HomeInput(BaseModel):
    total_area: float
    rooms: int 
    to_center_km: float
    metro_distance_min: int
    district: str
    floor: int
    total_floors: int
    complex_class: str
    developer: str
    okrug: str



@app.post("/predict")
def predict_price(house: HomeInput):
    input_data = pd.DataFrame([house.dict()])
    input_final = input_data.reindex(columns=expected_columns, fill_value=0)
    prediction = model.predict(input_final)[0]


    return {
        "predicted_price_rub": round(prediction, 2),
        "model_r2_score": round(metrics["r2_score"], 3), 
    }