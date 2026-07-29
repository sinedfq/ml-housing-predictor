from fastapi import FastAPI
import joblib
import json
from pydantic import BaseModel
import pandas as pd
import torch
import torch.nn as nn

app = FastAPI()

model = joblib.load('models/housing_model.plk')
expected_columns = joblib.load('models/column_names.pkl')

class PricePredictor(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(PricePredictor, self).__init__()

        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.layer2(x)
        return x

trend_model = PricePredictor(input_size=1, hidden_size=16)
trend_model.load_state_dict(torch.load('models/price_trend_model.pth', weights_only=True))
trend_model.eval()

history_df = pd.read_csv('data/monthly_stats.csv')
global_max_price = joblib.load('models/max_price.pkl')

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

    base_price = model.predict(input_final)[0]
    last_3_month = history_df.tail(3)['avg_price'].values / global_max_price
    input_tensor = torch.FloatTensor(last_3_month).unsqueeze(-1)

    with torch.no_grad():
        prediction_norm = trend_model(input_tensor)

    future_avg_price = prediction_norm[-1].item() * global_max_price



    return {
        "predicted_price_rub": round(prediction, 2),
        "model_r2_score": round(metrics["r2_score"], 3), 
        "market_trend_prediction": round(future_avg_price, 2)
    }