from fastapi import FastAPI
import joblib
import json
from pydantic import BaseModel
import pandas as pd
import torch
import torch.nn as nn

app = FastAPI()

cat_boost_model = joblib.load('models/housing_model.plk')
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

trend_models = {}
max_prices = {}
classes = ['business', 'comfort', 'economy']

for cls in classes:
    t_model = PricePredictor(input_size=3, hidden_size=16) 
    t_model.load_state_dict(torch.load(f'models/trend_{cls}.pth', weights_only=True))
    t_model.eval()
    trend_models[cls] = t_model
    max_prices[cls] = joblib.load(f'models/max_price_{cls}.pkl')

history_df = pd.read_csv('data/class_monthly_stats.csv')

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
    prediction = cat_boost_model.predict(input_final)[0]

    house_class = house.complex_class.lower()

    if house_class in trend_models:
        class_history = history_df[history_df['complex_class'] == house_class]

        if len(class_history) >= 3:
            last_3 = class_history.tail(3)['avg_price'].values
            current_max = max_prices[house_class]
            last_3_norm = last_3 / current_max

            input_tensor = torch.FloatTensor(last_3_norm)

            with torch.no_grad():
                trend_pred_norm = trend_models[house_class](input_tensor)

            future_avg_price = trend_pred_norm[-1].item() * current_max
        else:
            future_avg_price = None 

    return {
        "predicted_price_rub": round(prediction, 2),
        "model_r2_score": round(metrics["r2_score"], 3), 
        "future_avg_price": round(future_avg_price, 4)
    }