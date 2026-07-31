from fastapi.testclient import TestClient
from api.main import app 

client = TestClient(app)
test_data = {
    "total_area": 54.5,
    "rooms": 2,
    "to_center_km": 8.3,
    "metro_distance_min": 7,
    "district": "Zayeltsovsky",
    "floor": 5,
    "total_floors": 17,
    "developer": "PIK", 
    "complex_class": "comfort",
    "okrug": "Central"
}

def test_predict_price():
    response = client.post("/predict", json=test_data)
    assert response.status_code == 200
    data = response.json()
    
    assert "predicted_price_rub" in data

    price = data["predicted_price_rub"]

    assert isinstance(price, float)
    assert price > 0

    assert "future_avg_price" in data 

    future_price = data["future_avg_price"]

    assert isinstance(future_price, float)
    assert future_price > 0