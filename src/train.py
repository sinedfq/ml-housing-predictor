import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
import json
import joblib
import os

def train_model():
    model = RandomForestRegressor(n_estimators=100, random_state=True)
    nb = pd.read_csv('data/new_builds.csv', encoding='utf-8')

    columns_to_use = ['total_area', 'rooms', 'to_center_km', 
                    'price_rub', 'metro_distance_min', 
                    'district']
    
    df_clean = nb[columns_to_use].dropna()
    df_encoded = pd.get_dummies(df_clean, columns=['district'])
    print("New DF: ", df_encoded.head())

    X = df_encoded.drop('price_rub', axis=1)
    y = df_encoded['price_rub']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    
    # A metric for determining the accuracy of an assumption
    r2 = r2_score(y_test, predictions)
    print("R2 metic:", r2)
    
    metrics = {
        "r2_score": float(r2),
        "features_count": len(X.columns)
    }

    with open('models/metrics.json', 'w') as f:
        json.dump(metrics, f)

    os.makedirs('models', exist_ok=True)
    # Export Model
    joblib.dump(model, 'models/housing_model.plk')
    # Export columns for API
    joblib.dump(list(X.columns), 'models/column_names.pkl') 

    # feature_importance = pd.Series(model.coef_, index=X.columns)
    # print("\n--- Важность признаков (коэффициенты) ---")
    # print(feature_importance.sort_values(ascending=False).head(10))

if __name__ == "__main__":
    train_model()