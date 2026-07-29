import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt
import json
import joblib
import os

def train_model():
    cat_features_list = ['district', 'developer', 'complex_class', 'okrug']
  
    nb = pd.read_csv('data/new_builds.csv', encoding='utf-8')

    columns_to_use = ['total_area', 'rooms', 'to_center_km', 
                    'price_rub', 'metro_distance_min', 
                    'district', 'developer', 'floor', 
                    'total_floors', 'complex_class', 'okrug']
    
    df_clean = nb[columns_to_use].dropna()

    print("New DF: ", df_clean.head())
    X = df_clean.drop('price_rub', axis=1)
    y = df_clean['price_rub']

    model = CatBoostRegressor(
            iterations=500,
            learning_rate=0.1,
            depth=6,
            cat_features=cat_features_list,
            verbose=False,
            random_state=42
        )

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

    # Graph for comparing the differences between actual and predicted values
    plt.figure(figsize=(10, 6))
    
    plt.scatter(y_test, predictions, alpha=0.5, color='blue', label='Predictions')
    
    min_val = min(y_test.min(), predictions.min())
    max_val = max(y_test.max(), predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal Prediction')
    
    plt.title(f'Actual vs Predicted Prices (R² = {r2:.2f})')
    plt.xlabel('Actual Price (Rub)')
    plt.ylabel('Predicted Price (Rub)')
    plt.legend()
    
    plt.savefig('models/prediction_quality.png')
    plt.show()

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