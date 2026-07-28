import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import joblib
import os

def train_model():
    model = LinearRegression()
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
    r2 = r2_score(y_test, predictions)

    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/housing_model.plk')

if __name__ == "__main__":
    train_model()