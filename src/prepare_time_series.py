import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt
import json
import joblib
import os

def prepare_time_series():
    nb = pd.read_csv('data/new_builds.csv', encoding='utf-8')
    nb['date_posted'] = pd.to_datetime(nb['date_posted'])
    nb['month'] = nb['date_posted'].dt.to_period('M').astype(str)

    monthly_data = nb.groupby(['complex_class', 'month']).agg(
        avg_price=('price_rub', 'mean'),
        count=('id', 'count')
    ).reset_index()
    monthly_data = monthly_data.sort_values(['complex_class', 'month'])

    os.makedirs('data', exist_ok=True)
    monthly_data.to_csv('data/class_monthly_stats.csv', index=False)

if __name__ == "__main__":
    prepare_time_series()