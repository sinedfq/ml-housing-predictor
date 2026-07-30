import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import joblib

class PricePredictor(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(PricePredictor, self).__init__()

        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.layer2(x)
        return x

def train_class_model(prices_norm, max_price, complex_class):
    windows_size = 3

    X, y = [], []

    for i in range(len(prices_norm) - windows_size):
        X.append(prices_norm[i:i+windows_size])
        y.append(prices_norm[i+windows_size])

    if not X: return

    X = np.array(X)
    y = np.array(y)

    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y)

    predicator = PricePredictor(input_size=3, hidden_size=16)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(predicator.parameters(), lr=0.01)

    epochs = 100

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = predicator(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], [Loss: {loss.item():4f}]')

    model_path = f'models/trend_{complex_class}.pth'
    
    os.makedirs('models', exist_ok=True)    

    torch.save(predicator.state_dict(), model_path)
    joblib.dump(max_price, f'models/max_price_{complex_class}.pkl')

    return predicator
    
df = pd.read_csv('data/class_monthly_stats.csv')
count = df['count'].values

# Business frame
df_business = df[df['complex_class'] == 'business']
business_prices = df_business['avg_price'].values
max_business_price = business_prices.max()
business_prices_norm = business_prices / max_business_price

# Comfort frame
df_comfort = df[df['complex_class'] == 'comfort']
comfort_prices = df_comfort['avg_price'].values
max_comfort_price = comfort_prices.max()
comfort_prices_norm = comfort_prices / max_comfort_price

# Economy frame 
df_economy = df[df['complex_class'] == 'economy']
economy_prices = df_economy['avg_price'].values
max_economy_price = economy_prices.max()
economy_prices_norm = economy_prices / max_economy_price

train_class_model(business_prices_norm, max_business_price, 'business')
train_class_model(comfort_prices_norm, max_comfort_price, 'comfort')
train_class_model(economy_prices_norm, max_economy_price, 'economy')







