import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import joblib
    
df = pd.read_csv('data/monthly_stats.csv')
prices = df['avg_price'].values

max_price = prices.max()
prices_norm = prices / max_price

windows_size = 3

X, y = [], []

for i in range(len(prices_norm) - windows_size):
    X.append(prices_norm[i:i+windows_size])
    y.append(prices_norm[i+windows_size])

X = np.array(X)
y = np.array(y)

X_tensor = torch.FloatTensor(X).unsqueeze(-1)
y_tensor = torch.FloatTensor(y)


class PricePredictor(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(PricePredictor, self).__init__()

        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.layer2(x)
        return x

predicator = PricePredictor(input_size=1, hidden_size=16)

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

prediction = predicator(X_tensor[-1])
result = prediction * max_price

os.makedirs('models', exist_ok=True)
torch.save(predicator.state_dict(), 'models/price_trend_model.pth')
joblib.dump(max_price, 'models/max_price.pkl')
