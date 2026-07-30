#!/bin/bash

if [ ! -f "models/housing_model.plk" ]; then
    echo "Models not found. Starting training pipeline..."
    python src/prepare_data.py      
    python src/train.py             
    python src/prepare_time_series.py 
    python src/train_nn.py          
else
    echo "Models found. Skipping training."
fi

echo "Starting FastAPI server..."
exec uvicorn api.main:app --host 0.0.0.0 --port 8000