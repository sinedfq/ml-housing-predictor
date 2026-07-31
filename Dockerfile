FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# Исправленная логика запуска: обучаем ВСЁ, если чего-то не хватает
CMD sh -c " \
    if [ ! -f 'models/housing_model.plk' ] || [ ! -f 'models/trend_business.pth' ]; then \
        echo '⚠️ Модели не найдены. Запускаем полное обучение...'; \
        python src/prepare_data.py; \
        python src/train.py; \
        python src/prepare_time_series.py; \
        python src/train_nn.py; \
        echo '✅ Обучение завершено!'; \
    else \
        echo '✅ Модели найдены. Пропускаем обучение.'; \
    fi && \
    echo '🚀 Запуск FastAPI сервера...' && \
    uvicorn api.main:app --host 0.0.0.0 --port 8000 \
"