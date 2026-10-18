FROM python:3.11-slim

# XGBoost needs this Linux library to use multiple CPU cores.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /code

COPY requirements.txt .

# On Linux, plain "xgboost" also installs a ~350 MB GPU library (nvidia-nccl).
# xgboost-cpu is the same code without it. We swap the name only inside the image.
RUN sed -i 's/^xgboost==/xgboost-cpu==/' requirements.txt \
    && pip install --no-cache-dir -r requirements.txt
# Then copy only what the running app needs.
COPY app ./app
COPY model ./model

EXPOSE 8000

# Use the PORT given by the hosting service, or 8000 on our own computer.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]