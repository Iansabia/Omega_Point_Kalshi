FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY ./src /app/src
COPY ./config /app/config

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV GEMINI_API_KEY=""
ENV POLYMARKET_PRIVATE_KEY=""
ENV SPORTRADAR_API_KEY=""

# Run application
CMD ["python", "src/main.py"]
