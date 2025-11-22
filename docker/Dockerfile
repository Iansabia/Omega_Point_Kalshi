# Multi-stage build for optimized production image
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt


# Production stage
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local

# Copy source code
COPY ./src /app/src
COPY ./config /app/config

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV GEMINI_API_KEY=""
ENV POLYMARKET_PRIVATE_KEY=""
ENV SPORTRADAR_API_KEY=""
ENV KALSHI_API_KEY=""
ENV KALSHI_EMAIL=""
ENV KALSHI_PASSWORD=""

# Create non-root user for security
RUN useradd -m -u 1000 trader && \
    chown -R trader:trader /app
USER trader

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)" || exit 1

# Run application
CMD ["python", "-u", "src/main.py"]
