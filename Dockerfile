# Stage 1: Build stage
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies into virtual env
COPY requirements.txt .
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime stage
FROM python:3.11-slim AS runner

WORKDIR /app

# Copy virtual env from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application files
COPY config.yaml .
COPY processed.cleveland.data .
COPY src/ ./src/
COPY data/ ./data/
COPY models/ ./models/


# Set Python path
ENV PYTHONPATH=/app

# Expose FastAPI server port
EXPOSE 8000

# Run Uvicorn ASGI server
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
