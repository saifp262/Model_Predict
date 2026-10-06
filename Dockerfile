# ==============================================================================
# MLOPS UNIT END PROJECT 1: PREDICTIVE MODELING & DEPLOYMENT DOCKERFILE
# ==============================================================================

FROM python:3.11-slim

WORKDIR /app

# Install OS utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application artifacts
COPY app.py .
COPY best_model.pkl .
COPY titanic.csv .

# Expose FastAPI service port
EXPOSE 8000

# Healthcheck probe
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Launch ASGI server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
