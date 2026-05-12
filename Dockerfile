FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY app/     ./app/
COPY data/    ./data/
COPY scripts/ ./scripts/
COPY models/  ./models/

# Train model if not already present (for fresh builds)
RUN python scripts/train.py --samples 10000 || echo "⚠️  Training skipped (model may already exist)"

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
