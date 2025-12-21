# Use Python slim (Debian-based, more compatible)
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies needed for all packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
# Install PyTorch CPU-only first to avoid CUDA dependencies (reduces image size by ~1GB)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.3.0 && \
    pip install --no-cache-dir -r requirements.txt

# Copy application
COPY flask_app/ ./flask_app/

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PORT=8080

EXPOSE 8080

# Change to flask_app directory
WORKDIR /app/flask_app

# Run the application  
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "2", "--timeout", "120", "app:create_app()"]
