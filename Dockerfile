# Use Python alpine for smallest size and fastest builds
FROM python:3.11-alpine

# Set working directory
WORKDIR /app

# Install system dependencies (alpine uses apk, much faster)
RUN apk add --no-cache gcc musl-dev postgresql-dev

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY flask_app/ ./flask_app/

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PORT=8080

# Expose port
EXPOSE 8080

# Run the application
CMD cd flask_app && gunicorn --bind 0.0.0.0:$PORT --workers 2 --timeout 120 'app:create_app()'
