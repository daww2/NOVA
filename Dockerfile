FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libpq-dev \
    pkg-config \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# Copy application source
COPY . .

# Create data and logs directories
RUN mkdir -p /app/data /app/logs

EXPOSE 8000

# main.py defines `app` — correct module path is main:app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
