FROM python:3.11-slim

WORKDIR /app

# System deps for TA-Lib and psycopg2
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ wget libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create data and model dirs
RUN mkdir -p data/cache models/saved

EXPOSE 8000 8501

# Default: run API (override with streamlit for dashboard)
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
