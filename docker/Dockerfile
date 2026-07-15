FROM python:3.13-slim

WORKDIR /app

# Install system deps for psycopg2 and ChromaDB
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libpq-dev curl && \
    rm -rf /var/lib/apt/lists/*

COPY recruitment_ai/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create non-root user
RUN useradd -m -u 1001 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

CMD ["uvicorn", "recruitment_ai.api.main:app", "--host", "0.0.0.0", "--port", "8001", "--timeout-keep-alive", "300"]
