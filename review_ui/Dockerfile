FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HYPERTRACE_DATA_DIR=/app/data \
    HYPERTRACE_DB_PATH=/data/hypertrace_study.sqlite3 \
    HYPERTRACE_DURABLE_STORAGE=0 \
    HYPERTRACE_PREVIEW_MODE=0

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN mkdir -p /data && useradd --create-home --uid 1000 appuser && chown -R appuser:appuser /app /data
USER appuser

EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--proxy-headers"]
