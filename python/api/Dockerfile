FROM python:3.11-slim AS base
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY python/api/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY python /app/python
COPY data/results /app/data/results
EXPOSE 8081
HEALTHCHECK --interval=15s --timeout=3s CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8081/healthz')"
CMD ["uvicorn", "python.api.predictor_api:app", "--host", "0.0.0.0", "--port", "8081"]
