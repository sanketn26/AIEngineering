# Build from repository root: docker build -f examples/production_triage/Dockerfile -t triage-reference .
FROM python:3.11-slim
WORKDIR /app
COPY examples/production_triage/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY src /app/src
COPY examples/production_triage /app/examples/production_triage
RUN mkdir /data && chown 10001:10001 /data
USER 10001:10001
ENV TRIAGE_DB=/data/state.sqlite
EXPOSE 8000
CMD ["uvicorn", "examples.production_triage.service:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
