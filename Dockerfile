FROM python:3.13-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY harness/ harness/
COPY tools/ tools/
COPY configs/ configs/

# Default: the mock sweep, which needs no API key. Override the command in the
# Job manifest to run a real config once secrets are mounted.
ENTRYPOINT ["python", "-m", "harness.run_eval"]
CMD ["--config", "configs/mock.yaml", "--trials", "30", "--out", "/results/run.jsonl"]
