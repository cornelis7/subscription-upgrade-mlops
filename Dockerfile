# Why containerize? "It works on my machine" is not good enough for
# production -- Docker guarantees the exact same Python version, OS libs,
# and dependency versions run everywhere: your laptop, CI, and the
# production server.

FROM python:3.12-slim

WORKDIR /code

# Install dependencies first (separate layer) so Docker can cache this step
# and skip reinstalling every package every time only app code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the actual application code and pre-trained model artifacts
COPY app/ ./app/
COPY src/ ./src/
COPY models/ ./models/
COPY mlruns/ ./mlruns/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health').raise_for_status()"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
