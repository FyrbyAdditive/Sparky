FROM python:3.12-slim

WORKDIR /app

# graph extra: the txtai-wikipedia-slim index includes a graph component
RUN pip install --no-cache-dir "txtai[graph]>=8.0" "fastapi>=0.115" "uvicorn>=0.30" "huggingface-hub>=0.26"

COPY app.py .

EXPOSE 8040

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8040"]
