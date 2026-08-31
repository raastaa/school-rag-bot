FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV EMBEDDING_PROVIDER=sentence-transformers
ENV EMBEDDING_MODEL=intfloat/multilingual-e5-small
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
