FROM python:3.11-slim
RUN apt-get update && apt-get install -y libreoffice-core libreoffice-writer unoconv curl && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
HEALTHCHECK CMD ["python", "-c", "import sys; sys.exit(0)"]
CMD ["python", "bot.py"]
