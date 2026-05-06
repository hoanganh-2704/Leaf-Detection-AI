FROM python:3.12-slim

WORKDIR /app

# Install system dependencies required by rembg, OpenCV and PyTorch
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . .

# Make the entrypoint executable
RUN chmod +x /app/docker-entrypoint.sh

# Ensure Python can find the project root
ENV PYTHONPATH="/app"

# NOTE: ChromaDB initialisation is done at container START (in entrypoint)
# because GEMINI_API_KEY is only available at runtime, not during build.
ENTRYPOINT ["/app/docker-entrypoint.sh"]

# Default: run the Streamlit frontend (overridden by docker-compose for backend)
CMD ["streamlit", "run", "src/ui/streamlit_app.py", \
     "--server.port=8506", "--server.address=0.0.0.0"]
