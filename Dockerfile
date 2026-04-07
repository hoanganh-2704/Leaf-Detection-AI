FROM python:3.12-slim

WORKDIR /app

# Install system dependencies required for CV models and PyTorch
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy dependencies logic first for caching
COPY requirements.txt .

# Install Python requirements
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Ensure PYTHONPATH is correctly targeted at the project root
ENV PYTHONPATH="${PYTHONPATH}:/app"

# Setup the Chroma database exactly once during the container build
RUN python -m src.core.knowledge_setup

# Default execution target (Can be overridden by Compose)
CMD ["streamlit", "run", "src/ui/streamlit_app.py", "--server.port=8506", "--server.address=0.0.0.0"]
