# Use official Python 3.11 slim image (TensorFlow compatible)
FROM python:3.11-slim

# Prevent Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1

# Ensure stdout/stderr are flushed immediately
ENV PYTHONUNBUFFERED=1

# Set NLTK data directory (important for Render)
ENV NLTK_DATA=/usr/local/nltk_data

# Set working directory
WORKDIR /app

# Install system dependencies (required for TensorFlow, numpy, lxml, etc.)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (Docker layer caching)
COPY requirements.txt .

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Download required NLTK resources at build time
RUN python -m nltk.downloader stopwords wordnet punkt

# Copy the rest of the application code
COPY . .

# Expose the port (Render uses $PORT dynamically)
EXPOSE 10000

# Start the application with Gunicorn
CMD sh -c "gunicorn app:app --bind 0.0.0.0:${PORT:-10000} --timeout 120"
