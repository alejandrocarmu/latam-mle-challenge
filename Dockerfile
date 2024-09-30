# syntax=docker/dockerfile:1.2
# Use the official Python image from the Docker Hub
FROM python:3.10-slim

# Set environment variables to avoid Python writing .pyc files and buffering output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory inside the container
WORKDIR /app

# Copy the requirements file first for layer caching
COPY requirements.txt /app/requirements.txt

# Install any system dependencies and install Python dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --upgrade pip && \
    pip install -r /app/requirements.txt

# Copy the rest of the application code
COPY . /app

# Expose port 8000 (default for FastAPI)
EXPOSE 8000

# Run the FastAPI application using uvicorn
CMD ["uvicorn", "challenge.api:app", "--host", "0.0.0.0", "--port", "8080"]
