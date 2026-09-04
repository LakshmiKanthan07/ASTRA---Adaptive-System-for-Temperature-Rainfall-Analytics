FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
# libeccodes is required for cfgrib
# libhdf5 and libnetcdf are required for netCDF4
RUN apt-get update && apt-get install -y \
    build-essential \
    libeccodes-dev \
    libhdf5-dev \
    libnetcdf-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Set PYTHONPATH
ENV PYTHONPATH=/app/src
