# Use an official Python runtime as the base image
FROM python:3.13-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install system dependencies required by Playwright
RUN apt-get update && apt-get install -y \
    libnss3 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libxkbcommon0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libasound2 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright browser binaries
RUN playwright install --with-deps

# Copy the rest of the application code
COPY . .

# Expose the port the app runs on (Digital Ocean App Platform sets $PORT)
EXPOSE $PORT

# Command to run the application, using $PORT environment variable
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port $PORT"]
