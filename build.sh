#!/bin/bash
set -e  # Exit on error
echo "Starting build.sh"

echo "Setting PLAYWRIGHT_BROWSERS_PATH"
export PLAYWRIGHT_BROWSERS_PATH=/ms-playwright

echo "Creating directory /ms-playwright"
mkdir -p /ms-playwright

echo "Updating apt-get"
apt-get update

echo "Installing system dependencies"
apt-get install -y --no-install-recommends \
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
  libasound2

echo "Cleaning up apt lists"
rm -rf /var/lib/apt/lists/*

echo "Installing Python dependencies"
pip install -r requirements.txt

echo "Installing Playwright Chromium"
playwright install --with-deps chromium

echo "Verifying Chromium installation"
ls -la /ms-playwright || echo "Directory /ms-playwright is empty or missing"
echo "build.sh completed"
