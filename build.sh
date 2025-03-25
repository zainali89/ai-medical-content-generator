export PLAYWRIGHT_BROWSERS_PATH=/ms-playwright && \
mkdir -p /ms-playwright && \
apt-get update && \
apt-get install -y --no-install-recommends \
libnss3 libatk1.0-0 libatk-bridge2.0-0 libcups2 libdrm2 libxkbcommon0 \
libxcomposite1 libxdamage1 libxfixes3 libxrandr2 libgbm1 libasound2 && \
rm -rf /var/lib/apt/lists/* && \
pip install -r requirements.txt && \
playwright install --with-deps chromium
