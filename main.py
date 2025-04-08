# At the very beginning of main.py, after imports but before any other code
from fastapi import FastAPI
fastapi_app = FastAPI()

@fastapi_app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Only initialize connections to external services after defining this basic health endpoint
