from fastapi import FastAPI, HTTPException
from firecrawl import FirecrawlApp
from pydantic import BaseModel
import logging
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app with the expected name
fastapi_app = FastAPI(
    title="Medical Content Extractor",
    description="Extracts medical content using Firecrawl",
    version="1.0.0"
)

# Firecrawl setup
API_KEY = os.getenv("FIRECRAWL_API_KEY")
if not API_KEY:
    logger.error("Missing FIRECRAWL_API_KEY")
    raise ValueError("FIRECRAWL_API_KEY required")

firecrawl = FirecrawlApp(api_key=API_KEY)
logger.info("Firecrawl initialized")

# Pydantic model
class UrlRequest(BaseModel):
    url: str

# Content extraction logic
def extract_medical_content(url: str) -> str:
    try:
        scraped = firecrawl.scrape_url(url)
        return scraped.get('markdown', 'No content found')
    except Exception as e:
        logger.error(f"Extraction failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Content extraction failed")

# API endpoints
@fastapi_app.post("/extract")
async def extract_content(request: UrlRequest):
    return {
        "url": request.url,
        "content": extract_medical_content(request.url)
    }

@fastapi_app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Only for local testing (DigitalOcean ignores this when Procfile exists)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:fastapi_app", host="0.0.0.0", port=8080)
