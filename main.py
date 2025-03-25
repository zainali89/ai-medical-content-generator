from fastapi import FastAPI, HTTPException
from crawl4ai import AsyncWebCrawler
from pydantic import BaseModel
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set Playwright browsers path
os.environ["PLAYWRIGHT_BROWSERS_PATH"] = "/ms-playwright"

# Initialize FastAPI app
app = FastAPI(title="Web Crawler API", description="Extracts main content from webpages", version="1.0.0")
fastapi_app = app  # Alias for compatibility with potential misconfiguration

# Define input model for URL
class UrlRequest(BaseModel):
    url: str

# Helper function to extract main content
async def extract_main_content(url: str) -> str:
    try:
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url)
            if not result.success:
                raise ValueError("Failed to crawl the webpage")

            markdown_content = result.markdown
            lines = markdown_content.split("\n")
            filtered_lines = []
            in_main_content = True
            saw_references = False

            for i, line in enumerate(lines):
                line = line.strip()
                if not line:
                    if in_main_content:
                        filtered_lines.append(line)
                    continue
                if line.startswith("## References"):
                    saw_references = True
                    in_main_content = False
                    break
                if saw_references or (line.startswith("## ") and i > len(lines) // 2):
                    in_main_content = False
                elif line.startswith("## ") and not saw_references:
                    in_main_content = True
                if in_main_content and not line.startswith("**Figure") and not line.startswith("**Table"):
                    filtered_lines.append(line)

            return "\n".join(filtered_lines).strip()

    except Exception as e:
        logger.error(f"Error processing URL {url}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing URL: {str(e)}")

# API endpoint
@app.post("/extract-main-content", response_model=dict)
async def extract_main_content_endpoint(request: UrlRequest):
    logger.info(f"Processing request for URL: {request.url}")
    main_content = await extract_main_content(request.url)
    return {"url": request.url, "main_content": main_content}

# Health check endpoint
@app.get("/health")
async def health_check():
    logger.info("Health check requested")
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    host = "0.0.0.0"
    logger.info(f"Starting app on {host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")
