from fastapi import FastAPI, HTTPException
from crawl4ai import AsyncWebCrawler
from pydantic import BaseModel
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Web Crawler API", description="Extracts main content from webpages", version="1.0.0")

# Define input model for URL
class UrlRequest(BaseModel):
    url: str

# Helper function to extract main content
async def extract_main_content(url: str) -> str:
    try:
        async with AsyncWebCrawler() as crawler:
            # Run the crawler on the provided URL
            result = await crawler.arun(url=url)
            
            if not result.success:
                raise ValueError("Failed to crawl the webpage")

            # Get the markdown content
            markdown_content = result.markdown

            # Split the content into lines
            lines = markdown_content.split("\n")
            filtered_lines = []
            in_main_content = True
            saw_references = False

            for i, line in enumerate(lines):
                line = line.strip()

                # Skip empty lines outside main content
                if not line:
                    if in_main_content:
                        filtered_lines.append(line)
                    continue

                # Detect "References" as the end of main content
                if line.startswith("## References"):
                    saw_references = True
                    in_main_content = False
                    break

                # Skip metadata-like sections after halfway or post-references
                if saw_references or (line.startswith("## ") and i > len(lines) // 2):
                    in_main_content = False
                elif line.startswith("## ") and not saw_references:
                    in_main_content = True

                # Add line if in main content and not a figure/table
                if in_main_content and not line.startswith("**Figure") and not line.startswith("**Table"):
                    filtered_lines.append(line)

            # Join the filtered lines
            main_content = "\n".join(filtered_lines).strip()
            return main_content

    except Exception as e:
        logger.error(f"Error processing URL {url}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing URL: {str(e)}")

# Define the API endpoint
@app.post("/extract-main-content", response_model=dict)
async def extract_main_content_endpoint(request: UrlRequest):
    """
    Extract the main content from a webpage given its URL.
    Returns the filtered content as a string in JSON format.
    """
    try:
        logger.info(f"Processing request for URL: {request.url}")
        main_content = await extract_main_content(request.url)
        return {"url": request.url, "main_content": main_content}
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Health check endpoint
@app.get("/health")
async def health_check():
    """Check if the API is running."""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    # Run the app with uvicorn for local testing
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
