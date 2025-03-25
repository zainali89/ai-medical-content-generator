from fastapi import FastAPI, HTTPException
from firecrawl import FirecrawlApp
from pydantic import BaseModel
import logging
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Firecrawl Main Content Extractor API",
    description="Extracts main content from webpages using Firecrawl",
    version="1.0.0"
)

# Get the API key from environment variable
API_KEY = os.getenv("FIRECRAWL_API_KEY")
if not API_KEY:
    logger.error("FIRECRAWL_API_KEY not found in environment variables")
    raise ValueError("FIRECRAWL_API_KEY not found in environment variables")

# Initialize Firecrawl with the API key from .env
firecrawl_app = FirecrawlApp(api_key=API_KEY)
logger.info("Firecrawl initialized successfully")

# Define input model for URL
class UrlRequest(BaseModel):
    url: str

# Helper function to extract main content
def extract_main_content(url: str, timeout: int = 30) -> str:
    try:
        # Scrape the page
        result = firecrawl_app.scrape_url(url, params={"timeout": timeout * 1000})  # Timeout in milliseconds
        
        # Check if scrape was successful and has data
        if not result or "markdown" not in result:
            return "Failed to scrape the webpage"
        
        # Get the markdown content
        markdown_content = result["markdown"]
        
        # Split content into lines
        lines = markdown_content.split("\n")
        filtered_lines = []
        in_main_content = True

        for i, line in enumerate(lines):
            line = line.strip()

            # Skip empty lines
            if not line:
                if in_main_content:
                    filtered_lines.append(line)
                continue

            # Stop at "References" section
            if line.lower().startswith("## references") or line.lower().startswith("# references"):
                in_main_content = False
                break

            # Skip metadata-like sections after halfway or post-main content
            if (line.startswith("## ") and i > len(lines) // 2) or not in_main_content:
                in_main_content = False
            elif line.startswith("## ") and in_main_content:
                # Keep section headers in main content
                pass

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
def extract_main_content_endpoint(request: UrlRequest):
    """
    Extract the main content from a webpage given its URL.
    Returns the filtered content as a string in JSON format.
    """
    try:
        logger.info(f"Processing request for URL: {request.url}")
        main_content = extract_main_content(request.url)
        return {"url": request.url, "main_content": main_content}
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Health check endpoint
@app.get("/health")
def health_check():
    """Check if the API is running."""
    logger.info("Health check requested")
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    # Get port from environment variable or default to 8080
    port = int(os.getenv("PORT", 8080))
    logger.info(f"Starting server on port {port}")
    # Run the app with uvicorn for local testing
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
