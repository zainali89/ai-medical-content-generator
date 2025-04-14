import os
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from playwright.sync_api import sync_playwright, Playwright
import uvicorn
from retrying import retry

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("main")

app = FastAPI()

# Environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MEDSCAPE_USERNAME = os.getenv("MEDSCAPE_USERNAME")
MEDSCAPE_PASSWORD = os.getenv("MEDSCAPE_PASSWORD")

# Request model for article generation
class ArticleRequest(BaseModel):
    user_input_topic: str
    user_input_description: str = ""
    article_length: str = "Short"
    target_audience: str = "general"
    reference_urls: list = []
    docs_files: list = []
    youtube_links: list = []

def setup_playwright() -> Playwright:
    """
    Initialize Playwright and install browsers if not present.
    """
    logger.info("Current working directory: %s", os.getcwd())
    logger.info("GOOGLE_API_KEY: %s", "Set" if GOOGLE_API_KEY else "Not set")
    logger.info("MEDSCAPE_USERNAME: %s", MEDSCAPE_USERNAME)
    logger.info("MEDSCAPE_PASSWORD: %s", MEDSCAPE_PASSWORD)

    playwright = sync_playwright().start()
    
    # Check if browsers are installed
    try:
        browser = playwright.chromium.launch()
        browser.close()
    except Exception as e:
        logger.info("Playwright browsers not found. Installing...")
        os.system("playwright install")
        logger.info("Playwright browsers installed successfully.")
    
    # Validate system dependencies
    try:
        browser = playwright.chromium.launch()
        browser.close()
        logger.info("Playwright system dependencies are satisfied.")
    except Exception as e:
        logger.error("Playwright system dependency check failed: %s", str(e))
        raise
    
    return playwright

@retry(stop_max_attempt_number=3, wait_fixed=2000)  # Retry 3 times with a 2-second delay between attempts
def scrape_medscape(playwright: Playwright, topic: str) -> str:
    """
    Scrape Medscape for the given topic.
    """
    browser = None
    try:
        # Launch browser with anti-bot measures
        browser = playwright.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        )
        page = context.new_page()

        # Navigate to Medscape login page with increased timeout
        logger.info("Navigating to Medscape login page")
        page.goto("https://login.medscape.com/login", timeout=60000)

        # Perform login
        logger.info("Attempting to log in to Medscape")
        page.fill('input[name="username"]', MEDSCAPE_USERNAME)
        page.fill('input[name="password"]', MEDSCAPE_PASSWORD)
        page.click('button[type="submit"]')

        # Wait for navigation after login
        try:
            page.wait_for_url("https://www.medscape.com/*", timeout=60000)
            logger.info("Login successful")
        except Exception as e:
            # Take a screenshot if login fails
            page.screenshot(path="login_failure.png")
            logger.error("Login failed, screenshot saved as login_failure.png: %s", str(e))
            raise Exception("Failed to log in to Medscape")

        # Search for the topic
        search_url = f"https://www.medscape.com/search?query={topic}"
        logger.info("Searching for topic: %s", topic)
        page.goto(search_url, timeout=60000)

        # Extract content (adjust selector based on Medscape's structure)
        content = page.inner_text("body")  # Replace with actual selector for article content
        logger.info("Successfully scraped content for topic: %s", topic)
        return content

    except Exception as e:
        logger.error("Medscape scraping failed: %s", str(e))
        raise Exception(f"Medscape scraping error: {str(e)}")
    finally:
        if browser:
            browser.close()

@app.post("/generate-article")
async def generate_article(request: ArticleRequest):
    """
    Generate an article based on the user input.
    """
    topic = request.user_input_topic
    logger.info("Received request to generate article for topic: %s", topic)

    # Initialize Playwright
    playwright = setup_playwright()
    try:
        # Scrape Medscape
        logger.info("Scraping Medscape for topic: %s", topic)
        medscape_content = scrape_medscape(playwright, topic)

        # Simplified article generation (replace with your actual logic)
        article = f"Article on {topic}\n\n{medscape_content[:500]}..."  # Truncate for demo
        return {"article": article}
    except Exception as e:
        logger.error("Article generation failed due to scraping errors: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Article generation failed: {str(e)}")
    finally:
        playwright.stop()

@app.get("/get-topics")
async def get_topics():
    """
    Endpoint removed - return 410 Gone.
    """
    logger.info("Received request to /get-topics, which has been removed")
    raise HTTPException(status_code=410, detail="This endpoint has been removed")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
