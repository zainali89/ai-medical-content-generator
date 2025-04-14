import os
import logging
import subprocess
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
import asyncio
import uvicorn
from playwright.async_api import async_playwright

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("webapp.log"), logging.StreamHandler()]
)
logger = logging.getLogger("main")

# Log current working directory
logger.info(f"Current working directory: {os.getcwd()}")

# Ensure Playwright browsers are installed at startup
async def ensure_playwright_browsers():
    try:
        # Check if Playwright browsers are already installed
        chromium_path = os.path.expanduser("~/.cache/ms-playwright/chromium-1161/chrome-linux/chrome")
        if not os.path.exists(chromium_path):
            logger.info("Playwright browsers not found. Installing...")
            subprocess.run(["playwright", "install"], check=True)
            logger.info("Playwright browsers installed successfully.")
        else:
            logger.info("Playwright browsers already installed.")

        # Test if Playwright can launch a browser in headless mode
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                await browser.close()
            logger.info("Playwright system dependencies are satisfied.")
        except Exception as e:
            logger.error(f"Playwright browser launch test failed: {str(e)}. System dependencies may be missing.")
            raise RuntimeError(f"Playwright setup failed: {str(e)}")

    except Exception as e:
        logger.error(f"Playwright setup failed: {str(e)}")
        raise RuntimeError("Playwright setup failed")

# Load environment variables
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

logger.info(f"GOOGLE_API_KEY: {'Set' if GOOGLE_API_KEY else 'Not set'}")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable is missing.")

# Hardcode Medscape credentials
MEDSCAPE_USERNAME = "shane@connectthedocs.com.au"
MEDSCAPE_PASSWORD = "Nelson01"

logger.info(f"MEDSCAPE_USERNAME: {MEDSCAPE_USERNAME}")
logger.info(f"MEDSCAPE_PASSWORD: {MEDSCAPE_PASSWORD}")

# Initialize FastAPI app
app = FastAPI()

# Run the Playwright setup during app startup
@app.on_event("startup")
async def startup_event():
    await ensure_playwright_browsers()

# Pydantic model for the /scrape-medscape request
class TopicRequest(BaseModel):
    topic: str

# Pydantic model for the /generate-article request
class GenerateArticleRequest(BaseModel):
    user_input_topic: str
    user_input_description: str = ""
    article_length: str = "Short"
    target_audience: str = "general"
    reference_urls: list[str] = []
    docs_files: list[str] = []
    youtube_links: list[str] = []

# Function to scrape Medscape using Playwright directly
async def scrape_medscape(topic: str) -> dict:
    logger.info(f"Scraping Medscape for topic: {topic}")
    
    try:
        async with async_playwright() as p:
            # Launch browser in headless mode
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()

            # Navigate to Medscape
            await page.goto("https://www.medscape.com")
            await page.wait_for_load_state("networkidle")

            # Log in
            await page.fill('input[name="username"]', MEDSCAPE_USERNAME)
            await page.fill('input[name="password"]', MEDSCAPE_PASSWORD)
            await page.click('button[type="submit"]')
            await page.wait_for_load_state("networkidle")

            # Search for the topic
            await page.fill('input[name="q"]', topic)
            await page.press('input[name="q"]', 'Enter')
            await page.wait_for_load_state("networkidle")

            # Extract the latest article (simplified selectors - adjust as needed)
            await page.wait_for_selector('article')
            article_title = await page.query_selector('article h2')
            title_text = await article_title.inner_text() if article_title else "No title found"
            article_link = await page.query_selector('article a')
            link_url = await article_link.get_attribute('href') if article_link else "No link found"

            # Use LLM to summarize if needed (optional)
            llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GOOGLE_API_KEY)
            summary_prompt = f"Summarize the following topic based on the title: {title_text}"
            summary_response = llm.invoke(summary_prompt)
            summary = summary_response.content if hasattr(summary_response, 'content') else f"This article discusses the latest findings on {topic}."

            # Format the source
            source = f"{title_text} (Unknown Author, Unknown Date). Link: {link_url}"

            await browser.close()

            medscape_data = [f"Medscape content: {summary}\nSource: {source}"]
        
        return {
            "medscape_data": medscape_data,
            "errors": [],
            "status": "success"
        }
    
    except Exception as e:
        logger.error(f"Medscape scraping failed: {str(e)}")
        return {
            "medscape_data": ["No relevant Medscape content found"],
            "errors": [f"Medscape scraping error: {str(e)}"],
            "status": "error"
        }

# Function to generate a simple article from Medscape data
def generate_article(medscape_data: list[str], topic: str, article_length: str, target_audience: str) -> str:
    logger.info(f"Generating article for topic: {topic}")
    
    # Define article length in approximate word counts
    length_map = {
        "Short": 150,
        "Medium": 300,
        "Long": 500
    }
    target_words = length_map.get(article_length, 150)  # Default to Short if invalid

    # Start the article with an introduction
    intro = f"This article provides an overview of {topic} for a {target_audience} audience.\n\n"
    
    # Combine Medscape data into the article body
    body = "Medscape Insights:\n"
    if medscape_data and medscape_data[0] != "No relevant Medscape content found":
        for data in medscape_data:
            body += f"{data}\n\n"
    else:
        body += f"No relevant information on {topic} was found in Medscape.\n\n"
    
    # Add a conclusion
    conclusion = f"In summary, this article has provided key insights into {topic} based on the latest medical information available."
    
    # Combine sections
    article = intro + body + conclusion
    
    # Trim the article to the target length (approximate by words)
    words = article.split()
    if len(words) > target_words:
        article = " ".join(words[:target_words]) + "..."
    
    logger.info(f"Generated article length: {len(article.split())} words")
    return article

# FastAPI endpoint to generate an article
@app.post("/generate-article")
async def generate_article_endpoint(request: GenerateArticleRequest):
    logger.info(f"Received request to generate article for topic: {request.user_input_topic}")
    
    # Scrape Medscape data
    medscape_result = await scrape_medscape(request.user_input_topic)
    
    # Check for errors in scraping
    if medscape_result["status"] == "error":
        logger.error(f"Article generation failed due to scraping errors: {medscape_result['errors']}")
        raise HTTPException(
            status_code=500,
            detail={"detail": medscape_result["errors"], "status": 500}
        )
    
    # Generate the article
    article = generate_article(
        medscape_data=medscape_result["medscape_data"],
        topic=request.user_input_topic,
        article_length=request.article_length,
        target_audience=request.target_audience
    )
    
    # Prepare the response in the format expected by the frontend
    response = {
        "generated_content": article,
        "performance_metrics": {
            "total_execution_time": 0  # Simplified; can add timing if needed
        },
        "errors": medscape_result["errors"]
    }
    
    logger.info("Article generation completed successfully")
    return JSONResponse(content=response)

# FastAPI endpoint to trigger Medscape scraping
@app.post("/scrape-medscape")
async def scrape_medscape_endpoint(request: TopicRequest):
    logger.info(f"Received request to scrape Medscape for topic: {request.topic}")
    result = await scrape_medscape(request.topic)
    if result["status"] == "error":
        raise HTTPException(status_code=500, detail={"detail": result["errors"], "status": 500})
    return JSONResponse(content=result)

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Placeholder for the removed /get-topics endpoint
@app.get("/get-topics")
async def get_topics():
    logger.info("Received request to /get-topics, which has been removed")
    return JSONResponse(
        status_code=410,
        content={"detail": "This endpoint has been removed."}
    )

# Run the app
if __name__ == "__main__":
    config = uvicorn.Config(app, host="0.0.0.0", port=8080, loop="asyncio")  # Use port 8080 for DigitalOcean
    server = uvicorn.Server(config)
    asyncio.run(server.serve())
