import os
import logging
import subprocess
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from browser_use import Agent
from langchain_google_genai import ChatGoogleGenerativeAI
import asyncio
import uvicorn

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
def ensure_playwright_browsers():
    try:
        chromium_path = os.path.expanduser("~/.cache/ms-playwright/chromium-1161/chrome-linux/chrome")
        if not os.path.exists(chromium_path):
            logger.info("Playwright browsers not found. Installing...")
            subprocess.run(["playwright", "install"], check=True)
            logger.info("Playwright browsers installed successfully.")
        else:
            logger.info("Playwright browsers already installed.")
    except Exception as e:
        logger.error(f"Failed to install Playwright browsers: {str(e)}")
        raise RuntimeError("Playwright browser installation failed")

# Run the check on startup
ensure_playwright_browsers()

# Load environment variables
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
MEDSCAPE_USERNAME = os.environ.get("MEDSCAPE_USERNAME")
MEDSCAPE_PASSWORD = os.environ.get("MEDSCAPE_PASSWORD")

logger.info(f"GOOGLE_API_KEY: {'Set' if GOOGLE_API_KEY else 'Not set'}")
logger.info(f"MEDSCAPE_USERNAME: {'Set' if MEDSCAPE_USERNAME else 'Not set'}")
logger.info(f"MEDSCAPE_PASSWORD: {'Set' if MEDSCAPE_PASSWORD else 'Not set'}")

if not all([GOOGLE_API_KEY, MEDSCAPE_USERNAME, MEDSCAPE_PASSWORD]):
    raise ValueError("One or more required environment variables are missing.")

# Initialize FastAPI app
app = FastAPI()

# Pydantic model for the request
class TopicRequest(BaseModel):
    topic: str

# Function to scrape Medscape using browser_use
async def scrape_medscape(topic: str) -> dict:
    logger.info(f"Scraping Medscape for topic: {topic}")
    
    try:
        # Initialize the agent
        agent = Agent(
            task=f"""Go to https://www.medscape.com, log in with credentials Username: {MEDSCAPE_USERNAME} and Password: {MEDSCAPE_PASSWORD},
                  search for '{topic}', and extract the latest article information, its contents, and write a brief summary.
                  Return the summary and source in the format:
                  - Summary: [Summary text]
                  - Source: [Title (Author(s), Date). Link: [URL]]""",
            llm=ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                google_api_key=GOOGLE_API_KEY
            ),
        )
        
        # Run the agent and await the result
        result = await agent.run()
        logger.info(f"Agent run completed. Result type: {type(result)}")
        logger.info(f"Agent result content: {result}")
        
        # Parse the result
        medscape_data = []
        if result:
            # Assuming result is a list (AgentHistoryList), take the last element
            if isinstance(result, list) and result:
                final_output = result[-1]
                if isinstance(final_output, str):
                    # Parse the summary and source
                    lines = final_output.split("\n")
                    summary = ""
                    source = ""
                    for line in lines:
                        if line.startswith("- Summary:"):
                            summary = line.replace("- Summary:", "").strip()
                        elif line.startswith("- Source:"):
                            source = line.replace("- Source:", "").strip()
                    if summary and source:
                        medscape_data.append(f"Medscape content: {summary}\nSource: {source}")
                    else:
                        logger.warning("Failed to parse Medscape summary and source")
                        medscape_data.append("No relevant Medscape content found")
                else:
                    logger.error(f"Final output is not a string: {type(final_output)}")
                    medscape_data.append("No relevant Medscape content found")
            else:
                logger.error(f"Unexpected response type: {type(result)}")
                medscape_data.append("No relevant Medscape content found")
        else:
            logger.warning("No Medscape content returned by agent")
            medscape_data.append("No relevant Medscape content found")
        
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

# Run the app
if __name__ == "__main__":
    config = uvicorn.Config(app, host="0.0.0.0", port=8080, loop="asyncio")  # Use port 8080 for DigitalOcean
    server = uvicorn.Server(config)
    asyncio.run(server.serve())
