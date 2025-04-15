from fastapi import FastAPI
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from browser_use import Agent, Browser, BrowserConfig
import os
import asyncio
import httpx
from dotenv import load_dotenv

load_dotenv()

# Create the FastAPI app instance
app = FastAPI()

# Define a Pydantic model for the request payload
class ArticleRequest(BaseModel):
    user_input_topic: str
    article_length: str
    target_audience: str
    reference_urls: list
    docs_files: list
    youtube_links: list

# Define the /generate-article endpoint
@app.post("/generate-article")
async def generate_article(request: ArticleRequest):
    # Extract the request data
    user_input_topic = request.user_input_topic
    article_length = request.article_length
    target_audience = request.target_audience

    # Placeholder logic for generating an article
    # In a real app, you'd use an LLM or other logic to generate the article
    generated_content = f"Generated article on {user_input_topic} for {target_audience} audience ({article_length} length)."

    return {"status": "success", "content": generated_content}

# Add a health check endpoint for the readiness probe
@app.get("/")
async def health_check():
    return {"message": "App is running"}

# Define the /browse-medscape endpoint
@app.get("/browse-medscape")
async def browse_medscape_endpoint():
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=os.environ["GOOGLE_API_KEY"]
    )
    task = """
    Navigate to https://www.medscape.com and extract the page content.
    """
    agent = Agent(
        task=task,
        llm=llm,
        browser=Browser(
            BrowserConfig(
                headless=True,
                args=[
                    '--no-sandbox',
                    '--disable-dev-shm-usage',
                    '--disable-gpu',
                    '--disable-setuid-sandbox'
                ]
            )
        )
    )
    result = await agent.run()
    return {"status": "success", "medscape_content": result}

# Test logic to run when the script is executed directly
async def call_generate_article_endpoint():
    payload = {
        "user_input_topic": "Animal Allergy",
        "article_length": "Short",
        "target_audience": "general",
        "reference_urls": [],
        "docs_files": [],
        "youtube_links": []
    }
    url = "https://your-app.ondigitalocean.app/generate-article"  # Replace with your actual app URL
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, json=payload, timeout=30.0)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"status": "error", "message": str(e)}

async def browse_medscape():
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=os.environ["GOOGLE_API_KEY"]
    )
    task = """
    Navigate to https://www.medscape.com and extract the page content.
    """
    agent = Agent(
        task=task,
        llm=llm,
        browser=Browser(
            BrowserConfig(
                headless=True,
                args=[
                    '--no-sandbox',
                    '--disable-dev-shm-usage',
                    '--disable-gpu',
                    '--disable-setuid-sandbox'
                ]
            )
        )
    )
    result = await agent.run()
    return result

async def run_tests():
    article_result = await call_generate_article_endpoint()
    print("Generate Article Result:", article_result)
    medscape_result = await browse_medscape()
    print("Medscape Content:", medscape_result)

# Run the tests when the script is executed directly
if __name__ == "__main__":
    asyncio.run(run_tests())
