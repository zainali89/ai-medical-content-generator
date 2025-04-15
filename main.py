from fastapi import FastAPI, Request
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from browser_use import Agent, Browser, BrowserConfig
import os
import asyncio
from dotenv import load_dotenv
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Create the FastAPI app instance
app = FastAPI()

# Define a Pydantic model for the request payload to match the frontend
class ArticleRequest(BaseModel):
    user_input_topic: str
    article_length: str
    target_audience: str
    description: str = ""  # Made optional with a default empty string
    youtube_links: bool
    reference_urls: bool
    docs_files: bool

# Function to fetch content from Medscape using BrowserUse
async def fetch_medscape_content(topic: str):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=os.environ["GOOGLE_API_KEY"]
    )
    task = f"""
    Navigate to https://www.medscape.com and search for "{topic}".
    Extract relevant content from the search results or article page.
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
    try:
        result = await agent.run()
        return result if result else "No relevant content found on Medscape."
    except Exception as e:
        return f"Error fetching Medscape content: {str(e)}"

# Define the /generate-article endpoint with logging
@app.post("/generate-article")
async def generate_article(request: Request):
    # Log the raw request body
    body = await request.json()
    logger.info(f"Received request payload: {body}")

    # Validate the request using the ArticleRequest model
    article_request = ArticleRequest(**body)

    # Extract the request data
    user_input_topic = article_request.user_input_topic
    article_length = article_request.article_length.lower()
    target_audience = article_request.target_audience.lower()
    description = article_request.description

    # Initialize the Gemini LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=os.environ["GOOGLE_API_KEY"]
    )

    # Fetch additional context from Medscape if any reference source is selected
    medscape_content = ""
    if article_request.reference_urls or article_request.youtube_links or article_request.docs_files:
        medscape_content = await fetch_medscape_content(user_input_topic)

    # Define the prompt for article generation
    prompt = f"""
    Write a {article_length} article on the topic "{user_input_topic}" for a {target_audience} audience.
    Use the following description as a guide: {description}
    If applicable, incorporate the following information from Medscape: {medscape_content}
    Ensure the article is clear, informative, and suitable for the target audience.
    """

    # Generate the article using Gemini LLM
    try:
        response = llm.invoke(prompt)
        generated_content = response.content
    except Exception as e:
        generated_content = f"Error generating article: {str(e)}"

    # Adjust the article length (simplified logic for demonstration)
    if article_length == "short":
        # Limit to a short paragraph (e.g., first 100 words)
        words = generated_content.split()
        generated_content = " ".join(words[:100])
    elif article_length == "medium":
        # Limit to a medium length (e.g., first 300 words)
        words = generated_content.split()
        generated_content = " ".join(words[:300])
    # For "long", use the full generated content

    return {"status": "success", "content": generated_content}

# Add a health check endpoint for the readiness probe
@app.get("/")
async def health_check():
    return {"message": "App is running"}

# Define the /browse-medscape endpoint for testing
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
