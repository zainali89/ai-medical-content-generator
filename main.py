from fastapi import FastAPI
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from browser_use import Agent, Browser, BrowserConfig
import os
import asyncio
from dotenv import load_dotenv

load_dotenv()

# Create the FastAPI app instance
app = FastAPI()

# Define a Pydantic model for the request payload to match the frontend
class ArticleRequest(BaseModel):
    user_input_topic: str
    article_length: str
    target_audience: str
    description: str
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

# Define the /generate-article endpoint
@app.post("/generate-article")
async def generate_article(request: ArticleRequest):
    # Extract the request data
    user_input_topic = request.user_input_topic
    article_length = request.article_length.lower()
    target_audience = request.target_audience.lower()
    description = request.description

    # Initialize the Gemini LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=os.environ["GOOGLE_API_KEY"]
    )

    # Fetch additional context from Medscape if any reference source is selected
    medscape_content = ""
    if request.reference_urls or request.youtube_links or request.docs_files:
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

# Function to generate article with hardcoded inputs
async def generate_article_with_hardcoded_inputs():
    # Hardcoded inputs matching the frontend form
    hardcoded_request = ArticleRequest(
        user_input_topic="Animal Allergy",
        article_length="Short",
        target_audience="public",
        description="A brief overview of animal allergies, their symptoms, and management.",  # Hardcoded description
        youtube_links=False,
        reference_urls=False,
        docs_files=False
    )

    # Call the generate_article function with the hardcoded inputs
    result = await generate_article(hardcoded_request)
    print("Generated Article:", result)

# Run the hardcoded generation when the script is executed directly
if __name__ == "__main__":
    asyncio.run(generate_article_with_hardcoded_inputs())
