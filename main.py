from langchain_google_genai import ChatGoogleGenerativeAI
from browser_use import Agent, Browser, BrowserConfig
import asyncio
import httpx
from dotenv import load_dotenv
import os

load_dotenv()

async def call_generate_article_endpoint():
    # Hardcoded inputs for the /generate-article endpoint
    payload = {
        "user_input_topic": "Animal Allergy",
        "article_length": "Short",
        "target_audience": "general",
        "reference_urls": [],
        "docs_files": [],
        "youtube_links": []
    }
    
    # Replace with your Digital Ocean App Platform URL
    url = "https://your-app.ondigitalocean.app/generate-article"
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, json=payload, timeout=30.0)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"status": "error", "message": str(e)}

async def browse_medscape():
    # Initialize Gemini LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=os.environ["GOOGLE_API_KEY"]
    )
    
    # Define the task for BrowserUse
    task = """
    Navigate to https://www.medscape.com and extract the page content.
    """
    
    # Configure BrowserUse for headless operation
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
    
    # Run the task and return the result
    result = await agent.run()
    return result

async def main():
    # Test the /generate-article endpoint directly
    article_result = await call_generate_article_endpoint()
    print("Generate Article Result:", article_result)
    
    # Test browser functionality by navigating to Medscape
    medscape_result = await browse_medscape()
    print("Medscape Content:", medscape_result)

if __name__ == "__main__":
    asyncio.run(main())
