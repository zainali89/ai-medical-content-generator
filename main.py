import logging
import os
from dotenv import load_dotenv
import json
import requests
# Removing XML parser since it's only used for PubMed
from typing import TypedDict, List, Dict, Annotated
import operator
from langgraph.graph import StateGraph, END
from openai import OpenAI
import time
import datetime
from functools import wraps
import traceback
from fastapi import FastAPI, HTTPException
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from pydantic import BaseModel
import nest_asyncio
import uvicorn
from celery import Celery
import pytz
from celery.schedules import crontab
from fastapi.middleware.cors import CORSMiddleware
# --- New imports for Firecrawl ---
from firecrawl import FirecrawlApp

# Apply nest_asyncio to allow nested event loops (e.g., in Jupyter)
nest_asyncio.apply()

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()]
)

logger.info(f"Current working directory: {os.getcwd()}")
env_file_path = os.path.join(os.getcwd(), ".env")
logger.info(f"Looking for .env file at: {env_file_path}")
if os.path.exists(env_file_path):
    logger.info(".env file found")
else:
    logger.warning(".env file not found")

load_dotenv()

# API Keys and MongoDB URI from environment variables - removed PubMed API key
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
PERPLEXITY_API_KEY = os.environ.get("PERPLEXITY_API_KEY")
FIRECRAWL_API_KEY = os.environ.get("FIRECRAWL_API_KEY")
MONGODB_URI = os.environ.get(
    "MONGODB_URI",
    "mongodb+srv://syedbasitabbas10:FZg3aL0FbRYyxGdh@topmedicalarticles.pfo2g.mongodb.net/?retryWrites=true&w=majority&appName=TopMedicalArticles"
)

logger.info(f"OPENAI_API_KEY: {'Set' if OPENAI_API_KEY else 'Not set'}")
logger.info(f"PERPLEXITY_API_KEY: {'Set' if PERPLEXITY_API_KEY else 'Not set'}")
logger.info(f"FIRECRAWL_API_KEY: {'Set' if FIRECRAWL_API_KEY else 'Not set'}")
logger.info(f"MONGODB_URI: {'Set' if MONGODB_URI else 'Not set'}")

# Updated check without PubMed API key
if not all([OPENAI_API_KEY, PERPLEXITY_API_KEY, MONGODB_URI, FIRECRAWL_API_KEY]):
    raise ValueError("One or more required keys (API keys or MongoDB URI) are missing.")

# Initialize OpenAI client
openai_client = OpenAI(api_key=OPENAI_API_KEY)
logger.info("OpenAI client initialized.")

# --- New Firecrawl initialization ---
firecrawl = FirecrawlApp(api_key=FIRECRAWL_API_KEY)
logger.info("Firecrawl client initialized.")

# Connect to MongoDB
try:
    client_mongo = MongoClient(MONGODB_URI, server_api=ServerApi('1'))
    client_mongo.admin.command('ping')
    logger.info("Successfully connected to MongoDB!")
except Exception as e:
    logger.error(f"Error connecting to MongoDB: {str(e)}")
    raise ValueError(f"Error connecting to MongoDB: {str(e)}")

db = client_mongo['TopMedicalArticles']
collection = db['TrendingTopics']

# Celery setup
try:
    celery_app = Celery("tasks", broker="redis://localhost:6379/0", backend="redis://localhost:6379/0")
    logger.info("Celery app initialized.")
except Exception as e:
    logger.error(f"Error initializing Celery: {str(e)}")
    raise

celery_app.conf.timezone = "Australia/Sydney"
celery_app.conf.beat_schedule = {
    "update-medical-topics-daily": {
        "task": "main.fetch_and_store_topics_task",
        "schedule": crontab(hour=0, minute=0),  # 12 AM AEST/AEDT
        "options": {"timezone": "Australia/Sydney"}
    },
}
celery_app.conf.update(result_expires=3600)
logger.info("Celery schedule configured for 12 AM Australia/Sydney time.")

# Decorator to time functions
def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time
        logger.info(f"{func.__name__} took {duration:.4f} seconds")
        if isinstance(result, dict):
            result["performance_metrics"] = result.get("performance_metrics", {})
            result["performance_metrics"][func.__name__] = duration
        return result
    return wrapper

# --- Updated State without PubMed-related fields ---
class State(TypedDict):
    user_input_topic: str
    user_input_description: str
    article_length: str
    target_audience: str
    perplexity_data: List[str]
    firecrawl_data: List[str]
    generated_content: str
    errors: Annotated[List[str], operator.add]
    performance_metrics: Annotated[Dict[str, float], lambda x, y: {**x, **y}]
    critical_error: Annotated[bool, lambda x, y: x or y]

# --- New function to extract content using Firecrawl ---
@timeit
def extract_firecrawl_content(state: State) -> dict:
    logger.info(f"Extracting Firecrawl content for topic: {state['user_input_topic']}")
    errors = []
    firecrawl_data = []
    
    # Using the topic to search for relevant content - you might want to modify this URL
    search_url = f"https://www.ncbi.nlm.nih.gov/search/all/?term={state['user_input_topic'].replace(' ', '+')}"
    try:
        scraped = firecrawl.scrape_url(search_url)
        content = scraped.get('markdown', 'No content found')
        firecrawl_data.append(f"Firecrawl content from {search_url}: {content}")
        logger.info(f"Successfully extracted Firecrawl content from {search_url}")
    except Exception as e:
        errors.append(f"Firecrawl extraction error: {str(e)}")
        logger.error(f"Firecrawl extraction failed: {str(e)}")
    
    return {
        "firecrawl_data": firecrawl_data,
        "errors": errors,
        "performance_metrics": {},
        "critical_error": False
    }

# Simplified process_user_input without PubMed ESpell
@timeit
def process_user_input(state: State) -> dict:
    logger.info(f"Processing user input: {state['user_input_topic']}")
    topic = state["user_input_topic"]
    # Simply use the topic as is, without PubMed correction
    corrected_topic = topic.title()
    logger.info(f"Using topic: '{corrected_topic}'")
    return {
        "user_input_topic": corrected_topic,
        "errors": [],
        "performance_metrics": {},
        "critical_error": False
    }

# Removed search_pubmed and fetch_article_details functions

@timeit
def search_perplexity(state: State) -> dict:
    logger.info(f"Searching Perplexity for: {state['user_input_topic']}")
    perplexity_url = "https://api.perplexity.ai/chat/completions"
    headers = {"Authorization": f"Bearer {PERPLEXITY_API_KEY}", "Content-Type": "application/json"}
    data = {
        "model": "sonar-reasoning",
        "messages": [
            {
                "role": "system",
                "content": """You are a medical research assistant with access to a vast array of medical literature.
                  Your task is to retrieve the most recent and credible information from authentic sources such as 
                  peer-reviewed medical journals, official health organization reports, and reputable medical institutions.
                    Provide detailed and accurate data that can be used to generate informative medical articles."""
            },
            {
                "role": "user",
                "content": f"""Retrieve the most recent research data on {state['user_input_topic']} based on the description: '{state['user_input_description']}' 
                from specified authentic sources. Never make up your own links.
                Start with a brief summary of the current state of research on this topic, tailored to the interests and comprehension level of the target_audience: '{state['target_audience']}', 
                followed by detailed information including key findings, methodologies, relevant statistics, and citations or links to the original sources. 
                Present the information in a structured format, such as bullet points or subsections, to facilitate easy integration into an article. 
                At the end of your response, list the sources you used, formatted as: Title (Author(s), Publication Date). Link: [URL]"""
            }
        ],
        "max_tokens": 3500,
    }
    errors = []
    critical_error = False
    try:
        response = requests.post(perplexity_url, headers=headers, json=data)
        response.raise_for_status()
        perplexity_data = response.json()["choices"][0]["message"]["content"].split("\n")
        logger.info(f"Retrieved {len(perplexity_data)} lines from Perplexity")
    except requests.exceptions.HTTPError as e:
        if e.response.status_code in [401, 403]:
            errors.append("Critical error: Invalid Perplexity API key")
            critical_error = True
            logger.error("Critical error: Invalid Perplexity API key")
        else:
            errors.append(f"Perplexity error: {str(e)}")
            logger.error(f"Perplexity search failed: {str(e)}")
        perplexity_data = []
    except requests.RequestException as e:
        errors.append(f"Perplexity error: {str(e)}")
        perplexity_data = []
        logger.error(f"Perplexity search failed: {str(e)}")
    return {
        "perplexity_data": perplexity_data,
        "errors": errors,
        "performance_metrics": {},
        "critical_error": critical_error
    }


@timeit
def validate_content(state: State) -> dict:
    if state["critical_error"]:
        logger.error("Critical error detected, skipping content validation")
        return {
            "errors": ["Critical error occurred, validation skipped"],
            "performance_metrics": {},
            "critical_error": True
        }
    logger.info("Validating content")
    if not state["generated_content"]:
        logger.warning("No content to validate")
        return {
            "errors": ["No content available"],
            "performance_metrics": {},
            "critical_error": True
        }
    current_date = datetime.date.today()
    validation_prompt = f"""
    You are a medical content validator tasked with reviewing a medical article for general quality.
    Your goal is to determine if the article is suitable for use, with a focus on being reasonably lenient while ensuring basic standards are met.
    Below is the article to validate:

    Article Content:
    {state['generated_content']}

    Validate the article based on the following criteria, but apply these criteria with flexibility to avoid being overly strict:

    1. **General Accuracy**: Check if the content is broadly accurate and consistent with common medical knowledge on the topic "{state['user_input_topic']}".
    Allow for minor inaccuracies or generalizations as long as they do not fundamentally misrepresent the topic or pose a risk of harm.

    2. **AMA Citation Format**: Verify that citations are present and generally follow the AMA format (e.g., author names, year, journal, DOI if available).
    Be lenient with minor formatting issues as long as the citations are recognizable and provide enough information to locate the source.

    3. **Appropriateness for Audience**: Ensure the content is reasonably suitable for the target_audience, which is "{state['target_audience']}".
    For a doctor audience, the tone should be professional and include some technical terminology, but slight variations in tone are acceptable.

    Validation Guidelines:
    - Mark the article as "Valid" if it meets the above criteria in a general sense, even if there are minor issues.
    - Mark the article as "Invalid" only if there are significant issues (e.g., major factual errors, no citations, completely inappropriate tone).
    - Provide a list of specific issues (if any) to explain why the article is Invalid, or an empty list if Valid.

    Return a JSON object with the following structure:
    {{
        "status": "Valid" or "Invalid",
        "issues": ["list of specific issues or empty list if Valid"]
    }}
    """
    errors = []
    critical_error = False
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": validation_prompt}],
            max_tokens=1024,
            response_format={"type": "json_object"}
        )
        validation_data = json.loads(response.choices[0].message.content)
        if validation_data.get("status") != "Valid":
            errors.extend(validation_data.get("issues", []))
            logger.warning(f"Validation failed: {validation_data.get('issues', [])}")
        else:
            logger.info("Content validated successfully")
    except Exception as e:
        errors.append(f"Validation error: {str(e)}")
        critical_error = True
        logger.error(f"Validation failed: {str(e)}")
    return {
        "errors": errors,
        "performance_metrics": {},
        "critical_error": critical_error
    }

def check_data_availability(state: State) -> dict:
    logger.info("Checking data availability")
    if state["critical_error"]:
        logger.warning("Critical error detected, but proceeding with available data")
    return {
        "errors": [],
        "performance_metrics": {},
        "critical_error": state["critical_error"]
    }

def route_after_pubmed(state: State) -> str:
    if state["critical_error"]:
        logger.error("Critical error detected after search_pubmed, proceeding to check_data_availability")
        state["article_data"] = []
        return "check_data_availability"
    if state["pmids"]:
        return "fetch_article_details"
    return "check_data_availability"

def route_after_check_data(state: State) -> str:
    return "generate_content"

# Celery task for fetching and storing topics
@celery_app.task(name="main.fetch_and_store_topics_task")
def fetch_and_store_topics_task():
    try:
        aus_tz = pytz.timezone("Australia/Sydney")
        current_time = datetime.datetime.now(aus_tz)
        logger.info(f"Fetching topics at {current_time.strftime('%Y-%m-%d %H:%M:%S %Z')} in Australian time")

        completion = openai_client.chat.completions.create(
            model="gpt-4o-search-preview",
            messages=[
                {
                    "role": "user",
                    "content": """Search the web and current online discussions to identify the 5 most talked-about medical topics today.
                    Provide only the list of topics, ranked by popularity, 
                    that are trending and suitable for creating articles for medical students. 
                    Just return the topic names, don't say any other thing
                    also don't add numbering"""
                }
            ]
        )
        topics = completion.choices[0].message.content.strip().split("\n")
        topics = [topic.strip() for topic in topics if topic.strip()]
        
        logger.info(f"Fetched topics: {topics}")
        collection.delete_many({})
        collection.insert_one({"topics": topics, "timestamp": current_time.isoformat()})
        logger.info(f"Stored {len(topics)} trending topics in MongoDB with timestamp {current_time.isoformat()}")
        return {"topics": topics}
    except Exception as e:
        logger.error(f"Error in task: {str(e)}")
        raise

# Workflow setup
workflow = StateGraph(State)
workflow.add_node("process_user_input", process_user_input)
workflow.add_node("search_perplexity", search_perplexity)
workflow.add_node("extract_firecrawl_content", extract_firecrawl_content)
workflow.add_node("check_data_availability", check_data_availability)
workflow.add_node("generate_content", generate_content)
workflow.add_node("validate_content", validate_content)

workflow.set_entry_point("process_user_input")
workflow.add_edge("process_user_input", "search_perplexity")
workflow.add_edge("process_user_input", "extract_firecrawl_content")
workflow.add_edge("search_perplexity", "check_data_availability")
workflow.add_edge("extract_firecrawl_content", "check_data_availability")
workflow.add_conditional_edges(
    "check_data_availability",
    route_after_check_data,
    {
        "generate_content": "generate_content"
    }
)
workflow.add_edge("generate_content", "validate_content")
workflow.add_edge("validate_content", END)

app = workflow.compile()

# FastAPI app
fastapi_app = FastAPI()

# CORS Middleware
fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class TopicsResponse(BaseModel):
    topics: List[str]

class UrlRequest(BaseModel):  # New model
    url: str

# FastAPI endpoints
@fastapi_app.post("/generate-article")
async def generate_article(request: dict):
    initial_state = {
        "user_input_topic": request.get("user_input_topic", ""),
        "user_input_description": request.get("user_input_description", ""),
        "article_length": request.get("article_length", "Short"),
        "target_audience": request.get("target_audience", "general"),
        "perplexity_data": [],
        "firecrawl_data": [],  # Added to initial state
        "generated_content": "",
        "errors": [],
        "performance_metrics": {},
        "critical_error": False
    }
    logger.info("Starting workflow execution")
    start_time = time.time()
    try:
        final_state = app.invoke(initial_state)
        end_time = time.time()
        total_time = end_time - start_time
        final_state["performance_metrics"]["total_execution_time"] = total_time
        logger.info(f"Total execution time: {total_time:.4f} seconds")
        if final_state["errors"]:
            raise HTTPException(status_code=500, detail={"detail": f"Errors occurred: {final_state['errors']}", "status": 500})
        return {
            "generated_content": final_state["generated_content"],
            "performance_metrics": final_state["performance_metrics"],
            "errors": final_state["errors"]
        }
    except Exception as e:
        end_time = time.time()
        total_time = end_time - start_time
        logger.error(f"Workflow failed: {str(e)} - Total time: {total_time:.4f} seconds")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail={"detail": f"Internal server error: {str(e)}", "status": 500})

@fastapi_app.get("/get-topics/", response_model=TopicsResponse)
async def get_topics():
    try:
        document = collection.find_one()
        if document and 'topics' in document:
            topics = document['topics']
            logger.info(f"Retrieved {len(topics)} topics from MongoDB")
            return {"topics": topics}
        else:
            logger.warning("No topics found in MongoDB")
            raise HTTPException(status_code=404, detail="No topics found in the database.")
    except Exception as e:
        logger.error(f"Error fetching topics from MongoDB: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching topics from MongoDB: {str(e)}")

# --- New endpoint for Firecrawl extraction ---
@fastapi_app.post("/extract")
async def extract_content(request: UrlRequest):
    try:
        scraped = firecrawl.scrape_url(request.url)
        content = scraped.get('markdown', 'No content found')
        return {
            "url": request.url,
            "content": content
        }
    except Exception as e:
        logger.error(f"Extraction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Content extraction failed: {str(e)}")

@fastapi_app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    config = uvicorn.Config(fastapi_app, host="0.0.0.0", port=8000, loop="asyncio")
    server = uvicorn.Server(config)
    server.run()
