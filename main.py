import logging
import os
from dotenv import load_dotenv
import json
import requests
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
from firecrawl import FirecrawlApp
import PyPDF2
from docx import Document
import re
import io
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter

# Apply nest_asyncio
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

# API Keys and MongoDB URI
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

if not all([OPENAI_API_KEY, PERPLEXITY_API_KEY, MONGODB_URI, FIRECRAWL_API_KEY]):
    raise ValueError("One or more required keys are missing.")

# Initialize clients
openai_client = OpenAI(api_key=OPENAI_API_KEY)
logger.info("OpenAI client initialized.")
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

# Celery setup
try:
    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    celery_app = Celery("tasks", broker=redis_url, backend=redis_url)
    logger.info(f"Celery app initialized with broker: {redis_url}")
except Exception as e:
    logger.error(f"Error initializing Celery: {str(e)}")
    raise

celery_app.conf.timezone = "Australia/Sydney"
celery_app.conf.beat_schedule = {
    "update-medical-topics-daily": {
        "task": "main.fetch_and_store_topics_task",
        "schedule": crontab(hour=0, minute=0),
        "options": {"timezone": "Australia/Sydney"}
    },
}
celery_app.conf.update(result_expires=3600)
logger.info("Celery schedule configured for 12 AM Australia/Sydney time.")

@fastapi_app.on_event("startup")
async def startup_event():
    try:
        fetch_and_store_topics_task.delay()
        logger.info("Triggered initial topic fetch at startup")
    except Exception as e:
        logger.error(f"Failed to trigger initial topic fetch: {str(e)}")

# Timeit decorator
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

# State definition
class State(TypedDict):
    user_input_topic: str
    user_input_description: str
    article_length: str
    target_audience: str
    perplexity_data: List[str]
    firecrawl_data: List[str]
    reference_urls: List[str]
    docs_files: List[str]
    youtube_links: List[str]
    docs_data: List[str]
    youtube_data: List[str]
    generated_content: str
    errors: Annotated[List[str], operator.add]
    performance_metrics: Annotated[Dict[str, float], lambda x, y: {**x, **y}]
    critical_error: Annotated[bool, lambda x, y: x or y]

@timeit
def extract_firecrawl_content(state: State) -> dict:
    if not state["reference_urls"]:
        logger.info("No reference URLs provided, skipping Firecrawl extraction")
        return {"firecrawl_data": [], "errors": [], "performance_metrics": {}, "critical_error": False}
    
    logger.info(f"Extracting Firecrawl content for {len(state['reference_urls'])} URLs")
    errors = []
    firecrawl_data = []
    
    for url in state["reference_urls"]:
        try:
            scraped = firecrawl.scrape_url(url)
            content = scraped.get('markdown', 'No content found')
            firecrawl_data.append(f"Firecrawl content from {url}: {content}")
            logger.info(f"Successfully extracted Firecrawl content from {url}")
        except Exception as e:
            errors.append(f"Firecrawl extraction error for {url}: {str(e)}")
            logger.error(f"Firecrawl extraction failed for {url}: {str(e)}")
    
    return {"firecrawl_data": firecrawl_data, "errors": errors, "performance_metrics": {}, "critical_error": False}

@timeit
def process_user_input(state: State) -> dict:
    logger.info(f"Processing user input: {state['user_input_topic']}")
    topic = state["user_input_topic"]
    corrected_topic = topic.title()
    logger.info(f"Using topic: '{corrected_topic}'")
    return {"user_input_topic": corrected_topic, "errors": [], "performance_metrics": {}, "critical_error": False}

@timeit
def search_perplexity(state: State) -> dict:
    logger.info(f"Searching Perplexity for: {state['user_input_topic']}")
    perplexity_url = "https://api.perplexity.ai/chat/completions"
    headers = {"Authorization": f"Bearer {PERPLEXITY_API_KEY}", "Content-Type": "application/json"}
    data = {
        "model": "sonar-reasoning",
        "messages": [
            {"role": "system", "content": "You are a medical research assistant with access to credible sources. Provide detailed, accurate data from peer-reviewed journals and reputable institutions."},
            {"role": "user", "content": f"Retrieve recent research on {state['user_input_topic']} based on '{state['user_input_description']}' for {state['target_audience']}. Include summary, findings, and sources as: Title (Author(s), Date). Link: [URL]"}
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
        else:
            errors.append(f"Perplexity error: {str(e)}")
        perplexity_data = []
        logger.error(f"Perplexity search failed: {str(e)}")
    except requests.RequestException as e:
        errors.append(f"Perplexity error: {str(e)}")
        perplexity_data = []
        logger.error(f"Perplexity search failed: {str(e)}")
    return {"perplexity_data": perplexity_data, "errors": errors, "performance_metrics": {}, "critical_error": critical_error}

@timeit
def process_docs(state: State) -> dict:
    if not state["docs_files"]:
        logger.info("No document files provided, skipping docs processing")
        return {"docs_data": [], "errors": [], "performance_metrics": {}, "critical_error": False}
    
    logger.info(f"Processing {len(state['docs_files'])} document files")
    docs_data = []
    errors = []
    
    for doc_url in state["docs_files"]:
        try:
            if doc_url.lower().endswith('.pdf'):
                logger.info(f"Fetching PDF from: {doc_url}")
                response = requests.get(doc_url, timeout=10)
                response.raise_for_status()
                pdf_file = io.BytesIO(response.content)
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                text = ""
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text() + "\n"
                text = re.sub(r'\s+', ' ', text.strip())
                if text:
                    docs_data.append(f"Content from PDF '{doc_url}': {text}")
                    logger.info(f"Successfully extracted text from PDF: {doc_url}")
                else:
                    errors.append(f"No text extracted from PDF: {doc_url}")
                    logger.warning(f"No text extracted from PDF: {doc_url}")
            
            elif doc_url.lower().endswith('.docx'):
                logger.info(f"Fetching DOCX from: {doc_url}")
                response = requests.get(doc_url, timeout=10)
                response.raise_for_status()
                docx_file = io.BytesIO(response.content)
                doc = Document(docx_file)
                text = ""
                for para in doc.paragraphs:
                    if para.text.strip():
                        text += para.text + "\n"
                text = text.strip()
                if text:
                    docs_data.append(f"Content from DOCX '{doc_url}': {text}")
                    logger.info(f"Successfully extracted text from DOCX: {doc_url}")
                else:
                    errors.append(f"No text extracted from DOCX: {doc_url}")
                    logger.warning(f"No text extracted from DOCX: {doc_url}")
            
            else:
                errors.append(f"Unsupported document type: {doc_url}")
                logger.warning(f"Unsupported document type: {doc_url}")
        
        except requests.RequestException as e:
            errors.append(f"Error fetching document {doc_url}: {str(e)}")
            logger.error(f"Error fetching document {doc_url}: {str(e)}")
        except Exception as e:
            errors.append(f"Error processing document {doc_url}: {str(e)}")
            logger.error(f"Error processing document {doc_url}: {str(e)}")
    
    return {"docs_data": docs_data, "errors": errors, "performance_metrics": {}, "critical_error": False}

@timeit
def process_youtube_links(state: State) -> dict:
    if not state["youtube_links"]:
        logger.info("No YouTube links provided, skipping YouTube processing")
        return {"youtube_data": [], "errors": [], "performance_metrics": {}, "critical_error": False}
    
    logger.info(f"Processing {len(state['youtube_links'])} YouTube links")
    youtube_data = []
    errors = []
    
    for yt_link in state["youtube_links"]:
        try:
            video_id = extract_youtube_video_id(yt_link)
            if not video_id:
                errors.append(f"Invalid YouTube URL: {yt_link}")
                logger.warning(f"Invalid YouTube URL: {yt_link}")
                continue
            
            logger.info(f"Fetching transcript for YouTube video ID: {video_id}")
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
            formatter = TextFormatter()
            transcript_text = formatter.format_transcript(transcript_list)
            video_title = f"YouTube video (ID: {video_id})"
            youtube_data.append(f"Transcript from {video_title}: {transcript_text}")
            logger.info(f"Successfully extracted transcript from {yt_link}")
            
        except Exception as e:
            errors.append(f"Error processing YouTube link {yt_link}: {str(e)}")
            logger.error(f"Error processing YouTube link {yt_link}: {str(e)}")
    
    return {"youtube_data": youtube_data, "errors": errors, "performance_metrics": {}, "critical_error": False}

def extract_youtube_video_id(url):
    youtube_regex = (
        r'(https?://)?(www\.)?'
        r'(youtube|youtu|youtube-nocookie)\.(com|be)/'
        r'(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})'
    )
    match = re.match(youtube_regex, url)
    if match:
        return match.group(6)
    youtu_be_regex = r'(https?://)?(www\.)?youtu\.be/([^&=%\?]{11})'
    match = re.match(youtu_be_regex, url)
    if match:
        return match.group(3)
    return None

@timeit
def generate_content(state: State) -> dict:
    if state["critical_error"]:
        logger.error("Critical error detected, skipping content generation")
        return {"generated_content": "", "errors": ["Critical error occurred"], "performance_metrics": {}, "critical_error": True}
    
    if not any([state["perplexity_data"], state["firecrawl_data"], state["docs_data"], state["youtube_data"]]):
        logger.warning("No data available for content generation")
        return {"generated_content": "", "errors": ["No data available"], "performance_metrics": {}, "critical_error": True}
    
    logger.info(f"Generating content for: {state['user_input_topic']}")
    current_date = datetime.date.today()
    perplexity_data = "\n".join([f"Perplexity: {item}" for item in state["perplexity_data"] if item])
    firecrawl_data = "\n".join([f"Firecrawl: {item}" for item in state["firecrawl_data"] if item])
    docs_data = "\n".join([f"Document: {item}" for item in state["docs_data"] if item])
    youtube_data = "\n".join([f"YouTube: {item}" for item in state["youtube_data"] if item])
    
    length_mapping = {"Short": 500, "Medium": 1000, "Long": 1500}
    length_words = length_mapping.get(state["article_length"], 500)
    prompt = f"""
    Write a referenced, fact-checked, neutral article about {state['user_input_topic']} for {state['target_audience']} using Australian English. Base all claims STRICTLY on provided data. If data is missing, state it explicitly. Adjust language:
    - Doctors: Precise medical terms, detailed analysis.
    - Students: Technical vocab, educational focus.
    - General Public: Simple words, clear explanations.
    - Patients: Clear language, practical advice.
    Use: Perplexity (Sources: Title (Author(s), Date). Link: [URL]), Firecrawl (prefix: Firecrawl content from [URL]), Documents (prefix: Document:), YouTube (prefix: YouTube:).
    Rules: Use exact URLs, cite "From document analysis" or "From [YouTube title]" if no URL, verify facts, clickable links only from data, note uncertainty. End with references: [Number]. Title (Author(s), Date). Link: [URL].
    - Description: {state['user_input_description']}
    - Length: ~{length_words} words
    - Data: 
      Perplexity: {perplexity_data or 'None'}
      Firecrawl: {firecrawl_data or 'None'}
      Documents: {docs_data or 'None'}
      YouTube: {youtube_data or 'None'}
    """
    errors = []
    critical_error = False
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Use only provided data. Never invent info. State limits if data is incomplete."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=8000,
            temperature=0.3
        )
        content = response.choices[0].message.content
        logger.info("Content generated successfully")
    except Exception as e:
        errors.append(f"OpenAI error: {str(e)}")
        content = ""
        critical_error = True
        logger.error(f"Content generation failed: {str(e)}")
    return {"generated_content": content, "errors": errors, "performance_metrics": {}, "critical_error": critical_error}

@timeit
def validate_content(state: State) -> dict:
    if state["critical_error"]:
        logger.error("Critical error detected, skipping validation")
        return {"errors": ["Critical error occurred"], "performance_metrics": {}, "critical_error": True}
    
    if not state["generated_content"]:
        logger.warning("No content to validate")
        return {"errors": ["No content available"], "performance_metrics": {}, "critical_error": True}
    
    logger.info("Validating content")
    validation_prompt = f"""
    Validate this article for quality:
    {state['generated_content']}
    Criteria (be lenient):
    1. Accuracy: Broadly consistent with medical knowledge on "{state['user_input_topic']}".
    2. Citations: Present, roughly AMA format.
    3. Audience: Suitable for "{state['target_audience']}".
    Return JSON: {{"status": "Valid" or "Invalid", "issues": ["list"]}}
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
    return {"errors": errors, "performance_metrics": {}, "critical_error": critical_error}

def check_data_availability(state: State) -> dict:
    logger.info("Checking data availability")
    return {"errors": [], "performance_metrics": {}, "critical_error": state["critical_error"]}

def route_after_check_data(state: State) -> str:
    return "generate_content"

@celery_app.task(name="main.fetch_and_store_topics_task")
def fetch_and_store_topics_task():
    try:
        aus_tz = pytz.timezone("Australia/Sydney")
        current_time = datetime.datetime.now(aus_tz)
        logger.info(f"Fetching topics at {current_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        completion = openai_client.chat.completions.create(
            model="gpt-4o-search-preview",
            messages=[{"role": "user", "content": "List 5 trending medical topics for students, no numbering."}]
        )
        topics = [t.strip() for t in completion.choices[0].message.content.strip().split("\n") if t.strip()]
        logger.info(f"Fetched topics: {topics}")
        collection.delete_many({})
        collection.insert_one({"topics": topics, "timestamp": current_time.isoformat()})
        logger.info(f"Stored {len(topics)} topics")
        return {"topics": topics}
    except Exception as e:
        logger.error(f"Error in task: {str(e)}")
        raise

# Workflow setup
workflow = StateGraph(State)
workflow.add_node("process_user_input", process_user_input)
workflow.add_node("search_perplexity", search_perplexity)
workflow.add_node("extract_firecrawl_content", extract_firecrawl_content)
workflow.add_node("process_docs", process_docs)
workflow.add_node("process_youtube_links", process_youtube_links)
workflow.add_node("check_data_availability", check_data_availability)
workflow.add_node("generate_content", generate_content)
workflow.add_node("validate_content", validate_content)

workflow.set_entry_point("process_user_input")
workflow.add_edge("process_user_input", "search_perplexity")
workflow.add_edge("process_user_input", "extract_firecrawl_content")
workflow.add_edge("process_user_input", "process_docs")
workflow.add_edge("process_user_input", "process_youtube_links")
workflow.add_edge("search_perplexity", "check_data_availability")
workflow.add_edge("extract_firecrawl_content", "check_data_availability")
workflow.add_edge("process_docs", "check_data_availability")
workflow.add_edge("process_youtube_links", "check_data_availability")
workflow.add_conditional_edges("check_data_availability", route_after_check_data, {"generate_content": "generate_content"})
workflow.add_edge("generate_content", "validate_content")
workflow.add_edge("validate_content", END)

app = workflow.compile()

# Pydantic models
class TopicsResponse(BaseModel):
    topics: List[str]

class UrlRequest(BaseModel):
    url: str

# FastAPI endpoints
@fastapi_app.post("/generate-article")
async def generate_article(request: dict):
    try:
        payload = request.get("data", {}).get("payload", {})
        if not payload:
            raise HTTPException(status_code=400, detail="Invalid request format: 'payload' missing")
        
        initial_state = {
            "user_input_topic": payload.get("user_input_topic", ""),
            "user_input_description": payload.get("user_input_description", ""),
            "article_length": payload.get("article_length", "Short"),
            "target_audience": payload.get("target_audience", "General Public"),
            "youtube_links": [link.replace(r"\/", "/") for link in payload.get("youtube_links", [])],  # Unescape URLs
            "reference_urls": payload.get("reference_urls", []),
            "docs_files": payload.get("docs_files", []),
            "perplexity_data": [],
            "firecrawl_data": [],
            "docs_data": [],
            "youtube_data": [],
            "generated_content": "",
            "errors": [],
            "performance_metrics": {},
            "critical_error": False
        }
        logger.info(f"Starting workflow with payload: {payload}")
        start_time = time.time()
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
        total_time = end_time - start_time if 'start_time' in locals() else 0
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
        logger.warning("No topics found in MongoDB")
        raise HTTPException(status_code=404, detail="No topics found.")
    except Exception as e:
        logger.error(f"Error fetching topics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching topics: {str(e)}")

@fastapi_app.post("/extract")
async def extract_content(request: UrlRequest):
    try:
        scraped = firecrawl.scrape_url(request.url)
        content = scraped.get('markdown', 'No content found')
        return {"url": request.url, "content": content}
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
