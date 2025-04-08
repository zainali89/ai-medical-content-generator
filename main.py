# main.py
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
from fastapi.responses import JSONResponse
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from pydantic import BaseModel
import nest_asyncio
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from firecrawl import FirecrawlApp
import pytz
from tasks import fetch_and_store_topics  # Import from tasks.py

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()]
)

load_dotenv()

# API Keys and MongoDB URI from environment variables
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
PERPLEXITY_API_KEY = os.environ.get("PERPLEXITY_API_KEY")
FIRECRAWL_API_KEY = os.environ.get("FIRECRAWL_API_KEY")
MONGODB_URI = os.environ.get("MONGODB_URI")

logger.info(f"OPENAI_API_KEY: {'Set' if OPENAI_API_KEY else 'Not set'}")
logger.info(f"PERPLEXITY_API_KEY: {'Set' if PERPLEXITY_API_KEY else 'Not set'}")
logger.info(f"FIRECRAWL_API_KEY: {'Set' if FIRECRAWL_API_KEY else 'Not set'}")
logger.info(f"MONGODB_URI: {'Set' if MONGODB_URI else 'Not set'}")

if not all([OPENAI_API_KEY, PERPLEXITY_API_KEY, MONGODB_URI, FIRECRAWL_API_KEY]):
    raise ValueError("One or more required keys (API keys or MongoDB URI) are missing.")

# Initialize OpenAI client
openai_client = OpenAI(api_key=OPENAI_API_KEY)
logger.info("OpenAI client initialized.")

# Initialize Firecrawl
firecrawl = FirecrawlApp(api_key=FIRECRAWL_API_KEY)
logger.info("Firecrawl client initialized.")

# Connect to MongoDB
try:
    client_mongo = MongoClient(MONGODB_URI, server_api=ServerApi('1'))
    client_mongo.admin.command('ping')
    logger.info("Successfully connected to MongoDB!")
except Exception as e:
    logger.error(f"Error connecting to MongoDB: {str(e)}")
    raise ValueError(f"Error connecting六 to MongoDB: {str(e)}")

db = client_mongo['TopMedicalArticles']
collection = db['topics']

# FastAPI app initialization
fastapi_app = FastAPI()

# CORS Middleware
fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

# State definition for LangGraph
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

# LangGraph node functions
@timeit
def extract_firecrawl_content(state: State) -> dict:
    if not state["reference_urls"]:
        logger.info("No reference URLs provided, skipping Firecrawl extraction")
        return {
            "firecrawl_data": [],
            "errors": [],
            "performance_metrics": {},
            "critical_error": False
        }
    
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
    
    return {
        "firecrawl_data": firecrawl_data,
        "errors": errors,
        "performance_metrics": {},
        "critical_error": False
    }

@timeit
def process_user_input(state: State) -> dict:
    logger.info(f"Processing user input: {state['user_input_topic']}")
    topic = state["user_input_topic"]
    corrected_topic = topic.title()
    logger.info(f"Using topic: '{corrected_topic}'")
    return {
        "user_input_topic": corrected_topic,
        "errors": [],
        "performance_metrics": {},
        "critical_error": False
    }

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
def process_docs(state: State) -> dict:
    if not state["docs_files"]:
        logger.info("No document files provided, skipping docs processing")
        return {
            "docs_data": [],
            "errors": [],
            "performance_metrics": {},
            "critical_error": False
        }
    
    logger.info(f"Processing {len(state['docs_files'])} document files")
    return {
        "docs_data": [],
        "errors": [],
        "performance_metrics": {},
        "critical_error": False
    }

@timeit
def process_youtube_links(state: State) -> dict:
    if not state["youtube_links"]:
        logger.info("No YouTube links provided, skipping YouTube processing")
        return {
            "youtube_data": [],
            "errors": [],
            "performance_metrics": {},
            "critical_error": False
        }
    
    logger.info(f"Processing {len(state['youtube_links'])} YouTube links")
    return {
        "youtube_data": [],
        "errors": [],
        "performance_metrics": {},
        "critical_error": False
    }

@timeit
def generate_content(state: State) -> dict:
    if state["critical_error"]:
        logger.error("Critical error detected, skipping content generation")
        return {
            "generated_content": "",
            "errors": ["Critical error occurred, content generation skipped"],
            "performance_metrics": {},
            "critical_error": True
        }
    logger.info(f"Generating content for: {state['user_input_topic']}")
    
    if not state["perplexity_data"] and not state["firecrawl_data"] and not state["docs_data"] and not state["youtube_data"]:
        logger.warning("No data available for content generation")
        return {
            "generated_content": "",
            "errors": ["No data available"],
            "performance_metrics": {},
            "critical_error": True
        }
    current_date = datetime.date.today()
    
    perplexity_data = "\n".join([f"Perplexity: {item}" for item in state["perplexity_data"] if item])
    firecrawl_data = "\n".join([f"Firecrawl: {item}" for item in state["firecrawl_data"] if item])
    docs_data = "\n".join([f"Document: {item}" for item in state["docs_data"] if item])
    youtube_data = "\n".join([f"YouTube: {item}" for item in state["youtube_data"] if item])
    
    length_mapping = {"Short": 500, "Medium": 1000, "Long": 1500}
    length_words = length_mapping.get(state["article_length"], 500)
    prompt = f"""
    Write a referenced, fact-checked, and neutral article about {state['user_input_topic']} specifically tailored for {state['target_audience']}. Use Australian English (e.g., 'organise', 'centre') and base all factual claims STRICTLY on the provided reference data from peer-reviewed or credible sources.
    
    IMPORTANT: DO NOT HALLUCINATE OR INVENT ANY INFORMATION. If the provided reference data doesn't cover a particular aspect of the topic, explicitly state that information is limited rather than making up facts. Only include information that is directly supported by the reference data provided below.
    
    Adjust language and detail for the audience:
    - Medical Professionals (Doctors): Employ precise medical terminology and provide comprehensive, detailed analysis.
    - Students: Utilize technical medical vocabulary and deliver thorough, educational analysis.
    - General Public: Use simple, everyday words, clarify any complex terms, and highlight useful, easy-to-apply information.
    - Patients: Use clear, straightforward language, explain medical terms simply, and emphasize practical, health-related advice
    
    Use the reference data below to support claims, ensuring the article is engaging and accessible. The data includes:
    - **Perplexity**: Text with a 'Sources' section (e.g., 'Title (Author(s), Date). Link: [URL]'). Use URLs exactly as provided.
    - **Firecrawl**: Scraped content prefixed with source URL (e.g., 'Firecrawl content from [URL]: [content]'). Use the URL provided in the prefix.
    - **Documents**: Content extracted from document files, prefixed with 'Document:'.
    - **YouTube**: Content transcribed from YouTube videos, prefixed with 'YouTube:'.
    
    Rules for references and content:
    - Extract Perplexity URLs from lines like 'Link: [URL]' and use them unchanged.
    - Extract Firecrawl URLs from the prefix 'Firecrawl content from [URL]' and use them unchanged.
    - For Document content, cite as "From document analysis" if no specific citation is available.
    - For YouTube content, cite as "From [YouTube video title]" if available.
    - Include a reference only if it has a valid URL from the data. If no URL exists, omit it—do NOT invent links (e.g., no '.example.com').
    - Verify all facts against the data and correct errors.
    - Always make sure the links are clickable.
    - Only include the links in the references.
    - If you're uncertain about any information, indicate this clearly rather than guessing.
    - For any statistical claims, medical recommendations, or specific treatments, cite the exact source from the reference data.
    
    Keep the tone objective and evidence-based, current as of {current_date}, and note missing data if applicable. End with a reference list in this format:
    - [Number]. Title (Author(s), Date). Link: [URL]
    
    - User Description: {state['user_input_description']}
    - Length: ~{length_words} words
    - Reference Data (Perplexity): 
    {perplexity_data if perplexity_data else 'No Perplexity data available'}
    - Reference Data (Firecrawl): 
    {firecrawl_data if firecrawl_data else 'No Firecrawl data available'}
    - Reference Data (Documents): 
    {docs_data if docs_data else 'No Document data available'}
    - Reference Data (YouTube): 
    {youtube_data if youtube_data else 'No YouTube data available'}
    """
    errors = []
    critical_error = False
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a medical content writer who ONLY uses provided reference data. Never invent or hallucinate information. If the reference data doesn't cover something, explicitly state that information is limited."},
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
    return {
        "generated_content": content,
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

def route_after_check_data(state: State) -> str:
    return "generate_content"

# LangGraph workflow setup
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

# Pydantic models
class TopicsResponse(BaseModel):
    topics: List[str]

class UrlRequest(BaseModel):
    url: str

# FastAPI endpoints
@fastapi_app.post("/generate-article")
async def generate_article(request: dict):
    initial_state = {
        "user_input_topic": request.get("user_input_topic", ""),
        "user_input_description": request.get("user_input_description", ""),
        "article_length": request.get("article_length", "Short"),
        "target_audience": request.get("target_audience", "general"),
        "reference_urls": request.get("reference_urls", []),
        "docs_files": request.get("docs_files", []),
        "youtube_links": request.get("youtube_links", []),
        "perplexity_data": [],
        "firecrawl_data": [],
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

@fastapi_app.get("/get-topics", response_model=TopicsResponse)
async def get_topics():
    try:
        logger.info("Attempting to fetch topics from MongoDB")
        document = collection.find_one({}, sort=[("timestamp", -1)])
        logger.info(f"Retrieved document from MongoDB: {document}")

        if not document:
            logger.warning("No documents found in MongoDB - returning empty list")
            return JSONResponse(content={"topics": []}, headers={"Cache-Control": "no-store"})

        if 'topics' not in document:
            logger.error("Document found but 'topics' field is missing - returning empty list")
            return JSONResponse(content={"topics": []}, headers={"Cache-Control": "no-store"})

        if 'timestamp' not in document:
            logger.warning("Document found but 'timestamp' field is missing")

        topics = document['topics']
        timestamp = document.get('timestamp', 'unknown')

        if not isinstance(topics, list):
            logger.error(f"Topics field is not a list: {topics} - returning empty list")
            return JSONResponse(content={"topics": []}, headers={"Cache-Control": "no-store"})

        logger.info(f"Successfully retrieved {len(topics)} topics from MongoDB with timestamp {timestamp}")
        return JSONResponse(content={"topics": topics}, headers={"Cache-Control": "no-store"})
    except Exception as e:
        logger.error(f"Error fetching topics from MongoDB: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching topics from MongoDB: {str(e)}")

@fastapi_app.post("/fetch-topics")
async def trigger_fetch_topics():
    try:
        result = await fetch_and_store_topics()
        return {"status": "success", "topics": result["topics"]}
    except Exception as e:
        logger.error(f"Failed to fetch topics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch topics: {str(e)}")

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
