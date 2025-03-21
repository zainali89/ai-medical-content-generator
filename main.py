import logging
import os
from dotenv import load_dotenv
import json
import requests
import xml.etree.ElementTree as ET
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

# API Keys and MongoDB URI from environment variables
PUBMED_API_KEY = os.environ.get("PUBMED_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
PERPLEXITY_API_KEY = os.environ.get("PERPLEXITY_API_KEY")
MONGODB_URI = os.environ.get(
    "MONGODB_URI",
    "mongodb+srv://syedbasitabbas10:FZg3aL0FbRYyxGdh@topmedicalarticles.pfo2g.mongodb.net/?retryWrites=true&w=majority&appName=TopMedicalArticles"
)

logger.info(f"PUBMED_API_KEY: {'Set' if PUBMED_API_KEY else 'Not set'}")
logger.info(f"OPENAI_API_KEY: {'Set' if OPENAI_API_KEY else 'Not set'}")
logger.info(f"PERPLEXITY_API_KEY: {'Set' if PERPLEXITY_API_KEY else 'Not set'}")
logger.info(f"MONGODB_URI: {'Set' if MONGODB_URI else 'Not set'}")

if not all([PUBMED_API_KEY, OPENAI_API_KEY, PERPLEXITY_API_KEY, MONGODB_URI]):
    raise ValueError("One or more required keys (API keys or MongoDB URI) are missing. Please check your .env file or environment variables.")

# Initialize OpenAI client
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Connect to MongoDB (no explicit tlsCAFile unless required)
try:
    client_mongo = MongoClient(
        MONGODB_URI,
        server_api=ServerApi('1'),
    )
    client_mongo.admin.command('ping')
    logger.info("Successfully connected to MongoDB!")
except Exception as e:
    logger.error(f"Error connecting to MongoDB: {str(e)}")
    raise ValueError(f"Error connecting to MongoDB: {str(e)}")

db = client_mongo['TopMedicalArticles']
collection = db['TrendingTopics']

# Celery setup (using Redis as the broker)
celery_app = Celery("tasks", broker="redis://localhost:6379/0", backend="redis://localhost:6379/0")

# Celery Beat scheduling configuration
celery_app.conf.timezone = "Australia/Sydney"
celery_app.conf.beat_schedule = {
    "update-medical-topics-daily": {
        "task": "app.fetch_and_store_topics_task",
        "schedule": crontab(hour=0, minute=0),
        "options": {"timezone": "Australia/Sydney"}
    },
}
celery_app.conf.update(
    result_expires=3600,
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

# State definition
class State(TypedDict):
    user_input_topic: str
    user_input_description: str
    article_length: str
    target_audience: str
    pmids: List[str]
    article_data: List[dict]
    perplexity_data: List[str]
    generated_content: str
    errors: Annotated[List[str], operator.add]
    performance_metrics: Annotated[Dict[str, float], lambda x, y: {**x, **y}]
    critical_error: Annotated[bool, lambda x, y: x or y]

# Workflow functions
@timeit
def process_user_input(state: State) -> dict:
    logger.info(f"Processing user input: {state['user_input_topic']}")
    topic = state["user_input_topic"]
    espell_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/espell.fcgi"
    espell_params = {"db": "pubmed", "term": topic.replace(" ", "+"), "api_key": PUBMED_API_KEY}
    errors = []
    try:
        response = requests.get(espell_url, params=espell_params)
        response.raise_for_status()
        root = ET.fromstring(response.text)
        corrected_query = root.find(".//CorrectedQuery")
        corrected_topic = corrected_query.text.title() if corrected_query is not None else topic
        logger.info(f"Corrected topic: '{topic}' -> '{corrected_topic}'")
    except (requests.RequestException, ET.ParseError) as e:
        errors.append(f"ESpell error: {str(e)}")
        corrected_topic = topic
        logger.warning(f"ESpell failed: {str(e)}")
    return {
        "user_input_topic": corrected_topic,
        "errors": errors,
        "performance_metrics": {},
        "critical_error": False
    }

@timeit
def search_pubmed(state: State) -> dict:
    logger.info(f"Searching PubMed for: {state['user_input_topic']}")
    search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": state["user_input_topic"].replace(" ", "+"),
        "retmode": "json",
        "retmax": 10,
        "api_key": PUBMED_API_KEY
    }
    errors = []
    critical_error = False
    try:
        response = requests.get(search_url, params=params)
        response.raise_for_status()
        pmids = response.json()["esearchresult"]["idlist"]
        logger.info(f"Found {len(pmids)} PMIDs")
    except requests.exceptions.HTTPError as e:
        if e.response.status_code in [401, 403]:
            errors.append("Critical error: Invalid PubMed API key")
            critical_error = True
            logger.error("Critical error: Invalid PubMed API key")
        else:
            errors.append(f"PubMed search error: {str(e)}")
            logger.error(f"PubMed search failed: {str(e)}")
        pmids = []
    except requests.RequestException as e:
        errors.append(f"PubMed search error: {str(e)}")
        pmids = []
        logger.error(f"PubMed search failed: {str(e)}")
    return {
        "pmids": pmids,
        "errors": errors,
        "performance_metrics": {},
        "critical_error": critical_error
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
        "max_tokens": 1500,
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
def fetch_article_details(state: State) -> dict:
    logger.info(f"Fetching details for PMIDs: {state['pmids']}")
    if not state["pmids"]:
        logger.warning("No PMIDs available")
        return {"article_data": [], "errors": ["No PMIDs available"], "performance_metrics": {}, "critical_error": False}
    fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {"db": "pubmed", "id": ",".join(state["pmids"]), "retmode": "xml", "rettype": "abstract"}
    errors = []
    critical_error = False
    try:
        response = requests.get(fetch_url, params=params)
        response.raise_for_status()
        root = ET.fromstring(response.text)
        article_data = []
        for article in root.findall(".//PubmedArticle"):
            pmid_elem = article.find(".//PMID")
            pmid = pmid_elem.text if pmid_elem is not None else "Unknown"
            logger.debug(f"Processing article with PMID: {pmid}")
            title_elem = article.find(".//ArticleTitle")
            title = title_elem.text if title_elem is not None else "No title"
            if title_elem is None:
                logger.warning(f"No ArticleTitle for PMID: {pmid}")
            abstract_elem = article.find(".//AbstractText")
            abstract = abstract_elem.text if abstract_elem is not None else "No abstract"
            if abstract_elem is None:
                logger.warning(f"No AbstractText for PMID: {pmid}")
            authors = []
            for auth in article.findall(".//Author"):
                last_name_elem = auth.find("LastName")
                fore_name_elem = auth.find("ForeName")
                last_name = last_name_elem.text if last_name_elem is not None else "Unknown"
                fore_name = fore_name_elem.text if fore_name_elem is not None else "Unknown"
                authors.append(f"{last_name}, {fore_name}")
            authors = authors if authors else ["Unknown Author"]
            journal_elem = article.find(".//Journal/Title")
            journal = journal_elem.text if journal_elem is not None else "No journal"
            if journal_elem is None:
                logger.warning(f"No Journal Title for PMID: {pmid}")
            year_elem = article.find(".//PubDate/Year")
            month_elem = article.find(".//PubDate/Month")
            year = year_elem.text if year_elem is not None else "Unknown"
            month = month_elem.text if month_elem is not None else "Unknown"
            pub_date = f"{year}-{month}"
            if year_elem is None or month_elem is None:
                logger.warning(f"Missing PubDate Year or Month for PMID: {pmid}")
            doi_elem = article.find(".//ELocationID[@EIdType='doi']")
            doi = doi_elem.text if doi_elem is not None else "No DOI"
            if doi_elem is None:
                logger.info(f"No DOI for PMID: {pmid}")
            article_data.append({
                "title": title,
                "abstract": abstract,
                "authors": authors,
                "journal": journal,
                "pub_date": pub_date,
                "doi": doi
            })
        logger.info(f"Fetched details for {len(article_data)} articles")
    except requests.RequestException as e:
        errors.append(f"EFetch error: {str(e)}")
        article_data = []
        logger.error(f"Article fetch failed: {str(e)}")
    return {
        "article_data": article_data,
        "errors": errors,
        "performance_metrics": {},
        "critical_error": critical_error
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
    if not state["article_data"] and not state["perplexity_data"]:
        logger.warning("No data available for content generation")
        return {
            "generated_content": "",
            "errors": ["No data available"],
            "performance_metrics": {},
            "critical_error": True
        }
    current_date = datetime.date.today()
    combined_data = "\n".join([f"PubMed: {json.dumps(item)}" for item in state["article_data"]] + 
                              [f"Perplexity: {item}" for item in state["perplexity_data"] if item])
    length_mapping = {"Short": 500, "Medium": 1000, "Long": 1500}
    length_words = length_mapping.get(state["article_length"], 500)
    prompt = f"""
    Write a referenced, fact-checked, and neutral article about {state['user_input_topic']} specifically tailored for {state['target_audience']}. The article should be written in Australian English (e.g., use 'organise' instead of 'organize', 'centre' instead of 'center', etc.) and must only include factual claims supported by peer-reviewed sources, government publications, or credible medical organisations (e.g., CDC, FDA, WHO, NEJM, JAMA).
    
    Adjust the language, tone, and level of detail to be appropriate for the target audience:
    - For medical professionals or students, use technical medical terminology and provide in-depth analysis.
    - For the general public or patients, use plain language, explain any medical terms, and focus on practical implications and takeaways.
    
    Use the provided reference data to support factual claims, presenting the information in a way that is accessible and engaging for the target audience. The reference data includes:
    - **PubMed Data**: JSON-formatted entries with fields like 'title', 'abstract', 'authors', 'journal', 'pub_date', and 'doi' (e.g., '10.1000/xyz123').
    - **Perplexity Data**: Text lines, including a 'Sources' section at the end with entries formatted as 'Title (Author(s), Publication Date). Link: [URL]'.
    
    Follow these rules for references and links:
    - Extract URLs directly from the Perplexity 'Sources' section (e.g., lines starting with 'Link: [URL]') and use them verbatim as clickable links.
    - For PubMed data, construct URLs using:
      - DOIs if available (format: 'https://doi.org/[DOI]'), where [DOI] is the exact value from the 'doi' field.
      - PMIDs if no DOI is present (format: 'https://pubmed.ncbi.nlm.nih.gov/[PMID]/'), where [PMID] is the article ID.
    - Only include a reference in the list if a valid URL can be sourced from the reference data (Perplexity URLs, PubMed DOIs, or PMIDs). If no valid URL is available for a piece of data, do not fabricate a link—omit the reference instead.
    - Cross-reference all factual claims with the reference data and correct any discrepancies before finalising the article.
    
    The tone must remain objective and evidence-based, avoiding speculation or editorial commentary unless clearly labelled as such. Ensure all data is current as of {current_date} and note any areas where data is unavailable for review. At the end of the article, provide a reference list with clickable links in this exact format:
    - [Number]. Title (Author(s), Publication Date). Link: [URL]
    Each URL must be a complete, functional link directly extracted or constructed from the reference data—do not invent or use placeholder URLs (e.g., '.example.com') under any circumstances.
    
    - User Description: {state['user_input_description']}
    - Length: Approximately {length_words} words
    - Reference Data: {combined_data}
    """
    errors = []
    critical_error = False
    try:
        response = openai_client.chat.completions.create(
            model="o3-mini",
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=8000
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
    For example, if a statistic is slightly off but the overall message is correct, consider it acceptable.

    2. **AMA Citation Format**: Verify that citations are present and generally follow the AMA format (e.g., author names, year, journal, DOI if available).
    Be lenient with minor formatting issues, such as missing DOIs, incorrect punctuation, or slight deviations in style,
    as long as the citations are recognizable and provide enough information to locate the source.

    3. **Appropriateness for Audience**: Ensure the content is reasonably suitable for the target_audience, which is "{state['target_audience']}".
    For a doctor audience, the tone should be professional and include some technical terminology, but it does not need to be perfectly tailored.
    Allow for slight variations in tone (e.g., occasional simpler language) as long as the content is not completely inappropriate
    (e.g., written for children when the audience is doctors).

    Validation Guidelines:
    - Mark the article as "Valid" if it meets the above criteria in a general sense, even if there are minor issues.
    For example, if the content is mostly accurate, has recognizable citations, and is reasonably appropriate for the audience, it should be considered Valid.
    - Mark the article as "Invalid" only if there are significant issues, such as:
      - Major factual errors that could mislead or harm (e.g., recommending a dangerous treatment).
      - Complete absence of citations when references are clearly needed.
      - Content that is entirely inappropriate for the audience (e.g., written in a childish tone for doctors).
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

# Workflow setup
workflow = StateGraph(State)
workflow.add_node("process_user_input", process_user_input)
workflow.add_node("search_pubmed", search_pubmed)
workflow.add_node("search_perplexity", search_perplexity)
workflow.add_node("fetch_article_details", fetch_article_details)
workflow.add_node("check_data_availability", check_data_availability)
workflow.add_node("generate_content", generate_content)
workflow.add_node("validate_content", validate_content)

workflow.set_entry_point("process_user_input")
workflow.add_edge("process_user_input", "search_pubmed")
workflow.add_edge("process_user_input", "search_perplexity")
workflow.add_conditional_edges(
    "search_pubmed",
    route_after_pubmed,
    {
        "fetch_article_details": "fetch_article_details",
        "check_data_availability": "check_data_availability"
    }
)
workflow.add_edge("fetch_article_details", "check_data_availability")
workflow.add_edge("search_perplexity", "check_data_availability")
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

# --- CORS Middleware ---
fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic model for trending topics response
class TopicsResponse(BaseModel):
    topics: List[str]

# Celery task for fetching and storing topics
@celery_app.task(name="fetch_and_store_topics_task")
def fetch_and_store_topics_task():
    try:
        completion = openai_client.chat.completions.create(
            model="gpt-4o-search-preview",
            messages=[
                {
                    "role": "user",
                    "content": """Search the web and current online discussions to identify the 5 most talked-about medical topics today.
                    Provide only the list of topics, ranked by popularity, 
                    that are trending and suitable for creating articles for medical students. 
                    Do not include explanations or details beyond the topic names.
                    Just return the topic names, don't say any other thing
                    also don't add numbering"""
                }
            ]
        )
        topics = completion.choices[0].message.content.strip().split("\n")
        topics = [topic.strip() for topic in topics if topic.strip()]
        collection.delete_many({})
        collection.insert_one({"topics": topics})
        logger.info(f"Stored {len(topics)} trending topics in MongoDB")
        return {"topics": topics}
    except Exception as e:
        logger.error(f"Error fetching or storing topics: {str(e)}")
        raise

@fastapi_app.post("/generate-article")
async def generate_article(request: dict):
    initial_state = {
        "user_input_topic": request.get("user_input_topic", ""),
        "user_input_description": request.get("user_input_description", ""),
        "article_length": request.get("article_length", "Short"),
        "target_audience": request.get("target_audience", "general"),
        "pmids": [],
        "article_data": [],
        "perplexity_data": [],
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

@fastapi_app.post("/add-topics/", response_model=TopicsResponse)
async def fetch_and_store_topics():
    try:
        result = fetch_and_store_topics_task.delay().get(timeout=60)
        return result
    except Exception as e:
        logger.error(f"Error in endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching or storing topics: {str(e)}")

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

if __name__ == "__main__":
    config = uvicorn.Config(fastapi_app, host="0.0.0.0", port=8000, loop="asyncio")
    server = uvicorn.Server(config)
    server.run()
