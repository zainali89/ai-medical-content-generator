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
from pydantic import BaseModel
import nest_asyncio
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from firecrawl import FirecrawlApp
import http.client
import re
from langchain_google_genai import ChatGoogleGenerativeAI
from browser_use import Agent, Browser, BrowserConfig
import asyncio

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("webapp.log"), logging.StreamHandler()]
)

logger.info(f"Current working directory: {os.getcwd()}")
env_file_path = os.path.join(os.getcwd(), ".env")
logger.info(f"Looking for .env file at: {env_file_path}")
if os.path.exists(env_file_path):
    logger.info(".env file found")
else:
    logger.warning(".env file not found")

load_dotenv()

# API Keys from environment variables
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
PERPLEXITY_API_KEY = os.environ.get("PERPLEXITY_API_KEY")
FIRECRAWL_API_KEY = os.environ.get("FIRECRAWL_API_KEY")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
SCHEDULER_API_URL = os.environ.get("SCHEDULER_API_URL", "http://localhost:8000")

logger.info(f"OPENAI_API_KEY: {'Set' if OPENAI_API_KEY else 'Not set'}")
logger.info(f"PERPLEXITY_API_KEY: {'Set' if PERPLEXITY_API_KEY else 'Not set'}")
logger.info(f"FIRECRAWL_API_KEY: {'Set' if FIRECRAWL_API_KEY else 'Not set'}")
logger.info(f"GOOGLE_API_KEY: {'Set' if GOOGLE_API_KEY else 'Not set'}")
logger.info(f"SCHEDULER_API_URL: {SCHEDULER_API_URL}")

if not all([OPENAI_API_KEY, PERPLEXITY_API_KEY, FIRECRAWL_API_KEY, GOOGLE_API_KEY]):
    raise ValueError("One or more required API keys are missing.")

# Initialize OpenAI client
openai_client = OpenAI(api_key=OPENAI_API_KEY)
logger.info("OpenAI client initialized.")

# Initialize Firecrawl
firecrawl = FirecrawlApp(api_key=FIRECRAWL_API_KEY)
logger.info("Firecrawl client initialized.")

# FastAPI app initialization
app = FastAPI()

# CORS Middleware
app.add_middleware(
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

# YouTube transcript functions
def extract_video_id(video_url: str) -> str:
    regex = r'(?:https?://)?(?:www\.)?(?:youtube\.com/(?:[^/]+/.*|(?:v|e(?:mbed)?)|.*[?&]v=)|youtu\.be/)([^&]{11})'
    match = re.search(regex, video_url)
    if match:
        return match.group(1)
    else:
        logger.error(f"Invalid YouTube URL: {video_url}")
        return None

def get_youtube_transcript(video_urls: List[str]) -> List[Dict]:
    results = []
    for video_url in video_urls:
        video_id = extract_video_id(video_url)
        if not video_id:
            results.append({"video_url": video_url, "transcript_text": "", "error": f"Invalid YouTube URL: {video_url}"})
            continue
        
        conn = http.client.HTTPSConnection("youtube-transcripts.p.rapidapi.com")
        headers = {
            'x-rapidapi-key': "c2e122140fmsh5a0be1dac9a5e0bp1aedb8jsnb98711b40c8d",
            'x-rapidapi-host': "youtube-transcripts.p.rapidapi.com"
        }
        request_url = f"/youtube/transcript?url=https://www.youtube.com/watch?v={video_id}"
        
        try:
            conn.request("GET", request_url, headers=headers)
            res = conn.getresponse()
            if res.status != 200:
                logger.error(f"Error fetching transcript for {video_url}: {res.status} {res.reason}")
                results.append({"video_url": video_url, "transcript_text": "", "error": f"Error fetching transcript: {res.status} {res.reason}"})
                continue
            
            data = res.read()
            transcript_json = json.loads(data.decode("utf-8"))
            all_text = ""
            if 'content' in transcript_json:
                for segment in transcript_json['content']:
                    all_text += segment['text'] + " "
            
            logger.info(f"Successfully fetched transcript for {video_url}")
            results.append({"video_url": video_url, "transcript_text": all_text.strip(), "error": None})
        except Exception as e:
            logger.error(f"Exception fetching transcript for {video_url}: {str(e)}")
            results.append({"video_url": video_url, "transcript_text": "", "error": str(e)})
    
    return results

# State definition for LangGraph
class State(TypedDict):
    user_input_topic: str
    user_input_description: str
    article_length: str
    target_audience: str
    perplexity_data: List[str]
    firecrawl_data: List[str]
    medscape_data: List[str]
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
def search_medscape(state: State) -> dict:
    logger.info(f"Searching Medscape for: {state['user_input_topic']}")
    errors = []
    medscape_data = []
    
    async def run_medscape_agent():
        try:
            agent = Agent(
                task=f"""Go to https://www.medscape.com, log in with credentials Username: shane@connectthedocs.com.au and Password: Nelson01,
                      search for '{state['user_input_topic']}', and extract the latest article information, its contents, and write a brief summary.
                      Return the summary and source in the format:
                      - Summary: [Summary text]
                      - Source: [Title (Author(s), Date). Link: [URL]]""",
                llm=ChatGoogleGenerativeAI(
                    model="gemini-2.0-flash",
                    google_api_key=GOOGLE_API_KEY
                ),
            )
            result = await agent.run()
            return result
        except Exception as e:
            logger.error(f"Medscape agent failed: {str(e)}")
            return None

    # Run the async agent in the current event loop
    try:
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(run_medscape_agent())
        
        if result:
            # Parse the result (assuming it's a string with summary and source)
            lines = result.split("\n")
            current_summary = []
            current_source = None
            for line in lines:
                if line.startswith("- Summary:"):
                    current_summary.append(line.replace("- Summary:", "").strip())
                elif line.startswith("- Source:"):
                    current_source = line.replace("- Source:", "").strip()
                else:
                    if current_summary:
                        current_summary.append(line.strip())
            
            summary_text = " ".join(current_summary).strip()
            if summary_text and current_source:
                medscape_data.append(f"Medscape content: {summary_text}\nSource: {current_source}")
                logger.info(f"Extracted Medscape content for topic: {state['user_input_topic']}")
            else:
                logger.warning("No valid Medscape content extracted")
                medscape_data.append("No relevant Medscape content found")
        else:
            logger.warning("No Medscape content returned by agent")
            medscape_data.append("No relevant Medscape content found")
    
    except Exception as e:
        errors.append(f"Medscape extraction error: {str(e)}")
        logger.error(f"Medscape extraction failed: {str(e)}")
        medscape_data.append("No relevant Medscape content found")
    
    return {
        "medscape_data": medscape_data,
        "errors": errors,
        "performance_metrics": {},
        "critical_error": False
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
    results = get_youtube_transcript(state["youtube_links"])
    youtube_data = []
    errors = []
    
    for result in results:
        if result.get("error"):
            errors.append(f"YouTube transcript error for {result['video_url']}: {result['error']}")
        else:
            youtube_data.append(f"Transcript from {result['video_url']}: {result['transcript_text']}")
    
    return {
        "youtube_data": youtube_data,
        "errors": errors,
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
    
    if not any([state["perplexity_data"], state["firecrawl_data"], state["medscape_data"], state["docs_data"], state["youtube_data"]]):
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
    medscape_data = "\n".join([f"Medscape: {item}" for item in state["medscape_data"] if item])
    docs_data = "\n".join([f"Document: {item}" for item in state["docs_data"] if item])
    youtube_data = "\n".join([f"YouTube: {item}" for item in state["youtube_data"] if item])
    
    length_mapping = {"Short": 750, "Medium": 1250, "Long": 2250}
    length_words = length_mapping.get(state["article_length"], 750)
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
    - **Medscape**: Scraped content prefixed with 'Medscape content: [content]\nSource: [Title (Author(s), Date). Link: [URL]]'. Use the URL provided in the source.
    - **Documents**: Content extracted from document files, prefixed with 'Document:'.
    - **YouTube**: Content transcribed from YouTube videos, prefixed with 'YouTube:'.
    
    Rules for references and content:
    - Extract Perplexity URLs from lines like 'Link: [URL]' and use them unchanged.
    - Extract Firecrawl URLs from the prefix 'Firecrawl content from [URL]' and use them unchanged.
    - Extract Medscape URLs from the 'Source: [Title (Author(s), Date). Link: [URL]]' line and use them unchanged.
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

    Make sure the link you return is always clickable
    
    - User Description: {state['user_input_description']}
    - Length: ~{length_words} words
    - Reference Data (Perplexity): 
    {perplexity_data if perplexity_data else 'No Perplexity data available'}
    - Reference Data (Firecrawl): 
    {firecrawl_data if firecrawl_data else 'No Firecrawl data available'}
    - Reference Data (Medscape): 
    {medscape_data if medscape_data else 'No Medscape data available'}
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

@timeit
def check_data_availability(state: State) -> dict:
    logger.info("Checking data availability")
    data_sources = [
        ("Perplexity", state["perplexity_data"]),
        ("Firecrawl", state["firecrawl_data"]),
        ("Medscape", state["medscape_data"]),
        ("Docs", state["docs_data"]),
        ("YouTube", state["youtube_data"])
    ]
    errors = []
    for source, data in data_sources:
        if not data:
            logger.warning(f"No data available from {source}")
            errors.append(f"No data available from {source}")
    
    if state["critical_error"]:
        logger.warning("Critical error detected, but proceeding with available data")
    
    return {
        "errors": errors,
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
workflow.add_node("search_medscape", search_medscape)
workflow.add_node("process_docs", process_docs)
workflow.add_node("process_youtube_links", process_youtube_links)
workflow.add_node("check_data_availability", check_data_availability)
workflow.add_node("generate_content", generate_content)
workflow.add_node("validate_content", validate_content)

workflow.set_entry_point("process_user_input")
workflow.add_edge("process_user_input", "search_perplexity")
workflow.add_edge("process_user_input", "extract_firecrawl_content")
workflow.add_edge("process_user_input", "search_medscape")
workflow.add_edge("process_user_input", "process_docs")
workflow.add_edge("process_user_input", "process_youtube_links")
workflow.add_edge("search_perplexity", "check_data_availability")
workflow.add_edge("extract_firecrawl_content", "check_data_availability")
workflow.add_edge("search_medscape", "check_data_availability")
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

langgraph_app = workflow.compile()

# Pydantic models for API requests
class UrlRequest(BaseModel):
    url: str

# API endpoint to fetch topics from the scheduler service
@app.get("/get-topics")
async def get_topics():
    try:
        logger.info(f"Fetching topics from scheduler at {SCHEDULER_API_URL}/get-topics")
        response = requests.get(f"{SCHEDULER_API_URL}/get-topics")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Error fetching topics from scheduler: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching topics from scheduler: {str(e)}")

# FastAPI endpoints
@app.post("/generate-article")
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
        "medscape_data": [],
        "docs_data": [],
        "youtube_data": [],
        "generated_content": "",
        "errors": [],
        "performance_metrics": {},
        "critical_error": False
    }
    logger.info("Starting workflow execution")
    start_time = time.time()
    try:
        final_state = langgraph_app.invoke(initial_state)
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

@app.post("/extract")
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

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    config = uvicorn.Config(app, host="0.0.0.0", port=8000, loop="asyncio")
    server = uvicorn.Server(config)
    server.run()
