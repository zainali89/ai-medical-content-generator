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
import PyPDF2
from docx import Document
import tempfile
import urllib.parse
from google import genai
from google.genai import types

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
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
SCHEDULER_API_URL = os.environ.get("SCHEDULER_API_URL", "http://localhost:8000")

logger.info(f"OPENAI_API_KEY: {'Set' if OPENAI_API_KEY else 'Not set'}")
logger.info(f"PERPLEXITY_API_KEY: {'Set' if PERPLEXITY_API_KEY else 'Not set'}")
logger.info(f"FIRECRAWL_API_KEY: {'Set' if FIRECRAWL_API_KEY else 'Not set'}")
logger.info(f"GEMINI_API_KEY: {'Set' if GEMINI_API_KEY else 'Not set'}")
logger.info(f"SCHEDULER_API_URL: {SCHEDULER_API_URL}")

if not all([PERPLEXITY_API_KEY, FIRECRAWL_API_KEY, GEMINI_API_KEY]):
    raise ValueError("One or more required API keys are missing.")

# Initialize OpenAI client
openai_client = OpenAI(api_key=OPENAI_API_KEY)
logger.info("OpenAI client initialized.")

# Initialize Google GenAI client
genai_client = genai.Client(api_key=GEMINI_API_KEY)
logger.info("Google GenAI client initialized.")

# Check for environment variables that might override model selection
gemini_model_env = os.environ.get("GEMINI_MODEL")
if gemini_model_env:
    logger.info(f"Found GEMINI_MODEL environment variable: {gemini_model_env}")

# Check if there's a default model configured
try:
    default_model = genai_client.default_model
    logger.info(f"Gemini default model: {default_model}")
except Exception as e:
    logger.info("No default Gemini model configured or could not access it")

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
    reference_urls: List[str]
    docs_files: List[str]
    youtube_links: List[str]
    docs_data: List[str]
    youtube_data: List[str]
    generated_content: str
    errors: Annotated[List[str], operator.add]
    performance_metrics: Annotated[Dict[str, float], lambda x, y: {**x, **y}]
    critical_error: Annotated[bool, lambda x, y: x or y]
    skip_perplexity: bool

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
            # Check if scraped is a dictionary or an object
            if isinstance(scraped, dict):
                content = scraped.get('markdown', 'No content found')
            else:
                # Assume it's a ScrapeResponse object with attributes
                content = getattr(scraped, 'markdown', 'No content found')
                if content == 'No content found' and hasattr(scraped, 'content'):
                    content = scraped.content or 'No content found'
            
            firecrawl_data.append(f"Firecrawl content from {url}: {content}")
            logger.info(f"Successfully extracted Firecrawl content from {url}")
        except Exception as e:
            error_msg = f"Firecrawl extraction error for {url}: {str(e)}"
            errors.append(error_msg)
            logger.error(error_msg)
            # Log response structure for debugging
            try:
                logger.debug(f"Firecrawl response for {url}: {scraped}")
            except:
                pass
    
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
    # Check if any of reference_urls, docs_files, or youtube_links are non-empty
    skip_perplexity = bool(state["reference_urls"] or state["docs_files"] or state["youtube_links"])
    logger.info(f"Using topic: '{corrected_topic}', skip_perplexity: {skip_perplexity}")
    return {
        "user_input_topic": corrected_topic,
        "skip_perplexity": skip_perplexity,
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
                "content": """You are a specialized medical research assistant with expertise in searching and retrieving highly technical, 
                clinical and research-focused information from medical literature that meets the highest standards of medical journalism. 
                You MUST retrieve information ONLY from:
                1. Peer-reviewed medical journals with high impact factors (e.g., NEJM, The Lancet, JAMA, BMJ)
                2. Official clinical guidelines from recognized health organizations (WHO, CDC, NIH, etc.)
                3. Medical academic institutions and teaching hospitals
                4. Specialized medical databases (PubMed, Cochrane Library, etc.)
                
                MEDICAL JOURNALISM STANDARDS:
                - Organize information using standardized medical categories and headings
                - Each source must be credible, preferably from indexed, peer-reviewed journals
                - Include exact statistics with their confidence intervals when available
                - Cover both established consensus and emerging research
                - Present balanced perspectives on controversial topics
                - Distinguish between practice guidelines and research findings
                - Ensure information is contextualized with appropriate caveats (population specifics, study limitations, etc.)
                
                CONTENT REQUIREMENTS:
                - Focus exclusively on scientifically validated, evidence-based medical information
                - Include specific medical terminology, diagnostic criteria, treatment protocols, and clinical outcomes
                - Cite recent research (within last 3-5 years when available)
                - Provide detailed statistics, study methodologies, and findings
                - Include information on current consensus and areas of ongoing research
                - NEVER fabricate or extrapolate beyond what is explicitly stated in reliable sources
                - NEVER reuse information from previous queries - each search must be completely fresh
                - RESET your memory and knowledge for each new query to prevent contamination
                
                Your responses must meet professional medical standards and be suitable for clinical or academic use."""
            },
            {
                "role": "user",
                "content": f"""Conduct a thorough search for specialized medical information on {state['user_input_topic']} 
                focusing specifically on: '{state['user_input_description']}'
                
                Target information for audience: {state['target_audience']}
                
                REQUIREMENTS:
                1. Begin with a concise summary of current clinical understanding and research status
                2. Provide detailed medical information including:
                   - Precise diagnostic criteria and classifications
                   - Evidence-based treatment approaches with efficacy data
                   - Pathophysiology and mechanisms of action
                   - Epidemiological data and relevant statistics
                   - Current clinical guidelines and standard of care
                   - Recent advances, trials, or novel approaches
                   - Areas of medical consensus vs. controversy
                
                3. Structure your response in clearly labeled clinical subsections
                4. Include ONLY facts that can be verified through medical literature
                5. End with a comprehensive reference list in this format ONLY:
                   [Number] Title (Author(s), Publication Date). Link: [direct URL to medical source]
                   Ensure all references are from peer-reviewed journals or official medical sources.
                
                QUALITY STANDARDS:
                - Citations must be accurate, recent (within 5 years when applicable), and from high-impact sources
                - Include both established knowledge and emerging research
                - Present conflicting views where consensus is lacking
                - Provide exact statistics with confidence intervals when available
                - Distinguish between guidelines, recommendations, and research findings
                
                IMPORTANT: Search fresh sources for THIS request only. Do not reference any information from 
                previous searches or include irrelevant topics (e.g., sleep apnea or other unrelated conditions). 
                Focus EXCLUSIVELY on {state['user_input_topic']}."""
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
    errors = []
    docs_data = []
    
    for doc in state["docs_files"]:
        try:
            # Check if the input is a URL
            is_url = doc.startswith("http://") or doc.startswith("https://")
            file_path = doc
            
            if is_url:
                # Create a temporary file for the downloaded content
                response = requests.get(doc, stream=True)
                response.raise_for_status()
                
                # Get the filename from the URL or Content-Disposition header
                content_disposition = response.headers.get("content-disposition")
                if content_disposition and "filename=" in content_disposition:
                    filename = content_disposition.split("filename=")[1].strip('"')
                else:
                    filename = urllib.parse.urlparse(doc).path.split("/")[-1]
                
                # Ensure the file has a valid extension
                if not filename.lower().endswith((".pdf", ".docx")):
                    raise ValueError(f"Unsupported file format in URL: {filename}")
                
                # Save to a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as temp_file:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            temp_file.write(chunk)
                    file_path = temp_file.name
                logger.info(f"Downloaded document from {doc} to temporary file: {file_path}")
            
            # Process the file
            file_ext = os.path.splitext(file_path)[1].lower()
            text = ""
            
            if file_ext == ".pdf":
                with open(file_path, "rb") as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page in pdf_reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + " "
                text = text.strip()
                if not text:
                    raise ValueError("No text extracted from PDF")
                logger.info(f"Successfully extracted text from PDF: {doc}")
            
            elif file_ext == ".docx":
                doc = Document(file_path)
                for para in doc.paragraphs:
                    if para.text.strip():
                        text += para.text + " "
                text = text.strip()
                if not text:
                    raise ValueError("No text extracted from DOCX")
                logger.info(f"Successfully extracted text from DOCX: {doc}")
            
            else:
                raise ValueError(f"Unsupported file format: {file_ext}")
            
            docs_data.append(f"Document: {text}")
            
            # Clean up temporary file if it was a URL
            if is_url:
                os.unlink(file_path)
                logger.info(f"Deleted temporary file: {file_path}")
        
        except Exception as e:
            errors.append(f"Document processing error for {doc}: {str(e)}")
            logger.error(f"Failed to process document {doc}: {str(e)}")
            # Clean up temporary file if it exists
            if is_url and os.path.exists(file_path):
                os.unlink(file_path)
                logger.info(f"Deleted temporary file after error: {file_path}")
    
    return {
        "docs_data": docs_data,
        "errors": errors,
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
    
    if not state["perplexity_data"] and not state["firecrawl_data"] and not state["docs_data"] and not state["youtube_data"]:
        logger.warning("No data available for content generation")
        return {
            "generated_content": "",
            "errors": ["No data available"],
            "performance_metrics": {},
            "critical_error": True
        }
    current_date = datetime.date.today()
    
    # Handle perplexity_data as None if skipped
    perplexity_data = None if state["skip_perplexity"] else "\n".join([f"Perplexity: {item}" for item in state["perplexity_data"] if item])
    firecrawl_data = "\n".join([f"Firecrawl: {item}" for item in state["firecrawl_data"] if item])
    docs_data = "\n".join([f"Document: {item}" for item in state["docs_data"] if item])
    youtube_data = "\n".join([f"YouTube: {item}" for item in state["youtube_data"] if item])
    
    length_mapping = {"Short": 800, "Medium": 1200, "Long": 2200}
    length_words = length_mapping.get(state["article_length"])
    print(state["user_input_topic"])
    print(state["user_input_description"])
    print(state["article_length"])
    print(state["target_audience"])
    print(state["perplexity_data"])
    print(state["firecrawl_data"])
    print(state["docs_data"])
    prompt = f"""
    Write a referenced, fact-checked, and neutral article about {state['user_input_topic']} specifically tailored for {state['target_audience']}. Use Australian English (e.g., 'organise', 'centre') and base all factual claims STRICTLY on the provided reference data from peer-reviewed or credible sources.

    ARTICLE FORMAT AND STRUCTURE:
    - Title: Use ONLY "{state['user_input_topic']}" as the title. Do not modify, expand, or rewrite this title.
    - Do NOT repeat the topic as an introduction paragraph or summary at the beginning of the article.
    - Structure the article using STANDARD MEDICAL SECTION HEADINGS
    - IMPORTANT: Use ONLY these standard medical section headings. Use the user's description text as section headings.
    - Include only relevant sections based on the topic - not all sections may be necessary.
    - Be concise and prioritize completion over verbose explanations.
    
    CRITICAL REQUIREMENT: ALL ARTICLES MUST INCLUDE A COMPLETE "REFERENCES" SECTION AT THE END WITH PROPERLY FORMATTED CITATIONS. This is non-negotiable.
    Your response will be rejected if references are missing or incomplete. Reserve at least 20% of your word count for references.

    IMPORTANT: The article MUST be EXACTLY {length_words} words in length (±10%) INCLUDING the references section. 
    Structure your article to fit this length requirement, ensuring references are never cut off.

    TO PREVENT CUTOFFS: Make the main content shorter to ensure you have enough space for the references section.
    DO NOT leave any sentences unfinished.

    IMPORTANT: DO NOT HALLUCINATE OR INVENT ANY INFORMATION. If the provided reference data doesn't cover a particular aspect of the topic, explicitly state that information is limited rather than making up facts. Only include information that is directly supported by the reference data provided below.

    Adjust language and detail for the audience:
    - Medical Professionals (Doctors): Employ precise medical terminology and provide comprehensive, detailed analysis.
    - Students: Utilize technical medical vocabulary and deliver thorough, educational analysis.
    - General Public: Use simple, everyday words, clarify any complex terms, and highlight useful, easy-to-apply information.
    - Patients: Use clear, straightforward language, explain medical terms simply, and emphasize practical, health-related advice

    Use the reference data below to support claims, ensuring the article is engaging and accessible. The data includes:
    - **Perplexity**: Text with a 'Sources' section (e.g., 'Title (Author(s), Date). Link: [URL]'). Use URLs exactly as provided.
    - **Firecrawl**: Scraped content prefixed with source URL (e.g., 'Website content from [URL]: [content]'). Use the URL provided in the prefix.
    - **Documents**: Content extracted from document files, prefixed with 'Document:'.
    - **YouTube**: Content from YouTube videos, prefixed with 'YouTube:'.

    Rules for references and content:
    - Extract Perplexity URLs from lines like 'Link: [URL]' and use them unchanged.
    - Extract Website URLs from the prefix 'Website content from [URL]' and use them unchanged.
    - For Document content, cite as "From document analysis" if no specific citation is available.
    - For YouTube content, cite as "From [YouTube video title]" if available.
    - Include a reference only if it has a valid URL from the data. If no URL exists, omit it—do NOT invent links (e.g., no '.example.com').
    - Verify all facts against the data and correct errors.
    - Always make sure the links are clickable.
    - Only include the links in the references.
    - If you're uncertain about any information, indicate this clearly rather than guessing.
    - For any statistical claims, medical recommendations, or specific treatments, cite the exact source from the reference data.

    Keep the tone objective and evidence-based, current as of {current_date}, and note missing data if applicable. 

    STRUCTURE OF YOUR RESPONSE:
    1. Title: ONLY "{state['user_input_topic']}" (not prefixed with "Medical Topic:" or any other text)
    2. Main article content (start immediately with relevant information, be concise)
    3. Mandatory "References" heading (exactly as shown: "References")
    4. Complete numbered reference list in this format ONLY:
       - [Number]. Title (Author(s), Date). Link: [URL]

    IMPORTANT: DO NOT include ANY additional text after the references section. Do not include word counts, notes about article length, or any other metadata at the end of your response.

    EVERY reference you cite in-text MUST appear in the references section. Reserve AT LEAST 20% of your word count for references.
    Double-check that your response ends with complete references before submitting.

    IMPORTANT: PRIORITIZE COMPLETING THE REFERENCES SECTION OVER ADDING MORE CONTENT. If you're running out of space, make the article content shorter to ensure you have room for references.

    - User Description: {state['user_input_description']}
    - Length: ~{length_words} words (INCLUDING references)
    - Reference Data (Perplexity): 
    {perplexity_data if perplexity_data else 'No Perplexity data available'}
    - Reference Data (Website): 
    {firecrawl_data if firecrawl_data else 'No Firecrawl data available'}
    - Reference Data (Documents): 
    {docs_data if docs_data else 'No Document data available'}
    - Reference Data (YouTube): 
    {youtube_data if youtube_data else 'No YouTube data available'}

    If Reference Data (Website), (Documents), or (YouTube) has any kind of data, then Perplexity data should be ignored and set to None.
    """
    errors = []
    critical_error = False
    try:
        # Estimate required tokens based on target word count (use higher ratio for medical content)
        estimated_tokens = int(length_words * 2.0)  # Increase from 1.5 to 2.0 for medical content
        reserved_tokens = 2000  # Increased buffer for references and formatting
        max_tokens = min(8000, max(4000, estimated_tokens * 2 + reserved_tokens))
        
        # Using Google's Gemini instead of OpenAI
        model_name = "gemini-2.5-pro-preview-05-06"
        logger.info(f"Attempting to use Gemini model: {model_name}")
        
        try:
            # Log available models
            available_models = genai_client.list_models()
            model_names = [model.name for model in available_models]
            logger.info(f"Available Gemini models: {model_names}")
        except Exception as e:
            logger.warning(f"Could not list available models: {str(e)}")
        
        # Add a safety check to adjust content length based on article length
        if state["article_length"] == "Long":
            # For longer articles, we need to ensure more space for references
            logger.info("Long article detected, adjusting max_output_tokens to ensure room for references")
            max_tokens = min(max_tokens, 7000)  # Limit to 7000 tokens to ensure completion
        
        response_stream = genai_client.models.generate_content_stream(
            model=model_name,
            contents=[types.Content(role="user", parts=[types.Part.from_text(text=prompt)])],
            config=types.GenerateContentConfig(
                temperature=0.3,
                top_p=0.9,
                max_output_tokens=max_tokens
            )
        )
        
        logger.info(f"Gemini API call initiated with model: {model_name}")
        
        # Collect response chunks
        response_chunks = []
        for chunk in response_stream:
            # Check if chunk.text is not None before appending
            if chunk.text is not None:
                response_chunks.append(chunk.text)
            # If text is None but has parts, try to extract text from parts
            elif hasattr(chunk, 'parts'):
                for part in chunk.parts:
                    if hasattr(part, 'text') and part.text:
                        response_chunks.append(part.text)
        
        content = "".join(response_chunks).strip()
        
        # If content is empty, try to extract content from the final response if available
        if not content and hasattr(response_stream, 'result'):
            result = response_stream.result
            if hasattr(result, 'text') and result.text:
                content = result.text
            elif hasattr(result, 'parts'):
                for part in result.parts:
                    if hasattr(part, 'text') and part.text:
                        content += part.text
        
        # Verify the word count matches the target length
        word_count = len(content.split())
        target_words = length_mapping.get(state["article_length"], 800)
        min_words = int(target_words * 0.9)  # 10% below target
        max_words = int(target_words * 1.1)  # 10% above target
        
        if min_words <= word_count <= max_words:
            logger.info(f"Content generated successfully using Gemini - Word count: {word_count} (target: {target_words})")
        else:
            logger.warning(f"Generated content doesn't meet length requirements - Got {word_count} words, target was {target_words} (±10%)")
        
        logger.info("Content generated successfully using Gemini")
    except Exception as e:
        errors.append(f"Gemini API error: {str(e)}")
        content = ""
        critical_error = True
        logger.error(f"Content generation failed with Gemini: {str(e)}")
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

def route_after_process_user_input(state: State) -> str:
    if state["skip_perplexity"]:
        return "extract_firecrawl_content"
    return "search_perplexity"

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
workflow.add_conditional_edges(
    "process_user_input",
    route_after_process_user_input,
    {
        "search_perplexity": "search_perplexity",
        "extract_firecrawl_content": "extract_firecrawl_content"
    }
)
workflow.add_edge("search_perplexity", "extract_firecrawl_content")
workflow.add_edge("extract_firecrawl_content", "process_docs")
workflow.add_edge("process_docs", "process_youtube_links")
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
    # Validate required fields
    if not request.get("user_input_topic"):
        logger.error("Missing required field: user_input_topic")
        raise HTTPException(status_code=400, detail="Missing required field: user_input_topic")
    
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
        "docs_data": [],
        "youtube_data": [],
        "generated_content": "",
        "errors": [],
        "performance_metrics": {},
        "critical_error": False,
        "skip_perplexity": False
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
        # Handle both dictionary and object responses
        if isinstance(scraped, dict):
            content = scraped.get('markdown', 'No content found')
        else:
            content = getattr(scraped, 'markdown', 'No content found')
            if content == 'No content found' and hasattr(scraped, 'content'):
                content = scraped.content or 'No content found'
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
