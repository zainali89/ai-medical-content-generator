"""
Application configuration: environment variables, logging, and API client initialization.
"""

import logging
import os

from dotenv import load_dotenv
from openai import OpenAI
from google import genai
from firecrawl import FirecrawlApp

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("ai_medical")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("webapp.log"), logging.StreamHandler()],
)

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
logger.info(f"Current working directory: {os.getcwd()}")
env_file_path = os.path.join(os.getcwd(), ".env")
if os.path.exists(env_file_path):
    logger.info(".env file found")
else:
    logger.warning(".env file not found")

load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
PERPLEXITY_API_KEY = os.environ.get("PERPLEXITY_API_KEY")
FIRECRAWL_API_KEY = os.environ.get("FIRECRAWL_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
RAPIDAPI_KEY = os.environ.get("RAPIDAPI_KEY")
SCHEDULER_API_URL = os.environ.get("SCHEDULER_API_URL", "http://localhost:8000")

logger.info(f"OPENAI_API_KEY: {'Set' if OPENAI_API_KEY else 'Not set'}")
logger.info(f"PERPLEXITY_API_KEY: {'Set' if PERPLEXITY_API_KEY else 'Not set'}")
logger.info(f"FIRECRAWL_API_KEY: {'Set' if FIRECRAWL_API_KEY else 'Not set'}")
logger.info(f"GEMINI_API_KEY: {'Set' if GEMINI_API_KEY else 'Not set'}")
logger.info(f"RAPIDAPI_KEY: {'Set' if RAPIDAPI_KEY else 'Not set'}")
logger.info(f"SCHEDULER_API_URL: {SCHEDULER_API_URL}")

if not all([PERPLEXITY_API_KEY, FIRECRAWL_API_KEY, GEMINI_API_KEY]):
    raise ValueError("One or more required API keys are missing.")

# ---------------------------------------------------------------------------
# API Clients
# ---------------------------------------------------------------------------
openai_client = OpenAI(api_key=OPENAI_API_KEY)
logger.info("OpenAI client initialized.")

genai_client = genai.Client(api_key=GEMINI_API_KEY)
logger.info("Google GenAI client initialized.")

firecrawl_client = FirecrawlApp(api_key=FIRECRAWL_API_KEY)
logger.info("Firecrawl client initialized.")
