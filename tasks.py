# tasks.py
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
import uuid
import tenacity  # Add tenacity for retry logic

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()]
)

# Note: load_dotenv() is not needed in DigitalOcean App Platform as variables are injected at runtime
# load_dotenv()  # Comment out or remove this line

# API Keys and MongoDB URI from environment variables
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
FIRECRAWL_API_KEY = os.environ.get("FIRECRAWL_API_KEY")
MONGODB_URI = os.environ.get("MONGO_URI")  # Changed from MONGODB_URI to MONGO_URI

# Check for missing environment variables and log specifics
missing_vars = []
if not OPENAI_API_KEY:
    missing_vars.append("OPENAI_API_KEY")
if not FIRECRAWL_API_KEY:
    missing_vars.append("FIRECRAWL_API_KEY")
if not MONGODB_URI:
    missing_vars.append("MONGO_URI")

if missing_vars:
    logger.error(f"Missing environment variables: {', '.join(missing_vars)}")
    raise ValueError(f"One or more required keys are missing: {', '.join(missing_vars)}")

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
    raise ValueError(f"Error connecting to MongoDB: {str(e)}")

db = client_mongo['TopMedicalArticles']
collection = db['topics']

# Retry decorator for OpenAI API calls
@tenacity.retry(
    stop=tenacity.stop_after_attempt(3),  # Retry 3 times
    wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),  # Exponential backoff: 4s, 8s, 10s
    retry=tenacity.retry_if_exception_type(Exception),  # Retry on any exception
    before_sleep=tenacity.before_sleep_log(logger, logging.INFO)  # Log before each retry
)
async def fetch_topics_from_openai(prompt):
    completion = openai_client.chat.completions.create(
        model="gpt-4o-search-preview",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    return completion

async def fetch_and_store_topics():
    try:
        aus_tz = pytz.timezone("Australia/Sydney")
        current_time = datetime.datetime.now(aus_tz)
        logger.info(f"Starting topic fetch at {current_time.strftime('%Y-%m-%d %H:%M:%S %Z')} in Australian time")

        # Add a unique identifier to the prompt to avoid caching
        unique_id = str(uuid.uuid4())
        prompt = f"""As of {current_time.strftime('%Y-%m-%d %H:%M:%S %Z')}, perform a fresh, real-time search of the web and current online discussions to identify the 5 most talked-about medical topics today. Focus on trends that have emerged or gained significant attention in the last 24 hours. This request is unique (ID: {unique_id}) to ensure a new search. Provide only the list of topics, ranked by popularity, that are trending and suitable for creating articles for medical students. Just return the topic names, don't say any other thing, and don't add numbering."""
        
        # Fetch topics with retry logic
        completion = await fetch_topics_from_openai(prompt)
        
        # Split the response into individual topics
        topics = completion.choices[0].message.content.strip().split("\n")
        # Clean up each topic
        cleaned_topics = []
        for topic in topics:
            topic = topic.strip()
            for marker in ['- ', '* ', '1. ', '2. ', '3. ', '4. ', '5. ']:
                if topic.startswith(marker):
                    topic = topic[len(marker):].strip()
                    break
            if topic:
                cleaned_topics.append(topic)
        
        logger.info(f"Fetched and cleaned topics: {cleaned_topics}")
        
        # Clear old topics and insert new ones
        deleted_count = collection.delete_many({}).deleted_count
        logger.info(f"Deleted {deleted_count} old topic documents from MongoDB")
        
        insert_result = collection.insert_one({"topics": cleaned_topics, "timestamp": current_time.isoformat()})
        logger.info(f"Stored {len(cleaned_topics)} trending topics in MongoDB with timestamp {current_time.isoformat()}, ID: {insert_result.inserted_id}")
        
        # Verify the insertion by querying MongoDB
        latest_document = collection.find_one({}, sort=[("timestamp", -1)])
        if latest_document:
            logger.info(f"Verified: Latest document in MongoDB: {latest_document}")
        else:
            logger.error("Verification failed: No documents found in MongoDB after insertion")
        
        return {"topics": cleaned_topics}
    except Exception as e:
        logger.error(f"Error fetching and storing topics: {str(e)}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        raise
