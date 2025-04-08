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
FIRECRAWL_API_KEY = os.environ.get("FIRECRAWL_API_KEY")
MONGODB_URI = os.environ.get("MONGODB_URI")

if not all([OPENAI_API_KEY, MONGODB_URI, FIRECRAWL_API_KEY]):
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
    raise ValueError(f"Error connecting to MongoDB: {str(e)}")

db = client_mongo['TopMedicalArticles']
collection = db['topics']

async def fetch_and_store_topics():
    try:
        aus_tz = pytz.timezone("Australia/Sydney")
        current_time = datetime.datetime.now(aus_tz)
        logger.info(f"Starting topic fetch at {current_time.strftime('%Y-%m-%d %H:%M:%S %Z')} in Australian time")

        # Add a unique identifier to the prompt to avoid caching
        unique_id = str(uuid.uuid4())
        prompt = f"""As of {current_time.strftime('%Y-%m-%d %H:%M:%S %Z')}, perform a fresh, real-time search of the web and current online discussions to identify the 5 most talked-about medical topics today. Focus on trends that have emerged or gained significant attention in the last 24 hours. This request is unique (ID: {unique_id}) to ensure a new search. Provide only the list of topics, ranked by popularity, that are trending and suitable for creating articles for medical students. Just return the topic names, don't say any other thing, and don't add numbering."""
        
        completion = openai_client.chat.completions.create(
            model="gpt-4o-search-preview",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
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
        
        return {"topics": cleaned_topics}
    except Exception as e:
        logger.error(f"Error fetching and storing topics: {str(e)}")
        raise
