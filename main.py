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
from fastapi.middleware.cors import CORSMiddleware
from firecrawl import FirecrawlApp
import pytz
from apscheduler.schedulers.asyncio import AsyncIOScheduler

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize MongoDB client
mongo_client = MongoClient(os.getenv("MONGO_URI"), server_api=ServerApi('1'))
db = mongo_client["medical_topics_db"]
collection = db["topics"]

# Initialize FastAPI app
fastapi_app = FastAPI()

# Fetch and store topics function
async def fetch_and_store_topics():
    try:
        aus_tz = pytz.timezone("Australia/Sydney")
        current_time = datetime.datetime.now(aus_tz)
        logger.info(f"Starting topic fetch at {current_time.strftime('%Y-%m-%d %H:%M:%S %Z')} in Australian time")

        # Modify the prompt to ensure fresh data
        prompt = f"""As of {current_time.strftime('%Y-%m-%d %H:%M:%S %Z')}, search the web and current online discussions to identify the 5 most talked-about medical topics today.
                    Provide only the list of topics, ranked by popularity, 
                    that are trending and suitable for creating articles for medical students. 
                    Just return the topic names, don't say any other thing
                    also don't add numbering"""
        
        completion = openai_client.chat.completions.create(
            model="gpt-4o-search-preview",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        topics = completion.choices[0].message.content.strip().split("\n")
        topics = [topic.strip() for topic in topics if topic.strip()]
        
        logger.info(f"Fetched topics: {topics}")
        
        # Clear old topics and insert new ones
        deleted_count = collection.delete_many({}).deleted_count
        logger.info(f"Deleted {deleted_count} old topic documents from MongoDB")
        
        insert_result = collection.insert_one({"topics": topics, "timestamp": current_time.isoformat()})
        logger.info(f"Stored {len(topics)} trending topics in MongoDB with timestamp {current_time.isoformat()}, ID: {insert_result.inserted_id}")
        
        return {"topics": topics}
    except Exception as e:
        logger.error(f"Error fetching and storing topics: {str(e)}")
        raise

# Create a scheduler
scheduler = AsyncIOScheduler()

# Startup event handler
@fastapi_app.on_event("startup")
async def startup_event():
    try:
        # Initial fetch
        await fetch_and_store_topics()
        logger.info("Initial topic fetch completed at startup")
        
        # Schedule the task to run every minute
        scheduler.add_job(fetch_and_store_topics, 'interval', minutes=1)
        scheduler.start()
        logger.info("Scheduled topic fetching every 1 minute")
    except Exception as e:
        logger.error(f"Failed to fetch initial topics or set up scheduler: {str(e)}")
        raise

# Add an endpoint to retrieve the latest topics
@fastapi_app.get("/topics")
async def get_topics():
    try:
        latest_topics = collection.find_one({}, sort=[("timestamp", -1)])  # Get the most recent document
        if not latest_topics:
            raise HTTPException(status_code=404, detail="No topics found")
        return {
            "topics": latest_topics.get("topics", []),
            "timestamp": latest_topics.get("timestamp", "")
        }
    except Exception as e:
        logger.error(f"Error retrieving topics: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Run the app locally
if __name__ == "__main__":
    uvicorn.run(fastapi_app, host="0.0.0.0", port=8000)
