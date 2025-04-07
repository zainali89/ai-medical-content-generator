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

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
fastapi_app = FastAPI()

# Keep the existing fetch_and_store_topics function
async def fetch_and_store_topics():
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
        logger.error(f"Error fetching and storing topics: {str(e)}")
        raise

# Create a scheduler
scheduler = AsyncIOScheduler()

# Modify the startup event to add the scheduled task using the fastapi_app instance
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

# Note: Ensure openai_client and collection are defined elsewhere in your code
# Example initialization (uncomment and configure as needed):
# openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# mongo_client = MongoClient(os.getenv("MONGO_URI"), server_api=ServerApi('1'))
# db = mongo_client["your_database"]
# collection = db["topics"]

if __name__ == "__main__":
    uvicorn.run(fastapi_app, host="0.0.0.0", port=8000)
