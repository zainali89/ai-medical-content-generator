from fastapi import FastAPI, HTTPException
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.proxies import GenericProxyConfig
from pydantic import BaseModel
from typing import List, Dict
import uvicorn
import os
import requests
import random

# Create FastAPI instance with the name fastapi_app
fastapi_app = FastAPI(title="YouTube Transcript API", description="Fetch YouTube video transcripts")

# Define a response model for the transcript
class TranscriptEntry(BaseModel):
    start: float
    duration: float
    text: str

class TranscriptResponse(BaseModel):
    transcript: List[TranscriptEntry]

# Reusable main function to fetch transcript with proxy rotation
def fetch_transcript(video_id: str) -> List[Dict]:
    """
    Fetch the transcript for a given YouTube video ID using Webshare proxies with rotation.
    
    Args:
        video_id (str): The YouTube video ID (e.g., 'C7OQHIpDlvA')
    
    Returns:
        List[Dict]: List of transcript entries with start, duration, and text
    
    Raises:
        Exception: If transcript fetching fails
    """
    try:
        # Webshare API key (provided by the user)
        WEBSHARE_API_KEY = "5qh3dvjmdskwl9zwxkfe7rylnxdgbwpobi7kkvzp"

        # Fetch proxy list from Webshare
        response = requests.get(
            "https://proxy.webshare.io/api/v2/proxy/list/?mode=direct&page=1&per_page=25",
            headers={"Authorization": f"Token {WEBSHARE_API_KEY}"}
        )
        response.raise_for_status()  # Raise an error for bad responses
        proxy_list = response.json()["results"]

        if not proxy_list:
            raise Exception("No proxies available from Webshare")

        # Select a random proxy from the list
        proxy = random.choice(proxy_list)
        proxy_url = f"http://{proxy['username']}:{proxy['password']}@{proxy['proxy_address']}:{proxy['port']}"

        # Debug: Print the selected proxy (remove in production)
        print(f"Using proxy: {proxy_url}")

        # Initialize YouTubeTranscriptApi with the selected proxy
        ytt_api = YouTubeTranscriptApi(
            proxy_config=GenericProxyConfig(http_url=proxy_url, https_url=proxy_url)
        )
        transcript = ytt_api.get_transcript(video_id)
        return transcript
    except Exception as e:
        raise Exception(f"Error fetching transcript: {str(e)}")

# Define the root endpoint
@fastapi_app.get("/")
async def root():
    return {"message": "Welcome to the YouTube Transcript API. Use /transcript/{video_id} to fetch a transcript."}

# Define the transcript endpoint
@fastapi_app.get("/transcript/{video_id}", response_model=TranscriptResponse)
async def get_transcript(video_id: str):
    try:
        transcript = fetch_transcript(video_id)
        return {"transcript": transcript}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Main function to run the FastAPI app programmatically
def main():
    """
    Run the FastAPI application using Uvicorn programmatically.
    Use the PORT environment variable if available (required for DigitalOcean),
    otherwise default to 8000 for local development.
    """
    port = int(os.environ.get("PORT", 8000))  # Use PORT from env, default to 8000
    uvicorn.run(fastapi_app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()
