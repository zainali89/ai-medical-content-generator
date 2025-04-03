from fastapi import FastAPI, HTTPException
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.proxies import WebshareProxyConfig
from pydantic import BaseModel
from typing import List, Dict
import uvicorn
import os

# Create FastAPI instance with the name fastapi_app
fastapi_app = FastAPI(title="YouTube Transcript API", description="Fetch YouTube video transcripts")

# Define a response model for the transcript
class TranscriptEntry(BaseModel):
    start: float
    duration: float
    text: str

class TranscriptResponse(BaseModel):
    transcript: List[TranscriptEntry]

# Reusable main function to fetch transcript
def fetch_transcript(video_id: str) -> List[Dict]:
    """
    Fetch the transcript for a given YouTube video ID using Webshare proxies.
    
    Args:
        video_id (str): The YouTube video ID (e.g., 'HFfXvfFe9F8')
    
    Returns:
        List[Dict]: List of transcript entries with start, duration, and text
    
    Raises:
        Exception: If transcript fetching fails
    """
    try:
        # Hardcode Webshare proxy credentials
        proxy_username = "pgdvzgig"
        proxy_password = "d55gle2cxicz"

        ytt_api = YouTubeTranscriptApi(
            proxy_config=WebshareProxyConfig(
                proxy_username=proxy_username,
                proxy_password=proxy_password,
            )
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
