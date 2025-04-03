from fastapi import FastAPI, HTTPException
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.proxies import GenericProxyConfig
from pydantic import BaseModel
from typing import List, Dict
import uvicorn
import os
import requests
import time
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Create FastAPI instance with the name fastapi_app
fastapi_app = FastAPI(title="YouTube Transcript API", description="Fetch YouTube video transcripts")

# Define a response model for the transcript
class TranscriptEntry(BaseModel):
    start: float
    duration: float
    text: str

class TranscriptResponse(BaseModel):
    transcript: List[TranscriptEntry]

# Reusable main function to fetch transcript with Bright Data proxies
def fetch_transcript_with_proxies(video_id: str) -> List[Dict]:
    """
    Fetch the transcript for a given YouTube video ID using Bright Data proxies.
    
    Args:
        video_id (str): The YouTube video ID (e.g., 'HFfXvfFe9F8')
    
    Returns:
        List[Dict]: List of transcript entries with start, duration, and text
    
    Raises:
        Exception: If transcript fetching fails
    """
    try:
        # Bright Data proxy credentials (replace with your actual credentials)
        BRIGHT_DATA_USERNAME = "your-bright-data-username"
        BRIGHT_DATA_PASSWORD = "your-bright-data-password"
        proxy_url = f"http://{BRIGHT_DATA_USERNAME}:{BRIGHT_DATA_PASSWORD}@zproxy.lum-superproxy.io:22225"

        # Debug: Print the proxy being used (remove in production)
        print(f"Using proxy: {proxy_url}")

        # Initialize YouTubeTranscriptApi with the Bright Data proxy
        ytt_api = YouTubeTranscriptApi(
            proxy_config=GenericProxyConfig(http_url=proxy_url, https_url=proxy_url)
        )
        transcript = ytt_api.get_transcript(video_id)
        return transcript
    except Exception as e:
        raise Exception(f"Proxy method failed: {str(e)}")

# Fallback function to fetch transcript using YouTube Data API
def fetch_transcript_with_youtube_api(video_id: str) -> List[Dict]:
    """
    Fetch the transcript for a given YouTube video ID using YouTube Data API.
    
    Args:
        video_id (str): The YouTube video ID (e.g., 'HFfXvfFe9F8')
    
    Returns:
        List[Dict]: List of transcript entries with start, duration, and text
    
    Raises:
        Exception: If transcript fetching fails
    """
    try:
        # YouTube Data API key (replace with your actual API key)
        YOUTUBE_API_KEY = "your-youtube-api-key"

        # Initialize YouTube API client
        youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

        # Get the caption ID for the video (e.g., English captions)
        captions_response = youtube.captions().list(
            part="id",
            videoId=video_id
        ).execute()

        caption_id = None
        for item in captions_response.get("items", []):
            if item["snippet"]["language"] == "en":  # Look for English captions
                caption_id = item["id"]
                break

        if not caption_id:
            raise Exception("No English captions available for this video")

        # Download the captions in SRT format
        caption_download = youtube.captions().download(
            id=caption_id,
            tfmt="srt"
        ).execute()

        # Parse SRT format into transcript entries
        transcript = []
        lines = caption_download.decode("utf-8").split("\n\n")
        for entry in lines:
            if not entry.strip():
                continue
            parts = entry.split("\n")
            if len(parts) < 3:
                continue
            # Parse timestamp (e.g., "00:00:00,000 --> 00:00:03,000")
            timestamp = parts[1].split(" --> ")
            start_time = timestamp[0].replace(",", ".")
            start_seconds = sum(float(x) * 60 ** i for i, x in enumerate(reversed(start_time.split(":"))))
            end_time = timestamp[1].replace(",", ".")
            end_seconds = sum(float(x) * 60 ** i for i, x in enumerate(reversed(end_time.split(":"))))
            duration = end_seconds - start_seconds
            text = " ".join(parts[2:])
            transcript.append({
                "start": start_seconds,
                "duration": duration,
                "text": text
            })

        return transcript
    except HttpError as e:
        raise Exception(f"YouTube API method failed: {str(e)}")

# Main function to fetch transcript with fallback
def fetch_transcript(video_id: str) -> List[Dict]:
    """
    Fetch the transcript for a given YouTube video ID, trying proxies first, then YouTube API.
    
    Args:
        video_id (str): The YouTube video ID (e.g., 'HFfXvfFe9F8')
    
    Returns:
        List[Dict]: List of transcript entries with start, duration, and text
    
    Raises:
        Exception: If both methods fail
    """
    try:
        # Try fetching with Bright Data proxies
        transcript = fetch_transcript_with_proxies(video_id)
        return transcript
    except Exception as proxy_error:
        print(f"Proxy method failed: {proxy_error}")
        try:
            # Fallback to YouTube Data API
            transcript = fetch_transcript_with_youtube_api(video_id)
            return transcript
        except Exception as api_error:
            raise Exception(f"Both methods failed. Proxy error: {proxy_error}, API error: {api_error}")

# Define the root endpoint
@fastapi_app.get("/")
async def root():
    return {"message": "Welcome to the YouTube Transcript API. Use /transcript/{video_id} to fetch a transcript."}

# Define the transcript endpoint with rate limiting
@fastapi_app.get("/transcript/{video_id}", response_model=TranscriptResponse)
async def get_transcript(video_id: str):
    try:
        # Simple rate limiting: wait 1 second between requests
        time.sleep(1)
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
    uvicorn.run(fastapi_app, host="127.0.0.1", port=port)

if __name__ == "__main__":
    main()
