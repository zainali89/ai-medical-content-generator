# main.py
from fastapi import FastAPI, HTTPException
from youtube_transcript_api import YouTubeTranscriptApi
from pydantic import BaseModel
from typing import List, Dict
import uvicorn

# Create FastAPI instance
app = FastAPI(title="YouTube Transcript API", description="Fetch YouTube video transcripts")

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
    Fetch the transcript for a given YouTube video ID.
    
    Args:
        video_id (str): The YouTube video ID (e.g., 'HFfXvfFe9F8')
    
    Returns:
        List[Dict]: List of transcript entries with start, duration, and text
    
    Raises:
        Exception: If transcript fetching fails
    """
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return transcript
    except Exception as e:
        raise Exception(f"Error fetching transcript: {str(e)}")

# Define the root endpoint
@app.get("/")
async def root():
    return {"message": "Welcome to the YouTube Transcript API. Use /transcript/{video_id} to fetch a transcript."}

# Define the transcript endpoint
@app.get("/transcript/{video_id}", response_model=TranscriptResponse)
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
    """
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
