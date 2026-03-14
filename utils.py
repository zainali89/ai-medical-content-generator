"""
Shared utility helpers: timing decorator, YouTube transcript fetcher.
"""

import http.client
import json
import logging
import re
import time
from functools import wraps
from typing import Dict, List

from config import RAPIDAPI_KEY

logger = logging.getLogger("ai_medical")


# ---------------------------------------------------------------------------
# Timing decorator
# ---------------------------------------------------------------------------
def timeit(func):
    """Decorator that logs execution time and stores it in the result dict."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start_time
        logger.info(f"{func.__name__} took {duration:.4f} seconds")
        if isinstance(result, dict):
            result.setdefault("performance_metrics", {})
            result["performance_metrics"][func.__name__] = duration
        return result
    return wrapper


# ---------------------------------------------------------------------------
# YouTube helpers
# ---------------------------------------------------------------------------
def extract_video_id(video_url: str) -> str | None:
    """Extract the 11-character video ID from a YouTube URL."""
    regex = (
        r"(?:https?://)?(?:www\.)?(?:youtube\.com/"
        r"(?:[^/]+/.*|(?:v|e(?:mbed)?)|.*[?&]v=)|youtu\.be/)"
        r"([^&]{11})"
    )
    match = re.search(regex, video_url)
    if match:
        return match.group(1)
    logger.error(f"Invalid YouTube URL: {video_url}")
    return None


def get_youtube_transcript(video_urls: List[str]) -> List[Dict]:
    """Fetch transcripts for a list of YouTube video URLs via RapidAPI."""
    if not RAPIDAPI_KEY:
        logger.error("RAPIDAPI_KEY is not set; cannot fetch YouTube transcripts")
        return [
            {"video_url": url, "transcript_text": "", "error": "RAPIDAPI_KEY not configured"}
            for url in video_urls
        ]

    results = []
    for video_url in video_urls:
        video_id = extract_video_id(video_url)
        if not video_id:
            results.append({
                "video_url": video_url,
                "transcript_text": "",
                "error": f"Invalid YouTube URL: {video_url}",
            })
            continue

        conn = http.client.HTTPSConnection("youtube-transcripts.p.rapidapi.com")
        headers = {
            "x-rapidapi-key": RAPIDAPI_KEY,
            "x-rapidapi-host": "youtube-transcripts.p.rapidapi.com",
        }
        request_url = f"/youtube/transcript?url=https://www.youtube.com/watch?v={video_id}"

        try:
            conn.request("GET", request_url, headers=headers)
            res = conn.getresponse()
            if res.status != 200:
                logger.error(f"Error fetching transcript for {video_url}: {res.status} {res.reason}")
                results.append({
                    "video_url": video_url,
                    "transcript_text": "",
                    "error": f"Error fetching transcript: {res.status} {res.reason}",
                })
                continue

            data = res.read()
            transcript_json = json.loads(data.decode("utf-8"))
            all_text = ""
            if "content" in transcript_json:
                for segment in transcript_json["content"]:
                    all_text += segment["text"] + " "

            logger.info(f"Successfully fetched transcript for {video_url}")
            results.append({"video_url": video_url, "transcript_text": all_text.strip(), "error": None})
        except Exception as e:
            logger.error(f"Exception fetching transcript for {video_url}: {e}")
            results.append({"video_url": video_url, "transcript_text": "", "error": str(e)})

    return results
