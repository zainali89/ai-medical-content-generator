"""
FastAPI route handlers.
"""

import logging
import time
import traceback

import requests
from fastapi import HTTPException

from config import SCHEDULER_API_URL, firecrawl_client
from models import UrlRequest
from pipeline import langgraph_app

logger = logging.getLogger("ai_medical")


async def get_topics():
    """Proxy trending topics from the scheduler service."""
    try:
        logger.info(f"Fetching topics from scheduler at {SCHEDULER_API_URL}/get-topics")
        response = requests.get(f"{SCHEDULER_API_URL}/get-topics")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Error fetching topics from scheduler: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching topics from scheduler: {e}")


async def generate_article(request: dict):
    """Run the full article-generation pipeline."""
    if not request.get("user_input_topic"):
        logger.error("Missing required field: user_input_topic")
        raise HTTPException(status_code=400, detail="Missing required field: user_input_topic")

    initial_state = {
        "user_input_topic": request.get("user_input_topic", ""),
        "user_input_description": request.get("user_input_description", ""),
        "article_length": request.get("article_length", "Short"),
        "target_audience": request.get("target_audience", "general"),
        "reference_urls": request.get("reference_urls", []),
        "docs_files": request.get("docs_files", []),
        "youtube_links": request.get("youtube_links", []),
        "perplexity_data": [],
        "firecrawl_data": [],
        "docs_data": [],
        "youtube_data": [],
        "generated_content": "",
        "errors": [],
        "performance_metrics": {},
        "critical_error": False,
        "skip_perplexity": False,
    }
    logger.info("Starting workflow execution")
    start_time = time.time()
    try:
        final_state = langgraph_app.invoke(initial_state)
        total_time = time.time() - start_time
        final_state["performance_metrics"]["total_execution_time"] = total_time
        logger.info(f"Total execution time: {total_time:.4f} seconds")
        if final_state["errors"]:
            raise HTTPException(
                status_code=500,
                detail={"detail": f"Errors occurred: {final_state['errors']}", "status": 500},
            )
        return {
            "generated_content": final_state["generated_content"],
            "performance_metrics": final_state["performance_metrics"],
            "errors": final_state["errors"],
        }
    except HTTPException:
        raise
    except Exception as e:
        total_time = time.time() - start_time
        logger.error(f"Workflow failed: {e} - Total time: {total_time:.4f} seconds")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail={"detail": f"Internal server error: {e}", "status": 500},
        )


async def extract_content(request: UrlRequest):
    """Scrape a single URL via Firecrawl."""
    try:
        scraped = firecrawl_client.scrape_url(request.url)
        if isinstance(scraped, dict):
            content = scraped.get("markdown", "No content found")
        else:
            content = getattr(scraped, "markdown", "No content found")
            if content == "No content found" and hasattr(scraped, "content"):
                content = scraped.content or "No content found"
        return {"url": request.url, "content": content}
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Content extraction failed: {e}")


async def health_check():
    return {"status": "healthy"}
