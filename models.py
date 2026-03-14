"""
Pydantic request models and LangGraph state definition.
"""

from typing import TypedDict, List, Dict, Annotated
import operator

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# LangGraph state
# ---------------------------------------------------------------------------
class State(TypedDict):
    user_input_topic: str
    user_input_description: str
    article_length: str
    target_audience: str
    perplexity_data: List[str]
    firecrawl_data: List[str]
    reference_urls: List[str]
    docs_files: List[str]
    youtube_links: List[str]
    docs_data: List[str]
    youtube_data: List[str]
    generated_content: str
    errors: Annotated[List[str], operator.add]
    performance_metrics: Annotated[Dict[str, float], lambda x, y: {**x, **y}]
    critical_error: Annotated[bool, lambda x, y: x or y]
    skip_perplexity: bool


# ---------------------------------------------------------------------------
# API request schemas
# ---------------------------------------------------------------------------
class UrlRequest(BaseModel):
    url: str
