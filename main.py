"""
Application entry point.

Creates the FastAPI app, registers routes, and runs the server.
"""

import nest_asyncio
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from models import UrlRequest
from routes import extract_content, generate_article, get_topics, health_check

# Allow nested event loops (needed in some deployment environments)
nest_asyncio.apply()

# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
app = FastAPI(title="AI Medical Content Generator")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Register routes
# ---------------------------------------------------------------------------
app.get("/get-topics")(get_topics)
app.post("/generate-article")(generate_article)
app.post("/extract")(extract_content)
app.get("/health")(health_check)

# ---------------------------------------------------------------------------
# Dev server
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    config = uvicorn.Config(app, host="0.0.0.0", port=8000, loop="asyncio")
    server = uvicorn.Server(config)
    server.run()
