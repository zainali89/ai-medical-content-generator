# AI Medical Content Generator

An AI-powered pipeline that researches medical topics from multiple sources and generates referenced, audience-tailored articles. It combines data from web search (Perplexity), web scraping (Firecrawl), uploaded documents (PDF/DOCX), and YouTube transcripts, then uses Google Gemini to draft the article and OpenAI GPT-4o-mini to validate it.

## Architecture

```
                         +---------------------+
                         |   FastAPI (main.py)  |
                         +----------+----------+
                                    |
                           routes.py (handlers)
                                    |
                    +---------------+---------------+
                    |       pipeline.py (LangGraph) |
                    +---------------+---------------+
                                    |
          +------------+------------+------------+------------+
          |            |            |            |            |
   process_user   search_     extract_     process_    process_
     _input      perplexity  firecrawl     _docs     youtube_links
          |            |            |            |            |
          +------------+-----+------+------------+------------+
                             |
                    check_data_availability
                             |
                     generate_content  (Gemini)
                             |
                     validate_content  (OpenAI)
                             |
                           [END]
```

**Module breakdown:**

| File | Responsibility |
|---|---|
| `main.py` | FastAPI app creation, middleware, route registration, dev server |
| `routes.py` | HTTP endpoint handlers |
| `pipeline.py` | LangGraph workflow: node functions, graph wiring |
| `models.py` | Pydantic request schemas and LangGraph `State` definition |
| `config.py` | Environment variable loading, logging setup, API client init |
| `utils.py` | Shared helpers (timing decorator, YouTube transcript fetcher) |

## Tech Stack

- **Framework:** FastAPI + Uvicorn
- **Orchestration:** LangGraph (state-machine workflow)
- **Content generation:** Google Gemini 2.5 Flash
- **Content validation:** OpenAI GPT-4o-mini
- **Web research:** Perplexity API (sonar-reasoning model)
- **Web scraping:** Firecrawl
- **Document parsing:** PyPDF2, python-docx
- **YouTube transcripts:** RapidAPI YouTube Transcripts
- **Deployment:** Docker, Heroku-compatible (Procfile + Aptfile)

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/<your-org>/ai-medical-content-generator.git
cd ai-medical-content-generator
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure environment variables

```bash
cp .env.example .env
# Edit .env and fill in your API keys
```

Required keys:

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | OpenAI API key (used for content validation via GPT-4o-mini) |
| `PERPLEXITY_API_KEY` | Perplexity API key (medical literature search) |
| `FIRECRAWL_API_KEY` | Firecrawl API key (web page scraping) |
| `GEMINI_API_KEY` | Google Gemini API key (article generation) |
| `RAPIDAPI_KEY` | *(optional)* RapidAPI key for YouTube transcript fetching |
| `SCHEDULER_API_URL` | *(optional)* URL of the companion scheduler service (default: `http://localhost:8000`) |

### 4. Run the application

```bash
uvicorn main:app --reload
```

The server starts at `http://localhost:8000`.

## API Endpoints

### `POST /generate-article`

Generate a medical article. Accepts a JSON body:

```json
{
  "user_input_topic": "Type 2 Diabetes Management",
  "user_input_description": "Latest treatment guidelines and lifestyle interventions",
  "article_length": "Medium",
  "target_audience": "Medical Professionals (Doctors)",
  "reference_urls": ["https://example.com/source"],
  "docs_files": [],
  "youtube_links": []
}
```

- `article_length`: `"Short"` (~800 words), `"Medium"` (~1200 words), or `"Long"` (~2200 words)
- `target_audience`: `"Medical Professionals (Doctors)"`, `"Students"`, `"General Public"`, or `"Patients"`

### `POST /extract`

Extract content from a single URL via Firecrawl.

```json
{ "url": "https://example.com/article" }
```

### `GET /get-topics`

Proxy to the companion scheduler service for trending medical topics.

### `GET /health`

Returns `{"status": "healthy"}`.

## Docker

```bash
docker build -t ai-medical-content-generator .
docker run -p 8000:8000 --env-file .env ai-medical-content-generator
```

## License

See repository for license details.
