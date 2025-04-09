# Medical Content Generator Webapp

This application generates medical content based on user input and various data sources, including web scraping, document processing, and API results.

## Features

- Generate medical articles based on user input
- Fetch research data from Perplexity API
- Extract content from web URLs using Firecrawl
- Process document files and YouTube links
- Validate generated content for medical accuracy
- Integrated with the Medical Content Scheduler for trending topics

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Create a `.env` file with the following variables:
   ```
   OPENAI_API_KEY=your_openai_api_key
   PERPLEXITY_API_KEY=your_perplexity_api_key
   FIRECRAWL_API_KEY=your_firecrawl_api_key
   ```

3. Run the application:
   ```
   uvicorn main:app --reload
   ```

## API Endpoints

- `POST /generate-article` - Generate a medical article based on user input
- `GET /get-topics` - Get trending topics (proxied from the scheduler service)
- `POST /extract` - Extract content from a URL using Firecrawl
- `GET /health` - Health check endpoint

## Deployment on Digital Ocean App Platform

1. Connect your GitHub repository to Digital Ocean App Platform
2. Select the webapp folder as the source directory
3. Configure environment variables (see above)
4. Deploy the application

## Environment Variables

- `OPENAI_API_KEY`: Required for GPT-4o API calls
- `PERPLEXITY_API_KEY`: Required for Perplexity API calls
- `FIRECRAWL_API_KEY`: Required for Firecrawl web scraping

