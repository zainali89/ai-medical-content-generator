from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
fastapi_app = app

class UrlRequest(BaseModel):
    url: str

async def extract_main_content(url: str) -> str:
    try:
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        driver = webdriver.Chrome(options=chrome_options)
        driver.get(url)
        content = driver.find_element(By.CSS_SELECTOR, "div.abstract-content").text
        driver.quit()
        return content
    except Exception as e:
        logger.error(f"Error processing URL {url}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing URL: {str(e)}")

@app.post("/extract-main-content", response_model=dict)
async def extract_main_content_endpoint(request: UrlRequest):
    logger.info(f"Processing request for URL: {request.url}")
    main_content = await extract_main_content(request.url)
    return {"url": request.url, "main_content": main_content}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    host = "0.0.0.0"
    uvicorn.run(app, host=host, port=port, log_level="info")
