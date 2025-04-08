# worker.py
import asyncio
import logging
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from tasks import fetch_and_store_topics
import pytz

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

async def main():
    # Create scheduler with Australia/Sydney timezone
    scheduler = AsyncIOScheduler(timezone=pytz.timezone("Australia/Sydney"))
    
    # Schedule the task to run every minute
    scheduler.add_job(
        fetch_and_store_topics,
        'interval',
        minutes=1
    )
    
    # Run immediately on startup
    logger.info("Running initial fetch_and_store_topics on startup")
    await fetch_and_store_topics()
    
    # Start the scheduler
    scheduler.start()
    logger.info("Scheduler started, waiting for scheduled tasks")
    
    # Keep the script running
    await asyncio.Event().wait()

if __name__ == "__main__":
    asyncio.run(main())
