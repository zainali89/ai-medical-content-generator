web: uvicorn main:fastapi_app --host=0.0.0.0 --port=$PORT
worker: celery -A main.celery_app worker --loglevel=info
beat: celery -A main.celery_app beat --loglevel=info
