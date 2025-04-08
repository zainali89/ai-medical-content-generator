web: uvicorn main:fastapi_app --host=0.0.0.0 --port=8080 --workers=1
worker: python worker.py
beat: celery -A main.celery_app beat --loglevel=info
