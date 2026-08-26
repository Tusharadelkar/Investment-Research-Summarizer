import os
from redis import Redis
from rq import Worker, Queue
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
redis_conn = Redis.from_url(redis_url)

# Ensure database is initialized before workers start processing
from services.database import init_db
init_db()

# We need the global DocumentService for processing tasks
from services.document_service import DocumentService
# Initialize it once in the worker process
document_service = DocumentService()

def process_document_task(document_id, file_path, filename, user_id=None):
    """
    Wrapper function to process a document using the global DocumentService instance.
    This avoids pickling issues with RQ and class methods.
    """
    return document_service.process(document_id, Path(file_path), filename, user_id=user_id)

if __name__ == '__main__':
    worker = Worker(['default'], connection=redis_conn)
    worker.work()
