# worker.py — NOT USED
#
# This file was previously used to run background document processing jobs
# via Redis Queue (RQ). Document processing is now handled synchronously
# inside the Flask request in app.py, so this worker is no longer needed.
#
# Redis and RQ dependencies have been removed from requirements.txt.
