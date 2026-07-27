# Import standard libraries for file and folder operations and unique identifier generation
import os
import uuid
from pathlib import Path

# Import Flask components and Werkzeug utility to secure filenames
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request, send_file
from werkzeug.utils import secure_filename

# Import the core document service logic
from services.document_service import DocumentService

# Define paths: base directory of the project and the upload directory
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)  # Create uploads directory if it doesn't exist

# Load environment variables from the .env file in the base directory
load_dotenv(BASE_DIR / ".env")

# Initialize the Flask application
app = Flask(__name__)

# Configure Flask app settings:
# - Set maximum allowed payload size to 40 MB
# - Set a secret key for session/security, falling back to a development default
app.config.update(
    MAX_CONTENT_LENGTH=40 * 1024 * 1024,
    SECRET_KEY=os.getenv("FLASK_SECRET_KEY", "development-secret-change-me"),
)

# Instantiate the singleton DocumentService to manage PDF processing and retrieval
document_service = DocumentService()


@app.get("/")
def index():
    """
    Route: GET /
    Purpose: Renders the home screen/UI (index.html).
    """
    return render_template("index.html")


@app.post("/api/documents")
def upload_document():
    """
    Route: POST /api/documents
    Purpose: Accepts a PDF upload, validates it, saves it to disk, and triggers
             document indexing/processing. Returns metadata and statistics of the document.
    """
    uploaded = request.files.get("file")
    
    # 1. Validate that a file is selected
    if not uploaded or not uploaded.filename:
        return jsonify({"error": "Select a PDF document first."}), 400

    # 2. Enforce PDF-only uploads
    if Path(uploaded.filename).suffix.lower() != ".pdf":
        return jsonify({"error": "Only PDF documents are supported."}), 400

    # 3. Generate a unique document ID and secure the uploaded filename
    document_id = uuid.uuid4().hex
    filename = secure_filename(uploaded.filename) or "document.pdf"
    
    # 4. Create a folder specific to this document ID under the uploads directory
    document_dir = UPLOAD_DIR / document_id
    document_dir.mkdir(parents=True, exist_ok=True)
    file_path = document_dir / filename
    
    # 5. Save the file to the generated path
    uploaded.save(file_path)

    # 6. Parse and index the document text/pages via DocumentService
    try:
        result = document_service.process(document_id, file_path, filename)
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": f"Document processing failed: {exc}"}), 500


@app.get("/api/documents/<document_id>")
def get_document(document_id):
    """
    Route: GET /api/documents/<document_id>
    Purpose: Retrieves metadata and analytics for a previously uploaded document by its ID.
             Returns 404 if the document ID is not found.
    """
    record = document_service.get(document_id)
    if record is None:
        return jsonify({"error": "Document not found or server was restarted."}), 404
    return jsonify(record.public_payload())


@app.post("/api/documents/<document_id>/ask")
def ask_document(document_id):
    """
    Route: POST /api/documents/<document_id>/ask
    Purpose: Accepts a user question and conversation history, runs a hybrid search over 
             the document's text chunks, feeds relevant context to the LLM, and returns the answer.
    """
    payload = request.get_json(silent=True) or {}
    question = str(payload.get("question", "")).strip()
    history = payload.get("history", [])

    # Validate that a question is provided
    if not question:
        return jsonify({"error": "Enter a question about the document."}), 400

    try:
        # Run retrieval and generate answer via DocumentService
        return jsonify(document_service.ask(document_id, question, history))
    except KeyError:
        return jsonify({"error": "Document not found or server was restarted."}), 404
    except Exception as exc:
        return jsonify({"error": f"Unable to answer the question: {exc}"}), 500


@app.get("/api/documents/<document_id>/pages/<int:page_number>/preview")
def page_preview(document_id, page_number):
    """
    Route: GET /api/documents/<document_id>/pages/<page_number>/preview
    Purpose: Renders a PDF page to a PNG and returns the image, cached for up to an hour.
    """
    try:
        # Request the page image preview path from the DocumentService
        preview_path = document_service.render_page(document_id, page_number)
        return send_file(preview_path, mimetype="image/png", max_age=3600)
    except KeyError:
        return jsonify({"error": "Document not found or server was restarted."}), 404
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 404


@app.errorhandler(413)
def too_large(_error):
    """
    Error Handler: 413 Payload Too Large
    Purpose: Catches files that exceed Flask's configured MAX_CONTENT_LENGTH and returns JSON.
    """
    return jsonify({"error": "The PDF is larger than the 40 MB upload limit."}), 413


# Main entry point to run the Flask application
if __name__ == "__main__":
    app.run(
        host=os.getenv("FLASK_HOST", "127.0.0.1"),
        port=int(os.getenv("FLASK_PORT", "5000")),
        debug=os.getenv("FLASK_DEBUG", "false").lower() == "true",
    )
