import os
import re
import threading
import time
import base64
from dataclasses import dataclass
from pathlib import Path

# fitz is PyMuPDF, used to load and parse PDF files
import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Relative imports from current app services
from services.analytics import build_analytics, clean_text
from services.llm_service import LLMService
from services.retrieval import HybridRetriever


@dataclass
class DocumentRecord:
    """
    Data structure representing a processed document in-memory.
    Holds document IDs, text chunks, cached analytics, vector search indexes,
    and visual page lists.
    """
    document_id: str
    filename: str
    file_path: Path
    chunks: list
    analytics: dict
    retriever: HybridRetriever
    overview: dict
    elapsed_seconds: float
    llm_available: bool
    vision_available: bool
    visual_pages: list
    preview_dir: Path

    def public_payload(self):
        """
        Filters and structures the raw document data to return a clean payload
        for the web interface. Contains references to visual previews up to 18 pages.
        """
        return {
            "document_id": self.document_id,
            "filename": self.filename,
            "analytics": self.analytics,
            "overview": self.overview,
            "processing_seconds": round(self.elapsed_seconds, 2),
            "llm_available": self.llm_available,
            "vision_available": self.retriever is not None and self.vision_available,
            "visual_pages": [
                {
                    "page": page,
                    "url": f"/api/documents/{self.document_id}/pages/{page}/preview",
                }
                for page in self.visual_pages[:18]
            ],
        }


class DocumentService:
    """
    Coordinates PDF processing, storage, querying, and preview generation.
    It manages the lifecycle of documents uploaded to the server.
    """
    def __init__(self):
        """
        Initializes the document service with default configuration settings.
        Sets up the text splitter and establishes the records cache.
        """
        self.records = {}  # In-memory document storage
        self.lock = threading.Lock()  # Ensure thread-safe read/write operations
        self.llm = LLMService()  # Configure the LLM service connection
        
        # Load embedding model and character chunk settings from environment variables
        self.embedding_model = os.getenv(
            "EMBEDDING_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2",
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=int(os.getenv("CHUNK_SIZE", "900")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "150")),
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def get(self, document_id):
        """
        Fetches an active DocumentRecord by its document ID.
        """
        return self.records.get(document_id)

    def process(self, document_id, file_path, filename):
        """
        Extracts, splits, indexes, and analyzes a PDF document.
        Steps:
        1. Open the PDF and extract text/metadata from each page.
        2. Identify visual pages (containing drawings or images).
        3. Split the text into overlapping chunks using RecursiveCharacterTextSplitter.
        4. Initialize the HybridRetriever and compile statistical analytics.
        5. Build an AI-based overview of the document (summary, findings, risks, opportunities).
        6. Cache the DocumentRecord in-memory.
        """
        started = time.perf_counter()
        chunks = []
        image_count = 0
        visual_pages = []

        # 1. Parse the PDF layout and text using fitz (PyMuPDF)
        with fitz.open(file_path) as pdf:
            page_count = pdf.page_count
            for page_index, page in enumerate(pdf):
                page_number = page_index + 1
                text = clean_text(page.get_text("text"))
                page_images = page.get_images(full=True)
                image_count += len(page_images)
                
                # Check for drawing shapes (curves/charts/lines)
                drawing_count = len(page.get_drawings())
                # If page contains images or complex graphics, label it as a visual page
                if page_images or drawing_count >= 4:
                    visual_pages.append(page_number)
                
                if not text:
                    continue

                # Split page text into manageable chunks
                for chunk_index, chunk_text in enumerate(self.splitter.split_text(text)):
                    if chunk_text.strip():
                        chunks.append(
                            {
                                "text": chunk_text.strip(),
                                "page": page_number,
                                "chunk_id": chunk_index,
                            }
                        )

        # 2. Prevent proceeding if there is no selectable text
        if not chunks:
            raise ValueError(
                "No selectable text was found. This may be a scanned PDF; run OCR first."
            )

        # 3. Create the search index and compute statistics
        retriever = HybridRetriever(chunks, self.embedding_model)
        analytics = build_analytics(chunks, page_count, image_count, filename)
        
        # 4. Generate structured summary overview via LLM
        overview = self._build_overview(chunks, retriever, analytics)

        # 5. Build the complete document record object
        record = DocumentRecord(
            document_id=document_id,
            filename=filename,
            file_path=file_path,
            chunks=chunks,
            analytics=analytics,
            retriever=retriever,
            overview=overview,
            elapsed_seconds=time.perf_counter() - started,
            llm_available=self.llm.available,
            vision_available=self.llm.vision_available,
            visual_pages=visual_pages,
            preview_dir=file_path.parent / "previews",
        )
        
        # 6. Save the record in the thread-safe repository dictionary
        with self.lock:
            self.records[document_id] = record
            
        return record.public_payload()

    def _build_overview(self, chunks, retriever, analytics):
        """
        Generates a summary overview (executive summary, key findings, risks, opportunities)
        for the uploaded document. Uses search results as context.
        """
        # Create a basic textual fallback from the start of the document in case the LLM is offline
        fallback = clean_text(" ".join(chunk["text"] for chunk in chunks[:4]))[:1500]
        if not self.llm.available:
            return {
                "summary": fallback,
                "key_findings": [],
                "risks": [],
                "opportunities": [],
                "note": "Configure an LLM API key to generate structured insights.",
            }

        # Define specialized prompts for each section of the overview
        prompts = {
            "Executive overview": "Summarize the document's purpose, major findings and conclusion.",
            "Key findings": "List the most important findings or claims.",
            "Risks": "List material risks, constraints, uncertainties or negative signals.",
            "Opportunities": "List opportunities, growth drivers or favorable signals.",
        }
        
        # Search the document index to gather text fragments relevant to each prompt
        contexts = {}
        for label, query in prompts.items():
            results = retriever.search(query, candidate_k=30, final_k=8)
            contexts[label] = "\n\n".join(
                f"[Page {row['page']}] {row['text']}" for row in results
            )

        # Construct the final prompt requesting JSON formatting
        prompt = f"""
You are analysing an investment or research PDF. Use only the supplied excerpts.
Return valid JSON with exactly these keys:
"summary" (a concise paragraph),
"key_findings" (array of up to 5 strings),
"risks" (array of up to 5 strings),
"opportunities" (array of up to 5 strings).
Do not invent figures. Add page citations like [Page 4] when supported.

Document statistics: {analytics['pages']} pages, {analytics['words']} words.

EXECUTIVE OVERVIEW EXCERPTS
{contexts['Executive overview']}

KEY FINDINGS EXCERPTS
{contexts['Key findings']}

RISKS EXCERPTS
{contexts['Risks']}

OPPORTUNITIES EXCERPTS
{contexts['Opportunities']}
"""
        try:
            # Query the LLM and parse response
            answer = self.llm.complete([{"role": "user", "content": prompt}], max_tokens=1100)
            parsed = self._parse_json(answer)
            if parsed:
                return parsed
            return {
                "summary": answer or fallback,
                "key_findings": [],
                "risks": [],
                "opportunities": [],
            }
        except Exception as exc:
            return {
                "summary": fallback,
                "key_findings": [],
                "risks": [],
                "opportunities": [],
                "note": f"LLM overview generation failed: {exc}",
            }

    def ask(self, document_id, question, history):
        """
        Retrieves matching excerpts and answers a user question about the document.
        - Uses hybrid search to extract matching text excerpts.
        - Decides if visual page previews are needed based on question context.
        - Encodes PDF pages to base64 images if multimodal/vision is supported.
        - Returns a structured answer, text sources, and visual previews.
        """
        record = self.records[document_id]
        
        # 1. Search the index for passages matching the user's question
        results = record.retriever.search(question, candidate_k=28, final_k=7)
        sources = [
            {
                "page": row["page"],
                "chunk_id": row["chunk_id"],
                "excerpt": row["text"][:420],
                "dense_score": round(row["dense_score"], 4),
                "bm25_score": round(row["bm25_score"], 4),
                "rrf_score": round(row["rrf_score"], 6),
            }
            for row in results
        ]
        
        # Identify pages of the top search results
        relevant_pages = list(dict.fromkeys(row["page"] for row in results))[:3]
        
        # Filter visual pages out of these top relevant pages
        visual_pages = [
            page for page in relevant_pages
            if page in record.visual_pages
        ]
        
        # If the user explicitly asks for visual charts/graphs but no visual pages were matched,
        # fallback to attaching the first few visual pages in the document
        if not visual_pages and self._is_visual_question(question):
            visual_pages = record.visual_pages[:3]
            
        # Create preview URL dictionary structures
        visual_sources = [
            {
                "page": page,
                "url": f"/api/documents/{document_id}/pages/{page}/preview",
            }
            for page in visual_pages
        ]

        # 2. If the LLM service key is not configured, return sources directly with a note
        if not self.llm.available:
            return {
                "answer": (
                    "The retrieval layer is working, but no LLM API key is configured. "
                    "The most relevant document excerpts are shown under Sources."
                ),
                "sources": sources,
                "visual_sources": visual_sources,
            }

        # 3. Format context excerpts and chat history safely
        context = "\n\n".join(
            f"[Page {row['page']}, chunk {row['chunk_id']}] {row['text']}"
            for row in results
        )
        safe_history = []
        for item in history[-6:]:  # Limit history to the last 6 messages
            role = item.get("role")
            content = str(item.get("content", ""))[:2000]
            if role in {"user", "assistant"} and content:
                safe_history.append({"role": role, "content": content})

        messages = [
            {
                "role": "system",
                "content": (
                    "Answer questions using only the current PDF excerpts. "
                    "Cite evidence as [Page X]. If the excerpts are insufficient, say so clearly. "
                    "Never use outside facts or fabricate numbers."
                ),
            },
            *safe_history,
            {
                "role": "user",
                "content": f"Question: {question}\n\nDocument excerpts:\n{context}",
            },
        ]
        
        answer = None
        # 4. If visual context is present and multimodal vision is supported, send images to the model
        if visual_pages and self.llm.vision_available:
            images = [
                {
                    "page": page,
                    "data_url": self._page_data_url(record, page),
                }
                for page in visual_pages
            ]
            answer = self.llm.complete_with_images(
                messages[0]["content"],
                question,
                context,
                images,
                max_tokens=900,
            )

        # 5. Text-only fallback request
        if not answer:
            answer = self.llm.complete(messages, max_tokens=900)
            if visual_pages and not self.llm.vision_available:
                answer += (
                    "\n\nRelevant visual pages are attached below. "
                    "Configure VISION_MODEL (or GROQ_VISION_MODEL) for direct chart interpretation."
                )
                
        return {
            "answer": answer,
            "sources": sources,
            "visual_sources": visual_sources,
        }

    def render_page(self, document_id, page_number):
        """
        Renders a PDF page to a PNG image on disk.
        Returns the absolute filepath of the generated image.
        Uses scaling matrix (1.45x) to guarantee quality.
        """
        record = self.records[document_id]
        if page_number < 1 or page_number > record.analytics["pages"]:
            raise ValueError("Page number is outside this document.")

        record.preview_dir.mkdir(parents=True, exist_ok=True)
        preview_path = record.preview_dir / f"page_{page_number}.png"
        
        # Return already rendered cached images if they exist
        if preview_path.exists():
            return preview_path

        # Render the PDF page using PyMuPDF (fitz) pixmap representation
        with fitz.open(record.file_path) as pdf:
            page = pdf[page_number - 1]
            pixmap = page.get_pixmap(matrix=fitz.Matrix(1.45, 1.45), alpha=False)
            pixmap.save(preview_path)
            
        return preview_path

    def _page_data_url(self, record, page_number):
        """
        Renders a page, reads the image bytes, base64 encodes it,
        and generates a data URL scheme suitable for LLM vision models.
        """
        preview_path = self.render_page(record.document_id, page_number)
        encoded = base64.b64encode(preview_path.read_bytes()).decode("ascii")
        return f"data:image/png;base64,{encoded}"

    @staticmethod
    def _is_visual_question(question):
        """
        Determines if the question refers to visual charts, graphs, or drawings.
        """
        return bool(
            re.search(
                r"\b(chart|charts|graph|graphs|figure|figures|image|images|diagram|"
                r"plot|table|visual|trend line|bar chart|pie chart)\b",
                question,
                re.IGNORECASE,
            )
        )

    @staticmethod
    def _parse_json(text):
        """
        Utility function to isolate and parse a JSON block from LLM text responses.
        Ensures required fields for the document overview are present.
        """
        if not text:
            return None
        import json

        # Capture text inside curly brackets
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        try:
            parsed = json.loads(match.group())
            required = {"summary", "key_findings", "risks", "opportunities"}
            # Ensure the required keys exist in the parsed dictionary
            return parsed if required.issubset(parsed) else None
        except json.JSONDecodeError:
            return None

