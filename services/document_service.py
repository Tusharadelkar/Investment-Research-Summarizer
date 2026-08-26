import os
import re
import time
import base64
import json
from pathlib import Path

# fitz is PyMuPDF, used to load and parse PDF files
import pymupdf as fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Relative imports from current app services
from services.analytics import build_analytics, clean_text
from services.llm_service import LLMService
from services.retrieval import HybridRetriever
from services.database import SessionLocal, DocumentModel, DocumentAnalyticsModel, DocumentOverviewModel


class DocumentService:
    """
    Coordinates PDF processing, storage, querying, and preview generation.
    It manages the lifecycle of documents uploaded to the server.
    """
    def __init__(self):
        """
        Initializes the document service with default configuration settings.
        Sets up the text splitter and establishes the retriever cache.
        NOTE: LLMService is NOT cached here — it is resolved per-request so
        that user-specific API keys stored in the database are always used.
        """
        # In-memory cache for HybridRetriever instances (keyed by document_id).
        # Populated during process() and reloaded lazily from disk after a restart.
        self._retriever_cache: dict = {}
        
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

    def _get_llm(self, user_id=None) -> LLMService:
        """
        Returns a fresh LLMService instance for the given user.
        If the user has a personal API key stored in the database it is used;
        otherwise the service falls back to the environment-variable keys.
        This ensures the key is never stale after a server restart or after
        the user saves/updates their key via the UI settings.
        """
        if user_id is not None:
            try:
                from services.database import SessionLocal, UserModel
                with SessionLocal() as session:
                    user = session.query(UserModel).filter_by(id=user_id).first()
                    if user and user.api_key:
                        return LLMService(api_key=user.api_key)
            except Exception:
                pass  # Fall through to env-based initialisation
        return LLMService()

    def _get_retriever(self, document_id: str) -> HybridRetriever:
        """
        Returns the cached HybridRetriever for document_id, loading it from
        disk (ChromaDB + pickle) on first access after a server restart.
        """
        if document_id not in self._retriever_cache:
            self._retriever_cache[document_id] = HybridRetriever(
                document_id, model_name=self.embedding_model
            )
        return self._retriever_cache[document_id]

    def get(self, document_id):
        """
        Fetches an active DocumentRecord by its document ID.
        """
        with SessionLocal() as session:
            record = session.query(DocumentModel).filter_by(id=document_id).first()
            if record:
                return record.public_payload()
            return None

    @property
    def llm(self):
        """Convenience accessor — returns a fresh env-based LLM (no user context)."""
        return self._get_llm()

    def regenerate_overview(self, document_id):
        """
        Re-runs the LLM overview generation for an existing indexed document.
        Loads chunks from the persisted retriever index and updates the DB overview.
        Uses the document owner's stored API key when available.
        """
        with SessionLocal() as session:
            record = session.query(DocumentModel).filter_by(id=document_id).first()
            if not record:
                raise KeyError("Document not found.")
            analytics_pages = record.analytics.pages
            analytics_words = record.analytics.words
            user_id = record.user_id  # Capture owner id for LLM key resolution

        # Resolve the LLM using the document owner's API key (if any)
        llm = self._get_llm(user_id)

        retriever = self._get_retriever(document_id)

        # Reconstruct chunks from BM25 index for overview generation
        chunks = retriever.child_chunks

        # Minimal analytics dict needed for the overview prompt
        analytics = {"pages": analytics_pages, "words": analytics_words}
        overview = self._build_overview(chunks, retriever, analytics, llm=llm)

        with SessionLocal() as session:
            record = session.query(DocumentModel).filter_by(id=document_id).first()
            if not record:
                raise KeyError("Document not found.")

            if record.overview:
                record.overview.summary = overview["summary"]
                record.overview.note = overview.get("note")
                record.overview.set_json_field("key_findings", overview["key_findings"])
                record.overview.set_json_field("risks", overview["risks"])
                record.overview.set_json_field("opportunities", overview["opportunities"])
            else:
                doc_overview = DocumentOverviewModel(
                    document_id=document_id,
                    summary=overview["summary"],
                    note=overview.get("note"),
                )
                doc_overview.set_json_field("key_findings", overview["key_findings"])
                doc_overview.set_json_field("risks", overview["risks"])
                doc_overview.set_json_field("opportunities", overview["opportunities"])
                record.overview = doc_overview

            record.llm_available = llm.available
            session.commit()
            return record.public_payload()


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
                for chunk_text in self.splitter.split_text(text):
                    if chunk_text.strip():
                        chunks.append(
                            {
                                "text": chunk_text.strip(),
                                "page": page_number,
                                "chunk_id": len(chunks),
                            }
                        )

        # 2. Prevent proceeding if there is no selectable text
        if not chunks:
            raise ValueError(
                "No selectable text was found. This may be a scanned PDF; run OCR first."
            )

        # 3. Create the search index and compute statistics
        retriever = HybridRetriever(document_id, child_chunks=chunks, parent_chunks=chunks, model_name=self.embedding_model)
        analytics = build_analytics(chunks, page_count, image_count, filename)

        # Resolve the LLM for this request (picks up user key from DB if set)
        llm = self._get_llm()

        # 4. Generate structured summary overview via LLM
        overview = self._build_overview(chunks, retriever, analytics, llm=llm)

        # 5. Build the complete document record object
        with SessionLocal() as session:
            doc = DocumentModel(
                id=document_id,
                filename=filename,
                file_path=str(file_path),
                elapsed_seconds=time.perf_counter() - started,
                llm_available=llm.available,
                vision_available=llm.vision_available
            )
            doc.set_visual_pages(visual_pages)
            
            doc_analytics = DocumentAnalyticsModel(
                pages=analytics["pages"],
                words=analytics["words"],
                characters=analytics["characters"],
                chunks=analytics["chunks"],
                images=analytics["images"],
                top_terms=json.dumps(analytics["top_terms"]),
                topics=json.dumps(analytics["topics"]),
                financial_metrics=json.dumps(analytics["financial_metrics"]),
                page_distribution=json.dumps(analytics["page_distribution"])
            )
            doc.analytics = doc_analytics
            
            doc_overview = DocumentOverviewModel(
                summary=overview["summary"],
                note=overview.get("note")
            )
            doc_overview.set_json_field("key_findings", overview["key_findings"])
            doc_overview.set_json_field("risks", overview["risks"])
            doc_overview.set_json_field("opportunities", overview["opportunities"])
            doc.overview = doc_overview
            
            session.add(doc)
            session.commit()
            
            # Cache the retriever so subsequent ask() calls don't reload from disk
            self._retriever_cache[document_id] = retriever
            
            return doc.public_payload()

    def _build_overview(self, chunks, retriever, analytics, llm: LLMService = None):
        """
        Generates a summary overview (executive summary, key findings, risks, opportunities)
        for the uploaded document. Uses search results as context.
        
        `llm` should be passed explicitly so the correct user-scoped LLM instance is used.
        If omitted, a fresh env-based LLMService is created as a fallback.
        """
        if llm is None:
            llm = LLMService()

        # Create a basic textual fallback from the start of the document in case the LLM is offline
        fallback = clean_text(" ".join(chunk["text"] for chunk in chunks[:4]))[:1500]
        if not llm.available:
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
"summary" (a concise paragraph, minimum 2 complete sentences),
"key_findings" (array of up to 5 strings, each a complete factual sentence),
"risks" (array of up to 5 strings, each a complete factual sentence),
"opportunities" (array of up to 5 strings, each a complete factual sentence).

CRITICAL RULES:
- Never use "..." or "…" as a placeholder. Write real content or omit the item.
- Each string must be a complete sentence with real information from the excerpts.
- Do not invent figures. Add page citations like [Page 4] when supported.
- Output ONLY the JSON object, no other text before or after it.

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
            # Use higher max_tokens (4000) to ensure long documents don't truncate JSON outputs
            answer = llm.complete([{"role": "user", "content": prompt}], max_tokens=4000)
            parsed = self._parse_json(answer)
            if parsed and not self._is_placeholder_response(parsed):
                return parsed
            if parsed:
                # JSON parsed but all values are "..." placeholders (thinking model artifact)
                return {
                    "summary": fallback,
                    "key_findings": [],
                    "risks": [],
                    "opportunities": [],
                    "note": "AI returned placeholder values. Click Regenerate to retry.",
                }
            # JSON parsing failed — surface the raw answer as summary and flag for regeneration
            return {
                "summary": answer or fallback,
                "key_findings": [],
                "risks": [],
                "opportunities": [],
                "note": "AI responded but the structured output could not be parsed. Click Regenerate to retry.",
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
        with SessionLocal() as session:
            record = session.query(DocumentModel).filter_by(id=document_id).first()
            if not record:
                raise KeyError("Document not found.")
            visual_pages_from_record = record.get_visual_pages()

        retriever = self._get_retriever(document_id)
        
        # 1. Search the index for passages matching the user's question
        results = retriever.search(question, candidate_k=28, final_k=7)
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
            if page in visual_pages_from_record
        ]
        
        # If the user explicitly asks for visual charts/graphs but no visual pages were matched,
        # fallback to attaching the first few visual pages in the document
        if not visual_pages and self._is_visual_question(question):
            visual_pages = visual_pages_from_record[:3]
            
        # Create preview URL dictionary structures
        visual_sources = [
            {
                "page": page,
                "url": f"/api/documents/{document_id}/pages/{page}/preview",
            }
            for page in visual_pages
        ]

        # Resolve the LLM for this request
        with SessionLocal() as session:
            doc_record = session.query(DocumentModel).filter_by(id=document_id).first()
            _user_id = doc_record.user_id if doc_record else None
        llm = self._get_llm(_user_id)

        # 2. If the LLM service key is not configured, return sources directly with a note
        if not llm.available:
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
        if visual_pages and llm.vision_available:
            images = [
                {
                    "page": page,
                    "data_url": self._page_data_url(document_id, page),
                }
                for page in visual_pages
            ]
            answer = llm.complete_with_images(
                messages[0]["content"],
                question,
                context,
                images,
                max_tokens=900,
            )

        # 5. Text-only fallback request
        if not answer:
            answer = llm.complete(messages, max_tokens=900)
            if visual_pages and not llm.vision_available:
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
        with SessionLocal() as session:
            record = session.query(DocumentModel).filter_by(id=document_id).first()
            if not record:
                raise KeyError("Document not found.")
            file_path = Path(record.file_path)
            pages_count = record.analytics.pages
            
        if page_number < 1 or page_number > pages_count:
            raise ValueError("Page number is outside this document.")

        preview_dir = file_path.parent / "previews"
        preview_dir.mkdir(parents=True, exist_ok=True)
        preview_path = preview_dir / f"page_{page_number}.png"
        
        # Return already rendered cached images if they exist
        if preview_path.exists():
            return preview_path

        # Render the PDF page using PyMuPDF (fitz) pixmap representation
        with fitz.open(file_path) as pdf:
            page = pdf[page_number - 1]
            pixmap = page.get_pixmap(matrix=fitz.Matrix(1.45, 1.45), alpha=False)
            pixmap.save(preview_path)
            
        return preview_path

    def _page_data_url(self, document_id, page_number):
        """
        Renders a page, reads the image bytes, base64 encodes it,
        and generates a data URL scheme suitable for LLM vision models.
        """
        preview_path = self.render_page(document_id, page_number)
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
    def _is_placeholder_response(parsed: dict) -> bool:
        """
        Returns True when a parsed overview JSON contains only placeholder '...' / '\u2026' values.
        This happens with Groq Qwen3 (and similar thinking models) when the reasoning tokens
        consume most of max_tokens and the model writes ellipsis filler for the actual answer.
        """
        PLACEHOLDERS = {"", "...", "\u2026", "…"}
        for key in ("summary", "key_findings", "risks", "opportunities"):
            val = parsed.get(key)
            if isinstance(val, list):
                # Flag if EVERY item in the list is a placeholder
                if val and all(str(v).strip() in PLACEHOLDERS for v in val):
                    return True
            elif isinstance(val, str):
                if val.strip() in PLACEHOLDERS:
                    return True
        return False

    @staticmethod
    def _parse_json(text):
        """
        Utility function to isolate and parse a JSON block from LLM text responses.
        Ensures required fields for the document overview are present.
        
        Handles:
        - Thinking-model output: strips <think>...</think> blocks (Qwen3, DeepSeek-R1, etc.)
        - Markdown code fences: extracts JSON from ```json ... ``` blocks
        - Bare JSON objects anywhere in the response text
        """
        if not text:
            return None
        import json

        # 1. Strip thinking-model reasoning blocks before attempting JSON extraction
        cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

        # 2. Prefer JSON inside a markdown code fence (``` json ... ``` or ``` ... ```)
        code_fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
        if code_fence:
            candidate = code_fence.group(1)
        else:
            # 3. Fall back to the outermost bare JSON object in the text
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if not match:
                return None
            candidate = match.group()

        try:
            parsed = json.loads(candidate)
            required = {"summary", "key_findings", "risks", "opportunities"}
            # Ensure the required keys exist in the parsed dictionary
            return parsed if required.issubset(parsed) else None
        except json.JSONDecodeError:
            return None

