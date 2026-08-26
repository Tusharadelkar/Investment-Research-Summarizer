import re
import os
import pickle
import numpy as np
from pathlib import Path
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer
import chromadb

def tokenize(text):
    """
    Splits text into a list of lowercase alphanumeric tokens.
    Retains decimal points, percentages, single quotes, and hyphens (e.g. "50%", "e-commerce", "U.S.").
    """
    return re.findall(r"\b[a-zA-Z0-9][a-zA-Z0-9.%'-]*\b", text.lower())

class HybridRetriever:
    """
    Implements a Hybrid Search system combining:
    1. Semantic Search (Dense Embeddings using SentenceTransformers and ChromaDB)
    2. Lexical Search (Keyword Matching using BM25 Okapi)
    3. RRF (Reciprocal Rank Fusion) to combine search rankings.
    4. Optional Deep Reranking (using a CrossEncoder model).
    """
    def __init__(self, document_id, child_chunks=None, parent_chunks=None, model_name=None):
        """
        Initializes the persistent retriever index for a document.
        - Computes and saves dense vector embeddings to ChromaDB.
        - Prepares tokenized chunks and builds the BM25 lexical database, saved via Pickle.
        """
        self.document_id = document_id
        
        if not model_name:
            model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
            
        self.model = SentenceTransformer(model_name)
        
        reranker_name = os.getenv("RERANKER_MODEL", "").strip()
        self.reranker = CrossEncoder(reranker_name) if reranker_name else None
        
        # Set up ChromaDB persistent client
        base_dir = Path(__file__).resolve().parent.parent
        chroma_path = base_dir / "chroma_db"
        self.chroma_client = chromadb.PersistentClient(path=str(chroma_path))
        
        # Collection named after document_id
        self.collection = self.chroma_client.get_or_create_collection(name=f"doc_{document_id}")
        
        bm25_path = base_dir / "uploads" / document_id / "bm25.pkl"
        chunks_path = base_dir / "uploads" / document_id / "chunks.pkl"
        
        if child_chunks is not None and parent_chunks is not None:
            # We are initializing for the first time
            self.child_chunks = child_chunks
            self.parent_chunks = parent_chunks
            
            with open(chunks_path, 'wb') as f:
                pickle.dump({"child": self.child_chunks, "parent": self.parent_chunks}, f)
            
            self.corpus_tokens = [tokenize(chunk["text"]) for chunk in child_chunks]
            self.bm25 = BM25Okapi(self.corpus_tokens)
            
            with open(bm25_path, 'wb') as f:
                pickle.dump(self.bm25, f)
                
            # Add to ChromaDB
            embeddings = self.model.encode(
                [chunk["text"] for chunk in child_chunks],
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            
            ids = [str(chunk["chunk_id"]) for chunk in child_chunks]
            metadatas = [{"page": chunk["page"], "chunk_id": chunk["chunk_id"], "parent_id": chunk.get("parent_id", chunk["chunk_id"])} for chunk in child_chunks]
            documents = [chunk["text"] for chunk in child_chunks]
            
            # Split into batches to avoid potential chroma size limits, though usually small for PDFs
            self.collection.add(
                embeddings=embeddings.tolist(),
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
        else:
            # We are loading an existing document index
            if not bm25_path.exists() or not chunks_path.exists():
                raise FileNotFoundError(f"Search index files missing for document {document_id}")
                
            with open(chunks_path, 'rb') as f:
                loaded_chunks = pickle.load(f)
                if isinstance(loaded_chunks, dict):
                    self.child_chunks = loaded_chunks["child"]
                    self.parent_chunks = loaded_chunks["parent"]
                else:
                    self.child_chunks = loaded_chunks
                    self.parent_chunks = loaded_chunks
                
            with open(bm25_path, 'rb') as f:
                self.bm25 = pickle.load(f)

    def search(self, query, candidate_k=24, final_k=6):
        """
        Runs the hybrid search pipeline:
        1. Encodes query & searches ChromaDB.
        2. Calculates BM25 token matches (lexical scores).
        3. Retrieves top candidate_k matches from both methods.
        4. Merges rankings via Reciprocal Rank Fusion (RRF).
        5. Reranks the top candidates using CrossEncoder (if enabled).
        6. Returns the top final_k results.
        """
        if not self.child_chunks:
            return []

        candidate_k = min(candidate_k, len(self.child_chunks))

        # --- Step 1: Semantic (Dense) Search ---
        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )[0]
        
        chroma_results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=candidate_k
        )
        
        dense_order = [int(i) for i in chroma_results['ids'][0]]

        # --- Step 2: Lexical (Keyword) Search ---
        bm25_scores = self.bm25.get_scores(tokenize(query))
        lexical_order = np.argsort(bm25_scores)[::-1][:candidate_k]

        # --- Step 3: Candidate Selection ---
        candidates = set(map(int, dense_order)) | set(map(int, lexical_order))
        
        dense_rank = {int(idx): rank for rank, idx in enumerate(dense_order, 1)}
        bm25_rank = {int(idx): rank for rank, idx in enumerate(lexical_order, 1)}

        # --- Step 4: Reciprocal Rank Fusion (RRF) ---
        results = []
        for idx in candidates:
            dense_position = dense_rank.get(idx, candidate_k + 1)
            bm25_position = bm25_rank.get(idx, candidate_k + 1)
            
            rrf = 1 / (60 + dense_position) + 1 / (60 + bm25_position)
            
            results.append(
                {
                    **self.child_chunks[idx],
                    "index": idx,
                    "dense_score": float(1.0 / dense_position), # Pseudo-score for UI
                    "bm25_score": float(bm25_scores[idx]),
                    "rrf_score": float(rrf),
                }
            )

        results.sort(key=lambda row: row["rrf_score"], reverse=True)
        shortlist = results[: max(final_k * 2, final_k)]

        # --- Step 5: Resolve to Parent Chunks and Deep Rerank ---
        unique_parents = {}
        for row in shortlist:
            parent_id = row.get("parent_id", row["index"])
            if parent_id not in unique_parents:
                parent = self.parent_chunks[parent_id].copy()
                parent["rrf_score"] = row["rrf_score"]  # inherit score
                parent["dense_score"] = row["dense_score"]
                parent["bm25_score"] = row["bm25_score"]
                unique_parents[parent_id] = parent

        parent_list = list(unique_parents.values())

        if self.reranker and parent_list:
            scores = self.reranker.predict(
                [(query, row["text"]) for row in parent_list],
                show_progress_bar=False,
            )
            for row, score in zip(parent_list, scores):
                row["reranker_score"] = float(score)
            parent_list.sort(key=lambda row: row["reranker_score"], reverse=True)

        return parent_list[:final_k]
