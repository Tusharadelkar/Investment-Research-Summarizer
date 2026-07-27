import re
import os

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer


def tokenize(text):
    """
    Splits text into a list of lowercase alphanumeric tokens.
    Retains decimal points, percentages, single quotes, and hyphens (e.g. "50%", "e-commerce", "U.S.").
    """
    return re.findall(r"\b[a-zA-Z0-9][a-zA-Z0-9.%'-]*\b", text.lower())


class HybridRetriever:
    """
    Implements a Hybrid Search system combining:
    1. Semantic Search (Dense Embeddings using SentenceTransformers)
    2. Lexical Search (Keyword Matching using BM25 Okapi)
    3. RRF (Reciprocal Rank Fusion) to combine search rankings.
    4. Optional Deep Reranking (using a CrossEncoder model).
    """
    def __init__(self, chunks, model_name):
        """
        Initializes the retriever index for a document.
        - Computes dense vector embeddings for all text chunks.
        - Prepares tokenized chunks and builds the BM25 lexical database.
        - Configures the CrossEncoder reranker if RERANKER_MODEL env variable is set.
        """
        self.chunks = chunks
        
        # Load the sentence transformer model for generating embeddings
        self.model = SentenceTransformer(model_name)
        
        # Check if a reranker model is specified in environment variables and load it
        reranker_name = os.getenv("RERANKER_MODEL", "").strip()
        self.reranker = CrossEncoder(reranker_name) if reranker_name else None
        
        # Tokenize chunks for the BM25 keyword matching engine
        self.corpus_tokens = [tokenize(chunk["text"]) for chunk in chunks]
        self.bm25 = BM25Okapi(self.corpus_tokens)
        
        # Pre-compute and normalize dense embeddings for all document chunks to make vector search fast
        self.embeddings = self.model.encode(
            [chunk["text"] for chunk in chunks],
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).astype(np.float32)

    def search(self, query, candidate_k=24, final_k=6):
        """
        Runs the hybrid search pipeline:
        1. Encodes query & calculates cosine similarities (dense scores).
        2. Calculates BM25 token matches (lexical scores).
        3. Retrieves top candidate_k matches from both methods.
        4. Merges rankings via Reciprocal Rank Fusion (RRF).
        5. Reranks the top candidates using CrossEncoder (if enabled).
        6. Returns the top final_k results.
        """
        if not self.chunks:
            return []

        # --- Step 1: Semantic (Dense) Search ---
        # Generate embedding for the search query
        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )[0]
        # Dot product calculates cosine similarity because embeddings are normalized
        dense_scores = self.embeddings @ query_embedding

        # --- Step 2: Lexical (Keyword) Search ---
        # Get BM25 scores for the tokenized query
        bm25_scores = self.bm25.get_scores(tokenize(query))

        # --- Step 3: Candidate Selection ---
        # Find index positions of top candidate_k results for both dense and lexical searches
        candidate_k = min(candidate_k, len(self.chunks))
        dense_order = np.argsort(dense_scores)[::-1][:candidate_k]
        lexical_order = np.argsort(bm25_scores)[::-1][:candidate_k]

        # Combine unique candidate indices from both methods
        candidates = set(map(int, dense_order)) | set(map(int, lexical_order))
        
        # Map indices to their respective ranks (1-based index)
        dense_rank = {int(idx): rank for rank, idx in enumerate(dense_order, 1)}
        bm25_rank = {int(idx): rank for rank, idx in enumerate(lexical_order, 1)}

        # --- Step 4: Reciprocal Rank Fusion (RRF) ---
        results = []
        for idx in candidates:
            # If a candidate didn't make the top list of one search method, penalize it with a rank of candidate_k + 1
            dense_position = dense_rank.get(idx, candidate_k + 1)
            bm25_position = bm25_rank.get(idx, candidate_k + 1)
            
            # Standard RRF formula with a constant parameter of 60
            rrf = 1 / (60 + dense_position) + 1 / (60 + bm25_position)
            
            results.append(
                {
                    **self.chunks[idx],
                    "index": idx,
                    "dense_score": float(dense_scores[idx]),
                    "bm25_score": float(bm25_scores[idx]),
                    "rrf_score": float(rrf),
                }
            )

        # Sort the candidates by RRF score in descending order
        results.sort(key=lambda row: row["rrf_score"], reverse=True)
        
        # Take a shortlist of candidates (at least final_k)
        shortlist = results[: max(final_k * 2, final_k)]

        # --- Step 5: Optional Deep Reranking ---
        # Use CrossEncoder reranker if configured to re-assess the relevance between query and text chunks
        if self.reranker and shortlist:
            scores = self.reranker.predict(
                [(query, row["text"]) for row in shortlist],
                show_progress_bar=False,
            )
            for row, score in zip(shortlist, scores):
                row["reranker_score"] = float(score)
            # Re-sort using the reranker's confidence score
            shortlist.sort(key=lambda row: row["reranker_score"], reverse=True)

        # Return the top final_k results
        return shortlist[:final_k]
