# src/services/embedding_service.py
import asyncio
import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from datetime import datetime
import hashlib
import pickle
from pathlib import Path

import openai
import streamlit as st
from tenacity import retry, stop_after_attempt, wait_exponential

from ..config import EmbeddingConfig, LLMConfig, StorageConfig
from ..models.document import Document, DocumentChunk
from ..utils.logger import get_logger
from ..utils.error_handling import handle_errors, EmbeddingError

logger = get_logger(__name__)

@dataclass
class EmbeddingResult:
    """Result of embedding operation."""
    embeddings: List[List[float]]
    processing_time: float
    token_count: int
    success: bool
    error: Optional[str] = None

@dataclass
class EmbeddingMetrics:
    """Metrics for embedding operations."""
    total_embeddings_created: int = 0
    total_tokens_processed: int = 0
    average_processing_time: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0

class EmbeddingCache:
    """Cache for storing embeddings to avoid recomputation."""
    
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_file = cache_dir / "embeddings_cache.pkl"
        self._cache: Dict[str, List[float]] = {}
        self._load_cache()
    
    def _load_cache(self):
        """Load cache from disk."""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'rb') as f:
                    self._cache = pickle.load(f)
                logger.info(f"Loaded {len(self._cache)} embeddings from cache")
        except Exception as e:
            logger.warning(f"Could not load embedding cache: {e}")
            self._cache = {}
    
    def _save_cache(self):
        """Save cache to disk."""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self._cache, f)
        except Exception as e:
            logger.warning(f"Could not save embedding cache: {e}")
    
    def get_cache_key(self, text: str, model: str) -> str:
        """Generate cache key for text and model."""
        combined = f"{model}:{text}"
        return hashlib.sha256(combined.encode()).hexdigest()
    
    def get(self, text: str, model: str) -> Optional[List[float]]:
        """Get embedding from cache."""
        key = self.get_cache_key(text, model)
        return self._cache.get(key)
    
    def set(self, text: str, model: str, embedding: List[float]):
        """Store embedding in cache."""
        key = self.get_cache_key(text, model)
        self._cache[key] = embedding
        
        # Save cache periodically
        if len(self._cache) % 100 == 0:
            self._save_cache()
    
    def clear(self):
        """Clear the cache."""
        self._cache.clear()
        if self.cache_file.exists():
            self.cache_file.unlink()
    
    def get_stats(self) -> Dict:
        """Get cache statistics."""
        return {
            "cached_embeddings": len(self._cache),
            "cache_size_mb": self.cache_file.stat().st_size / (1024 * 1024) if self.cache_file.exists() else 0
        }

class EmbeddingService:
    """Service for creating and managing embeddings."""
    
    def __init__(self, embedding_config: EmbeddingConfig, llm_config: LLMConfig, 
                 storage_config: StorageConfig):
        self.config = embedding_config
        self.llm_config = llm_config
        self.storage_config = storage_config
        
        # Initialize OpenAI client
        self.client = openai.OpenAI(api_key=llm_config.openai_api_key)
        
        # Initialize cache
        self.cache = EmbeddingCache(storage_config.data_dir / "embeddings_cache")
        
        # Metrics tracking
        self.metrics = EmbeddingMetrics()
        
        # Rate limiting
        self._last_request_time = 0
        self._min_request_interval = 0.1  # 100ms between requests
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    @handle_errors(show_user_error=True, default_return=None)
    def create_embedding(self, text: str) -> Optional[List[float]]:
        """Create embedding for a single text."""
        result = self.create_embeddings([text])
        if result.success and result.embeddings:
            return result.embeddings[0]
        return None
    
    @handle_errors(show_user_error=True, default_return=EmbeddingResult([], 0, 0, False))
    def create_embeddings(self, texts: List[str]) -> EmbeddingResult:
        """Create embeddings for multiple texts with caching and batching."""
        start_time = time.time()
        
        if not texts:
            return EmbeddingResult([], 0, 0, True)
        
        # Check cache for existing embeddings
        embeddings = []
        texts_to_process = []
        indices_to_process = []
        
        for i, text in enumerate(texts):
            cached_embedding = self.cache.get(text, self.config.model_name)
            if cached_embedding:
                embeddings.append(cached_embedding)
                self.metrics.cache_hits += 1
            else:
                embeddings.append(None)  # Placeholder
                texts_to_process.append(text)
                indices_to_process.append(i)
                self.metrics.cache_misses += 1
        
        # Process uncached texts
        if texts_to_process:
            new_embeddings = self._create_embeddings_batch(texts_to_process)
            if new_embeddings:
                # Insert new embeddings into the correct positions
                for idx, embedding in zip(indices_to_process, new_embeddings):
                    embeddings[idx] = embedding
                    # Cache the new embedding
                    self.cache.set(texts_to_process[indices_to_process.index(idx)], 
                                 self.config.model_name, embedding)
        
        # Check if all embeddings were created successfully
        if any(emb is None for emb in embeddings):
            return EmbeddingResult([], 0, 0, False, "Failed to create some embeddings")
        
        processing_time = time.time() - start_time
        token_count = sum(len(text.split()) for text in texts)  # Rough estimate
        
        # Update metrics
        self.metrics.total_embeddings_created += len(embeddings)
        self.metrics.total_tokens_processed += token_count
        self.metrics.average_processing_time = (
            (self.metrics.average_processing_time * (self.metrics.total_embeddings_created - len(embeddings)) + 
             processing_time) / self.metrics.total_embeddings_created
        )
        
        return EmbeddingResult(
            embeddings=embeddings,
            processing_time=processing_time,
            token_count=token_count,
            success=True
        )
    
    def _create_embeddings_batch(self, texts: List[str]) -> Optional[List[List[float]]]:
        """Create embeddings for a batch of texts using OpenAI API."""
        try:
            # Rate limiting
            self._enforce_rate_limit()
            
            # Process in smaller batches if needed
            batch_size = min(self.config.batch_size, len(texts))
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                
                # Clean texts (remove excessive whitespace, ensure non-empty)
                cleaned_batch = [self._clean_text(text) for text in batch]
                
                # Create embeddings
                response = self.client.embeddings.create(
                    model=self.config.model_name,
                    input=cleaned_batch
                )
                
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                
                # Update last request time
                self._last_request_time = time.time()
            
            return all_embeddings
            
        except openai.RateLimitError as e:
            logger.warning(f"Rate limit hit: {e}")
            time.sleep(60)  # Wait 1 minute
            raise EmbeddingError("Rate limit exceeded. Please try again later.")
        
        except openai.AuthenticationError as e:
            logger.error(f"Authentication error: {e}")
            raise EmbeddingError("Invalid API key. Please check your OpenAI API key.")
        
        except Exception as e:
            logger.error(f"Error creating embeddings: {e}")
            raise EmbeddingError(f"Failed to create embeddings: {str(e)}")
    
    def _clean_text(self, text: str) -> str:
        """Clean text for embedding creation."""
        if not text or not text.strip():
            return "empty content"
        
        # Remove excessive whitespace
        cleaned = " ".join(text.split())
        
        # Ensure minimum length
        if len(cleaned) < 3:
            return f"short content: {cleaned}"
        
        # Truncate if too long (model-specific limits)
        max_length = 8000  # Conservative limit for most models
        if len(cleaned) > max_length:
            cleaned = cleaned[:max_length] + "..."
        
        return cleaned
    
    def _enforce_rate_limit(self):
        """Enforce rate limiting between API requests."""
        time_since_last = time.time() - self._last_request_time
        if time_since_last < self._min_request_interval:
            time.sleep(self._min_request_interval - time_since_last)
    
    def add_document(self, document: Document) -> bool:
        """Add a document by creating embeddings for all its chunks."""
        try:
            # Extract text content from chunks
            texts = [chunk.content for chunk in document.chunks]
            
            if not texts:
                logger.warning(f"Document {document.metadata.id} has no chunks")
                return False
            
            # Create embeddings
            with st.spinner(f"Creating embeddings for {document.metadata.title}..."):
                result = self.create_embeddings(texts)
            
            if not result.success:
                logger.error(f"Failed to create embeddings for document {document.metadata.id}")
                return False
            
            # Store embeddings in chunks
            for chunk, embedding in zip(document.chunks, result.embeddings):
                chunk.embedding = embedding
            
            logger.info(f"Created {len(result.embeddings)} embeddings for document {document.metadata.title}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding document to embedding service: {e}")
            return False
    
    def compute_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Compute cosine similarity between two embeddings."""
        try:
            # Convert to numpy arrays
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # Compute cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return dot_product / (norm1 * norm2)
            
        except Exception as e:
            logger.error(f"Error computing similarity: {e}")
            return 0.0
    
    def find_similar_chunks(self, query_embedding: List[float], 
                          chunks: List[DocumentChunk], 
                          top_k: int = 5, 
                          min_similarity: float = 0.1) -> List[Tuple[DocumentChunk, float]]:
        """Find similar chunks based on embedding similarity."""
        try:
            similarities = []
            
            for chunk in chunks:
                if chunk.embedding:
                    similarity = self.compute_similarity(query_embedding, chunk.embedding)
                    if similarity >= min_similarity:
                        similarities.append((chunk, similarity))
            
            # Sort by similarity (descending)
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            return similarities[:top_k]
            
        except Exception as e:
            logger.error(f"Error finding similar chunks: {e}")
            return []
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings for the current model."""
        model_dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536
        }
        return model_dimensions.get(self.config.model_name, 1536)
    
    def validate_embedding(self, embedding: List[float]) -> bool:
        """Validate that an embedding has the correct format and dimension."""
        if not isinstance(embedding, list):
            return False
        
        if len(embedding) != self.get_embedding_dimension():
            return False
        
        if not all(isinstance(x, (int, float)) for x in embedding):
            return False
        
        return True
    
    def get_metrics(self) -> Dict:
        """Get embedding service metrics."""
        cache_stats = self.cache.get_stats()
        
        return {
            "total_embeddings_created": self.metrics.total_embeddings_created,
            "total_tokens_processed": self.metrics.total_tokens_processed,
            "average_processing_time": round(self.metrics.average_processing_time, 3),
            "cache_hits": self.metrics.cache_hits,
            "cache_misses": self.metrics.cache_misses,
            "cache_hit_rate": (
                self.metrics.cache_hits / max(1, self.metrics.cache_hits + self.metrics.cache_misses)
            ),
            **cache_stats
        }
    
    def clear_cache(self):
        """Clear the embedding cache."""
        self.cache.clear()
        logger.info("Embedding cache cleared")
    
    def health_check(self) -> Dict:
        """Perform a health check of the embedding service."""
        try:
            # Test embedding creation with a simple text
            test_text = "This is a test for the embedding service."
            test_embedding = self.create_embedding(test_text)
            
            if test_embedding and self.validate_embedding(test_embedding):
                return {
                    "status": "healthy",
                    "model": self.config.model_name,
                    "embedding_dimension": len(test_embedding),
                    "cache_size": len(self.cache._cache)
                }
            else:
                return {
                    "status": "unhealthy",
                    "error": "Failed to create test embedding"
                }
                
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
