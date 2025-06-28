# src/services/embedding_service.py
import time
from typing import List, Dict, Optional
from dataclasses import dataclass
import numpy as np
from datetime import datetime

import openai
import streamlit as st

from ..config import EmbeddingConfig, LLMConfig
from ..models.document import Document, DocumentChunk
from ..utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class EmbeddingResult:
    """Result of embedding operation."""
    embeddings: List[List[float]]
    processing_time: float
    success: bool
    error: Optional[str] = None

class EmbeddingService:
    """Service for creating and managing embeddings."""
    
    def __init__(self, embedding_config: EmbeddingConfig, llm_config: LLMConfig):
        """Initialize with embedding and LLM configs."""
        self.config = embedding_config
        self.llm_config = llm_config
        self.client = openai.OpenAI(api_key=llm_config.openai_api_key)
    
    @st.cache_data(ttl=3600)  # Cache for 1 hour
    def create_embeddings(_self, texts: List[str]) -> List[List[float]]:
        """Create embeddings for a list of texts."""
        try:
            if not texts:
                return []
            
            # Process in batches
            all_embeddings = []
            batch_size = min(_self.config.batch_size, len(texts))
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                
                # Clean texts
                cleaned_batch = [_self._clean_text(text) for text in batch]
                
                response = _self.client.embeddings.create(
                    model=_self.config.model_name,
                    input=cleaned_batch
                )
                
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
            
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Error creating embeddings: {e}")
            raise
    
    def create_embedding(self, text: str) -> List[float]:
        """Create embedding for a single text."""
        result = self.create_embeddings([text])
        return result[0] if result else []
    
    def _clean_text(self, text: str) -> str:
        """Clean text for embedding creation."""
        if not text or not text.strip():
            return "empty content"
        
        # Remove excessive whitespace
        cleaned = " ".join(text.split())
        
        # Ensure minimum length
        if len(cleaned) < 3:
            return f"short content: {cleaned}"
        
        # Truncate if too long
        max_length = 8000
        if len(cleaned) > max_length:
            cleaned = cleaned[:max_length] + "..."
        
        return cleaned
    
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
                embeddings = self.create_embeddings(texts)
            
            if not embeddings:
                logger.error(f"Failed to create embeddings for document {document.metadata.id}")
                return False
            
            # Store embeddings in chunks
            for chunk, embedding in zip(document.chunks, embeddings):
                chunk.embedding = embedding
            
            logger.info(f"Created {len(embeddings)} embeddings for document {document.metadata.title}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding document to embedding service: {e}")
            return False
