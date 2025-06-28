# src/services/rag_service.py
import json
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np
from datetime import datetime

import openai
import chromadb
from chromadb.config import Settings
import streamlit as st

from ..config import LLMConfig, EmbeddingConfig, StorageConfig
from ..models.document import Document, DocumentChunk
from ..utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class RetrievalResult:
    """Result of document retrieval."""
    chunks: List[DocumentChunk]
    scores: List[float]
    total_found: int

@dataclass
class RAGResponse:
    """Complete RAG response."""
    answer: str
    sources: List[DocumentChunk]
    retrieval_score: float
    processing_time: float

class EmbeddingService:
    """Service for creating and managing embeddings."""
    
    def __init__(self, config: EmbeddingConfig, llm_config: LLMConfig):
        self.config = config
        self.llm_config = llm_config
        self.client = openai.OpenAI(api_key=llm_config.openai_api_key)
    
    @st.cache_data(ttl=3600)  # Cache for 1 hour
    def create_embeddings(_self, texts: List[str]) -> List[List[float]]:
        """Create embeddings for a list of texts."""
        try:
            # Process in batches
            all_embeddings = []
            batch_size = _self.config.batch_size
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                
                response = _self.client.embeddings.create(
                    model=_self.config.model_name,
                    input=batch
                )
                
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
            
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Error creating embeddings: {e}")
            raise
    
    def create_embedding(self, text: str) -> List[float]:
        """Create embedding for a single text."""
        return self.create_embeddings([text])[0]

class VectorDatabase:
    """Vector database using ChromaDB."""
    
    def __init__(self, storage_config: StorageConfig, collection_name: str = "datamap_docs"):
        self.storage_config = storage_config
        self.collection_name = collection_name
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path=str(storage_config.vector_db_path),
            settings=Settings(allow_reset=True)
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "DataMap AI document chunks"}
        )
    
    def add_chunks(self, chunks: List[DocumentChunk], embeddings: List[List[float]]):
        """Add document chunks to the vector database."""
        try:
            # Prepare data for ChromaDB
            ids = [chunk.id for chunk in chunks]
            documents = [chunk.content for chunk in chunks]
            metadatas = [
                {
                    **chunk.metadata,
                    "document_id": chunk.document_id,
                    "created_at": datetime.now().isoformat()
                }
                for chunk in chunks
            ]
            
            # Add to collection
            self.collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )
            
            logger.info(f"Added {len(chunks)} chunks to vector database")
            
        except Exception as e:
            logger.error(f"Error adding chunks to vector database: {e}")
            raise
    
    def search(self, query_embedding: List[float], 
               user_id: str, 
               n_results: int = 5,
               document_ids: Optional[List[str]] = None) -> RetrievalResult:
        """Search for similar chunks in the vector database."""
        try:
            # Build where clause for filtering
            where_clause = {"user_id": user_id}
            if document_ids:
                where_clause["document_id"] = {"$in": document_ids}
            
            # Search in collection
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where_clause,
                include=["documents", "metadatas", "distances"]
            )
            
            # Convert results to DocumentChunk objects
            chunks = []
            scores = []
            
            if results['documents'] and results['documents'][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )):
                    chunk = DocumentChunk(
                        id=results['ids'][0][i],
                        document_id=metadata.get('document_id', ''),
                        content=doc,
                        metadata=metadata
                    )
                    chunks.append(chunk)
                    scores.append(1 - distance)  # Convert distance to similarity score
            
            return RetrievalResult(
                chunks=chunks,
                scores=scores,
                total_found=len(chunks)
            )
            
        except Exception as e:
            logger.error(f"Error searching vector database: {e}")
            return RetrievalResult(chunks=[], scores=[], total_found=0)
    
    def delete_document(self, document_id: str):
        """Delete all chunks for a specific document."""
        try:
            self.collection.delete(where={"document_id": document_id})
            logger.info(f"Deleted chunks for document {document_id}")
        except Exception as e:
            logger.error(f"Error deleting document chunks: {e}")
            raise
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the collection."""
        try:
            count = self.collection.count()
            return {
                "total_chunks": count,
                "collection_name": self.collection_name
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {"total_chunks": 0, "collection_name": self.collection_name}

class RAGService:
    """Complete RAG (Retrieval-Augmented Generation) service."""
    
    def __init__(self, llm_config: LLMConfig, embedding_config: EmbeddingConfig, 
                 storage_config: StorageConfig):
        self.llm_config = llm_config
        self.embedding_service = EmbeddingService(embedding_config, llm_config)
        self.vector_db = VectorDatabase(storage_config)
        self.client = openai.OpenAI(api_key=llm_config.openai_api_key)
    
    def add_document(self, document: Document) -> bool:
        """Add a document to the RAG system."""
        try:
            # Create embeddings for all chunks
            texts = [chunk.content for chunk in document.chunks]
            embeddings = self.embedding_service.create_embeddings(texts)
            
            # Add to vector database
            self.vector_db.add_chunks(document.chunks, embeddings)
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding document to RAG system: {e}")
            return False
    
    def query(self, question: str, user_id: str, 
              document_ids: Optional[List[str]] = None,
              max_chunks: int = 5) -> RAGResponse:
        """Query the RAG system."""
        start_time = datetime.now()
        
        try:
            # Create embedding for the question
            question_embedding = self.embedding_service.create_embedding(question)
            
            # Retrieve relevant chunks
            retrieval_result = self.vector_db.search(
                query_embedding=question_embedding,
                user_id=user_id,
                n_results=max_chunks,
                document_ids=document_ids
            )
            
            if not retrieval_result.chunks:
                return RAGResponse(
                    answer="I couldn't find any relevant information to answer your question.",
                    sources=[],
                    retrieval_score=0.0,
                    processing_time=(datetime.now() - start_time).total_seconds()
                )
            
            # Generate answer using retrieved context
            answer = self._generate_answer(question, retrieval_result.chunks)
            
            # Calculate average retrieval score
            avg_score = sum(retrieval_result.scores) / len(retrieval_result.scores) if retrieval_result.scores else 0.0
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return RAGResponse(
                answer=answer,
                sources=retrieval_result.chunks,
                retrieval_score=avg_score,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Error in RAG query: {e}")
            return RAGResponse(
                answer=f"An error occurred while processing your question: {str(e)}",
                sources=[],
                retrieval_score=0.0,
                processing_time=(datetime.now() - start_time).total_seconds()
            )
    
    def _generate_answer(self, question: str, chunks: List[DocumentChunk]) -> str:
        """Generate answer using LLM with retrieved context."""
        try:
            # Prepare context from retrieved chunks
            context_parts = []
            for i, chunk in enumerate(chunks, 1):
                source_info = f"Source {i}"
                if 'source_type' in chunk.metadata:
                    source_info += f" ({chunk.metadata['source_type']})"
                
                context_parts.append(f"{source_info}:\n{chunk.content}\n")
            
            context = "\n".join(context_parts)
            
            # Create prompt
            prompt = f"""Based on the following context, please answer the question. If the context doesn't contain enough information to answer the question completely, say so and provide whatever relevant information you can find.

Context:
{context}

Question: {question}

Answer: Please provide a comprehensive answer based on the context above. If you reference specific information, mention which source it came from."""
            
            # Generate response
            response = self.client.chat.completions.create(
                model=self.llm_config.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context. Always be accurate and cite your sources when possible."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.llm_config.max_tokens,
                temperature=self.llm_config.temperature
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return f"I encountered an error while generating the answer: {str(e)}"
    
    def delete_document(self, document_id: str):
        """Delete a document from the RAG system."""
        self.vector_db.delete_document(document_id)
    
    def get_stats(self) -> Dict:
        """Get RAG system statistics."""
        return self.vector_db.get_collection_stats()

class VisualizationService:
    """Service for generating visual representations from RAG responses."""
    
    def __init__(self, rag_service: RAGService):
        self.rag_service = rag_service
        self.client = rag_service.client
    
    def generate_flowchart(self, question: str, user_id: str) -> Tuple[str, str]:
        """Generate flowchart from RAG response."""
        try:
            # Get initial RAG response
            rag_response = self.rag_service.query(question, user_id)
            
            if not rag_response.sources:
                return "No relevant information found", ""
            
            # Generate flowchart
            flowchart_prompt = f"""
Based on the following information, create a directed graph representation in DOT notation.

Content: {rag_response.answer}

Requirements:
1. Create a flowchart that represents the main concepts and their relationships
2. Use proper DOT notation syntax
3. Maximum 10 nodes to keep it readable
4. Focus on the most important concepts and relationships
5. Return ONLY the DOT notation, no explanations

Example format:
digraph {{
    "Concept A" -> "Concept B"
    "Concept B" -> "Concept C"
    "Concept A" -> "Concept D"
}}
"""
            
            response = self.client.chat.completions.create(
                model=self.rag_service.llm_config.model_name,
                messages=[
                    {"role": "system", "content": "You are an expert at creating clear, informative flowcharts in DOT notation."},
                    {"role": "user", "content": flowchart_prompt}
                ],
                max_tokens=1000,
                temperature=0.1
            )
            
            flowchart_dot = response.choices[0].message.content.strip()
            
            return rag_response.answer, flowchart_dot
            
        except Exception as e:
            logger.error(f"Error generating flowchart: {e}")
            return f"Error generating flowchart: {str(e)}", ""
    
    def generate_mindmap(self, question: str, user_id: str) -> Tuple[str, List[Dict]]:
        """Generate mindmap from RAG response."""
        try:
            # Get initial RAG response
            rag_response = self.rag_service.query(question, user_id)
            
            if not rag_response.sources:
                return "No relevant information found", []
            
            # Generate mindmap structure
            mindmap_prompt = f"""
Based on the following information, create a mindmap structure.

Content: {rag_response.answer}

Requirements:
1. Extract the main topic and subtopics
2. Create a hierarchical structure
3. Maximum 15 nodes total
4. Return as a JSON list of connections in this format:
[
    {{"source": "Main Topic", "target": "Subtopic 1"}},
    {{"source": "Main Topic", "target": "Subtopic 2"}},
    {{"source": "Subtopic 1", "target": "Detail 1"}},
    ...
]

Return ONLY the JSON array, no explanations.
"""
            
            response = self.client.chat.completions.create(
                model=self.rag_service.llm_config.model_name,
                messages=[
                    {"role": "system", "content": "You are an expert at creating clear, informative mindmaps. Always return valid JSON."},
                    {"role": "user", "content": mindmap_prompt}
                ],
                max_tokens=1000,
                temperature=0.1
            )
            
            mindmap_json = response.choices[0].message.content.strip()
            
            try:
                mindmap_connections = json.loads(mindmap_json)
            except json.JSONDecodeError:
                # Fallback to simple structure
                mindmap_connections = [
                    {"source": "Main Topic", "target": "Key Point 1"},
                    {"source": "Main Topic", "target": "Key Point 2"}
                ]
            
            return rag_response.answer, mindmap_connections
            
        except Exception as e:
            logger.error(f"Error generating mindmap: {e}")
            return f"Error generating mindmap: {str(e)}", []
