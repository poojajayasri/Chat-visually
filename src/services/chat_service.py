# src/services/chat_service.py
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import json
import uuid

import streamlit as st

from .rag_service import RAGService, VisualizationService
from .document_service import DocumentService
from ..models.chat import ChatMessage, ChatSession, MessageType
from ..utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class ChatResponse:
    """Response from chat service."""
    message: str
    message_type: MessageType
    sources: List[str] = None
    visualization_data: Optional[Dict] = None
    processing_time: float = 0.0
    error: Optional[str] = None

class ConversationMemory:
    """Manages conversation context and memory."""
    
    def __init__(self, max_history: int = 10):
        self.max_history = max_history
        self.conversations: Dict[str, List[ChatMessage]] = {}
    
    def add_message(self, session_id: str, message: ChatMessage):
        """Add a message to conversation history."""
        if session_id not in self.conversations:
            self.conversations[session_id] = []
        
        self.conversations[session_id].append(message)
        
        # Trim history if too long
        if len(self.conversations[session_id]) > self.max_history * 2:  # *2 for user+assistant pairs
            self.conversations[session_id] = self.conversations[session_id][-self.max_history * 2:]
    
    def get_context(self, session_id: str, include_last_n: int = 5) -> str:
        """Get conversation context as a formatted string."""
        if session_id not in self.conversations:
            return ""
        
        messages = self.conversations[session_id][-include_last_n * 2:]  # Get last N exchanges
        
        context_parts = []
        for msg in messages:
            role = "Human" if msg.type == MessageType.USER else "Assistant"
            context_parts.append(f"{role}: {msg.content}")
        
        return "\n".join(context_parts)
    
    def get_messages(self, session_id: str) -> List[ChatMessage]:
        """Get all messages for a session."""
        return self.conversations.get(session_id, [])
    
    def clear_session(self, session_id: str):
        """Clear conversation history for a session."""
        if session_id in self.conversations:
            del self.conversations[session_id]

class ChatService:
    """Main chat service coordinating RAG, visualization, and conversation management."""
    
    def __init__(self, llm_config, embedding_service, document_service):
        self.rag_service = RAGService(llm_config, embedding_service.config, document_service.config)
        self.visualization_service = VisualizationService(self.rag_service)
        self.document_service = document_service
        self.memory = ConversationMemory()
        self.llm_config = llm_config
    
    def create_session(self, user_id: str) -> str:
        """Create a new chat session."""
        session_id = f"{user_id}_{uuid.uuid4().hex[:8]}"
        return session_id
    
    def process_message(self, 
                       session_id: str,
                       user_id: str, 
                       message: str,
                       message_type: MessageType = MessageType.TEXT,
                       document_ids: Optional[List[str]] = None) -> ChatResponse:
        """Process a user message and return appropriate response."""
        
        start_time = datetime.now()
        
        try:
            # Add user message to memory
            user_message = ChatMessage(
                id=str(uuid.uuid4()),
                session_id=session_id,
                content=message,
                type=MessageType.USER,
                timestamp=datetime.now(),
                user_id=user_id
            )
            self.memory.add_message(session_id, user_message)
            
            # Process based on message type
            if message_type == MessageType.FLOWCHART:
                response = self._handle_flowchart_request(message, user_id, session_id)
            elif message_type == MessageType.MINDMAP:
                response = self._handle_mindmap_request(message, user_id, session_id)
            else:
                response = self._handle_text_request(message, user_id, session_id, document_ids)
            
            # Calculate processing time
            response.processing_time = (datetime.now() - start_time).total_seconds()
            
            # Add assistant response to memory
            assistant_message = ChatMessage(
                id=str(uuid.uuid4()),
                session_id=session_id,
                content=response.message,
                type=response.message_type,
                timestamp=datetime.now(),
                user_id=user_id,
                metadata={
                    'sources': response.sources or [],
                    'processing_time': response.processing_time,
                    'visualization_data': response.visualization_data
                }
            )
            self.memory.add_message(session_id, assistant_message)
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            error_response = ChatResponse(
                message=f"I apologize, but I encountered an error while processing your request: {str(e)}",
                message_type=MessageType.ERROR,
                error=str(e),
                processing_time=(datetime.now() - start_time).total_seconds()
            )
            return error_response
    
    def _handle_text_request(self, message: str, user_id: str, 
                           session_id: str, document_ids: Optional[List[str]]) -> ChatResponse:
        """Handle regular text-based questions."""
        
        # Get conversation context
        context = self.memory.get_context(session_id, include_last_n=3)
        
        # Enhance query with context if available
        if context:
            enhanced_query = f"""Previous conversation context:
{context}

Current question: {message}

Please answer the current question, taking into account the conversation context when relevant."""
        else:
            enhanced_query = message
        
        # Query RAG system
        rag_response = self.rag_service.query(
            question=enhanced_query,
            user_id=user_id,
            document_ids=document_ids,
            max_chunks=5
        )
        
        # Prepare sources information
        sources = []
        if rag_response.sources:
            for i, chunk in enumerate(rag_response.sources, 1):
                source_type = chunk.metadata.get('source_type', 'document')
                source_info = f"Source {i} ({source_type})"
                if 'page' in chunk.metadata:
                    source_info += f" - Page {chunk.metadata['page']}"
                sources.append(source_info)
        
        return ChatResponse(
            message=rag_response.answer,
            message_type=MessageType.TEXT,
            sources=sources
        )
    
    def _handle_flowchart_request(self, message: str, user_id: str, session_id: str) -> ChatResponse:
        """Handle flowchart generation requests."""
        
        try:
            answer, flowchart_dot = self.visualization_service.generate_flowchart(message, user_id)
            
            if not flowchart_dot:
                return ChatResponse(
                    message="I couldn't generate a flowchart for this topic. Please try with a different question or ensure you have relevant documents uploaded.",
                    message_type=MessageType.ERROR
                )
            
            visualization_data = {
                'type': 'flowchart',
                'dot_notation': flowchart_dot,
                'description': answer
            }
            
            return ChatResponse(
                message=f"Here's a flowchart representation:\n\n{answer}",
                message_type=MessageType.FLOWCHART,
                visualization_data=visualization_data
            )
            
        except Exception as e:
            logger.error(f"Error generating flowchart: {e}")
            return ChatResponse(
                message=f"I encountered an error while generating the flowchart: {str(e)}",
                message_type=MessageType.ERROR,
                error=str(e)
            )
    
    def _handle_mindmap_request(self, message: str, user_id: str, session_id: str) -> ChatResponse:
        """Handle mindmap generation requests."""
        
        try:
            answer, mindmap_connections = self.visualization_service.generate_mindmap(message, user_id)
            
            if not mindmap_connections:
                return ChatResponse(
                    message="I couldn't generate a mindmap for this topic. Please try with a different question or ensure you have relevant documents uploaded.",
                    message_type=MessageType.ERROR
                )
            
            visualization_data = {
                'type': 'mindmap',
                'connections': mindmap_connections,
                'description': answer
            }
            
            return ChatResponse(
                message=f"Here's a mindmap representation:\n\n{answer}",
                message_type=MessageType.MINDMAP,
                visualization_data=visualization_data
            )
            
        except Exception as e:
            logger.error(f"Error generating mindmap: {e}")
            return ChatResponse(
                message=f"I encountered an error while generating the mindmap: {str(e)}",
                message_type=MessageType.ERROR,
                error=str(e)
            )
    
    def get_conversation_history(self, session_id: str) -> List[ChatMessage]:
        """Get the conversation history for a session."""
        return self.memory.get_messages(session_id)
    
    def clear_conversation(self, session_id: str):
        """Clear the conversation history for a session."""
        self.memory.clear_session(session_id)
    
    def add_document_to_rag(self, document) -> bool:
        """Add a document to the RAG system."""
        return self.rag_service.add_document(document)
    
    def get_rag_stats(self) -> Dict:
        """Get RAG system statistics."""
        return self.rag_service.get_stats()
    
    def delete_document(self, document_id: str):
        """Delete a document from the RAG system."""
        self.rag_service.delete_document(document_id)

class ChatModeManager:
    """Manages different chat modes and their behaviors."""
    
    MODES = {
        'quick_query': {
            'name': 'Quick Query',
            'description': 'Get fast answers from your documents',
            'supports_context': False,
            'default_type': MessageType.TEXT
        },
        'conversational': {
            'name': 'Conversational Chat',
            'description': 'Have a conversation with memory of previous messages',
            'supports_context': True,
            'default_type': MessageType.TEXT
        },
        'flowchart': {
            'name': 'Flowchart Generation',
            'description': 'Generate flowcharts from your questions',
            'supports_context': False,
            'default_type': MessageType.FLOWCHART
        },
        'mindmap': {
            'name': 'Mindmap Generation',
            'description': 'Generate mindmaps from your questions',
            'supports_context': False,
            'default_type': MessageType.MINDMAP
        }
    }
    
    @classmethod
    def get_mode_info(cls, mode: str) -> Dict:
        """Get information about a specific mode."""
        return cls.MODES.get(mode, cls.MODES['quick_query'])
    
    @classmethod
    def get_all_modes(cls) -> Dict:
        """Get all available modes."""
        return cls.MODES
