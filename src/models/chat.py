# src/models/chat.py
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Any, Optional
from enum import Enum

class MessageType(Enum):
    """Types of chat messages."""
    USER = "user"
    ASSISTANT = "assistant"
    TEXT = "text"
    FLOWCHART = "flowchart"
    MINDMAP = "mindmap"
    ERROR = "error"

@dataclass
class ChatMessage:
    """A single chat message."""
    id: str
    session_id: str
    content: str
    type: MessageType
    timestamp: datetime
    user_id: str
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class ChatSession:
    """A chat session containing multiple messages."""
    id: str
    user_id: str
    created_at: datetime
    last_activity: datetime
    messages: List[ChatMessage]
    metadata: Optional[Dict[str, Any]] = None
