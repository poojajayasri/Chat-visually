from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Any, Optional
from enum import Enum

@dataclass
class DocumentMetadata:
    """Metadata for a document."""
    id: str
    title: str
    filename: str
    file_type: str
    mime_type: Optional[str]
    file_size: int
    user_id: str
    created_at: datetime
    file_path: Optional[str] = None
    source_url: Optional[str] = None
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []

@dataclass
class DocumentChunk:
    """A chunk of text from a document."""
    id: str
    document_id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None

@dataclass
class Document:
    """Complete document with metadata and chunks."""
    metadata: DocumentMetadata
    chunks: List[DocumentChunk]
