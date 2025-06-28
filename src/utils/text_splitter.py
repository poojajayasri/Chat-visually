# src/utils/text_splitter.py
import re
from typing import List, Optional
from dataclasses import dataclass

@dataclass
class TextSplitterConfig:
    """Configuration for text splitting."""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    separators: List[str] = None
    keep_separator: bool = True
    
    def __post_init__(self):
        if self.separators is None:
            self.separators = ["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""]

class SmartTextSplitter:
    """Intelligent text splitter that preserves context."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, 
                 separators: Optional[List[str]] = None):
        self.config = TextSplitterConfig(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators
        )
    
    def split_text(self, text: str) -> List[str]:
        """Split text into chunks while preserving context."""
        if not text.strip():
            return []
        
        # Clean the text
        text = self._clean_text(text)
        
        if len(text) <= self.config.chunk_size:
            return [text]
        
        chunks = []
        current_chunk = ""
        
        # Split by separators in order of preference
        parts = self._split_by_separators(text)
        
        for part in parts:
            # If adding this part would exceed chunk size
            if len(current_chunk) + len(part) > self.config.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    
                    # Start new chunk with overlap
                    overlap_text = self._get_overlap(current_chunk)
                    current_chunk = overlap_text + part
                else:
                    # Part is larger than chunk size, split it
                    if len(part) > self.config.chunk_size:
                        sub_chunks = self._force_split(part)
                        chunks.extend(sub_chunks[:-1])
                        current_chunk = sub_chunks[-1]
                    else:
                        current_chunk = part
            else:
                current_chunk += part
        
        # Add the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove excessive newlines
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
        return text.strip()
    
    def _split_by_separators(self, text: str) -> List[str]:
        """Split text by separators in order of preference."""
        parts = [text]
        
        for separator in self.config.separators:
            new_parts = []
            for part in parts:
                if separator in part:
                    split_parts = part.split(separator)
                    for i, split_part in enumerate(split_parts):
                        if i < len(split_parts) - 1:
                            new_parts.append(split_part + separator)
                        else:
                            new_parts.append(split_part)
                else:
                    new_parts.append(part)
            parts = new_parts
        
        return [p for p in parts if p.strip()]
    
    def _get_overlap(self, text: str) -> str:
        """Get overlap text from the end of current chunk."""
        if len(text) <= self.config.chunk_overlap:
            return text
        
        # Try to find a good breaking point for overlap
        overlap_text = text[-self.config.chunk_overlap:]
        
        # Find the last sentence boundary
        for separator in [". ", "! ", "? ", "\n"]:
            last_sep = overlap_text.rfind(separator)
            if last_sep > self.config.chunk_overlap // 2:
                return overlap_text[last_sep + len(separator):]
        
        # If no good boundary found, use the last chunk_overlap characters
        return overlap_text
    
    def _force_split(self, text: str) -> List[str]:
        """Force split text that's larger than chunk size."""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.config.chunk_size
            if end >= len(text):
                chunks.append(text[start:])
                break
            
            # Try to find a good breaking point
            chunk = text[start:end]
            for separator in [". ", "! ", "? ", "\n", " "]:
                last_sep = chunk.rfind(separator)
                if last_sep > self.config.chunk_size * 0.7:
                    chunk = text[start:start + last_sep + len(separator)]
                    break
            
            chunks.append(chunk)
            start += len(chunk) - self.config.chunk_overlap
        
        return chunks
