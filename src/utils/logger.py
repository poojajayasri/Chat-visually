# src/utils/logger.py - FIXED VERSION
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional  # ← MISSING IMPORT ADDED

def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """Setup application logging."""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[]
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(file_handler)

def get_logger(name: str) -> logging.Logger:
    """Get a logger instance."""
    return logging.getLogger(name)

# src/utils/text_splitter.py - FIXED VERSION
import re
from typing import List, Optional  # ← ADDED IMPORT
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

# src/utils/validators.py - FIXED VERSION  
import re
from typing import Optional, Tuple  # ← ADDED IMPORTS
from urllib.parse import urlparse

def validate_email(email: str) -> bool:
    """Validate email address format."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def validate_url(url: str) -> Tuple[bool, Optional[str]]:
    """Validate URL format and return validation result."""
    try:
        result = urlparse(url)
        if not all([result.scheme, result.netloc]):
            return False, "Invalid URL format"
        
        if result.scheme not in ['http', 'https']:
            return False, "URL must use HTTP or HTTPS"
        
        return True, None
    except Exception as e:
        return False, f"Invalid URL: {str(e)}"

def validate_youtube_url(url: str) -> Tuple[bool, Optional[str]]:
    """Validate YouTube URL and extract video ID."""
    youtube_patterns = [
        r'(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]{11})',
        r'youtube\.com/embed/([a-zA-Z0-9_-]{11})',
        r'youtube\.com/v/([a-zA-Z0-9_-]{11})'
    ]
    
    for pattern in youtube_patterns:
        match = re.search(pattern, url)
        if match:
            return True, match.group(1)
    
    return False, None

def validate_api_key(api_key: str) -> bool:
    """Validate OpenAI API key format."""
    # OpenAI API keys start with 'sk-' and are typically 51 characters long
    pattern = r'^sk-[a-zA-Z0-9]{48}$'
    return bool(re.match(pattern, api_key))

def sanitize_filename(filename: str) -> str:
    """Sanitize filename by removing invalid characters."""
    # Remove invalid characters for filenames
    invalid_chars = r'[<>:"/\\|?*]'
    sanitized = re.sub(invalid_chars, '_', filename)
    
    # Remove leading/trailing dots and spaces
    sanitized = sanitized.strip('. ')
    
    # Ensure filename is not empty
    if not sanitized:
        sanitized = "untitled"
    
    return sanitized
