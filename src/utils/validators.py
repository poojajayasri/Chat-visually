# src/utils/validators.py
import re
from typing import Optional, Tuple
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
