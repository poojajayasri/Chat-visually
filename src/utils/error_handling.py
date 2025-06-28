# src/utils/error_handling.py
import functools
import traceback
from typing import Callable, Any, Optional
import streamlit as st
from .logger import get_logger

logger = get_logger(__name__)

def handle_errors(show_user_error: bool = True, 
                 default_return: Any = None,
                 error_message: Optional[str] = None):
    """Decorator for handling errors in functions."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Log the full error
                logger.error(f"Error in {func.__name__}: {str(e)}")
                logger.error(traceback.format_exc())
                
                # Show user-friendly error
                if show_user_error:
                    user_msg = error_message or f"An error occurred in {func.__name__}"
                    st.error(f"{user_msg}: {str(e)}")
                
                return default_return
        return wrapper
    return decorator

class DataMapError(Exception):
    """Base exception class for DataMap AI."""
    pass

class DocumentProcessingError(DataMapError):
    """Error during document processing."""
    pass

class EmbeddingError(DataMapError):
    """Error during embedding creation."""
    pass

class RAGError(DataMapError):
    """Error during RAG operations."""
    pass

class VisualizationError(DataMapError):
    """Error during visualization generation."""
    pass
