# main.py
import streamlit as st
import asyncio
from pathlib import Path
from typing import Optional

from src.config import Config, load_config
from src.auth.auth_manager import AuthManager
from src.ui.pages import HomePage, ChatPage, UploadPage, VisualizationPage
from src.ui.components import Sidebar
from src.services.document_service import DocumentService
from src.services.embedding_service import EmbeddingService
from src.services.chat_service import ChatService
from src.utils.logger import get_logger

# Configure Streamlit
st.set_page_config(
    page_title="DataMap AI",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

logger = get_logger(__name__)

class DataMapApp:
    """Main application class for DataMap AI."""
    
    def __init__(self):
        self.config = load_config()
        self.auth_manager = AuthManager(self.config.auth)
        self._initialize_services()
        self._initialize_session_state()
    
    def _initialize_services(self):
        """Initialize core services."""
        self.document_service = DocumentService(self.config.storage)
        self.embedding_service = EmbeddingService(self.config.embeddings)
        self.chat_service = ChatService(
            self.config.llm,
            self.embedding_service,
            self.document_service
        )
    
    def _initialize_session_state(self):
        """Initialize session state variables."""
        if 'authenticated' not in st.session_state:
            st.session_state.authenticated = False
        if 'user_id' not in st.session_state:
            st.session_state.user_id = None
        if 'current_page' not in st.session_state:
            st.session_state.current_page = 'home'
        if 'documents' not in st.session_state:
            st.session_state.documents = []
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
    
    def run(self):
        """Main application entry point."""
        try:
            # Load custom CSS
            self._load_styles()
            
            # Handle authentication
            if not st.session_state.authenticated:
                self._render_auth()
                return
            
            # Render main application
            self._render_app()
            
        except Exception as e:
            logger.error(f"Application error: {e}")
            st.error("An unexpected error occurred. Please refresh the page.")
    
    def _load_styles(self):
        """Load custom CSS styles."""
        css_file = Path(__file__).parent / "src" / "ui" / "styles" / "main.css"
        if css_file.exists():
            with open(css_file) as f:
                st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    
    def _render_auth(self):
        """Render authentication interface."""
        auth_result = self.auth_manager.render_auth_ui()
        if auth_result.authenticated:
            st.session_state.authenticated = True
            st.session_state.user_id = auth_result.user_id
            st.rerun()
    
    def _render_app(self):
        """Render main application interface."""
        # Sidebar navigation
        sidebar = Sidebar()
        current_page = sidebar.render()
        
        # Update current page if changed
        if current_page != st.session_state.current_page:
            st.session_state.current_page = current_page
            st.rerun()
        
        # Render appropriate page
        self._render_page(current_page)
    
    def _render_page(self, page_name: str):
        """Render the specified page."""
        pages = {
            'home': HomePage(),
            'upload': UploadPage(self.document_service, self.embedding_service),
            'chat': ChatPage(self.chat_service),
            'visualization': VisualizationPage(self.chat_service)
        }
        
        page = pages.get(page_name)
        if page:
            page.render()
        else:
            st.error(f"Page '{page_name}' not found.")

def main():
    """Application entry point."""
    app = DataMapApp()
    app.run()

if __name__ == "__main__":
    main()
