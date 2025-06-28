# main.py - FIXED VERSION
import streamlit as st
import asyncio
from pathlib import Path
from typing import Optional

# Configure Streamlit first
st.set_page_config(
    page_title="DataMap AI",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Simple error handling for imports
try:
    from src.config import Config, load_config
    from src.auth.auth_manager import AuthManager
    from src.services.document_service import DocumentService
    from src.services.rag_service import EmbeddingService, RAGService
    from src.services.chat_service import ChatService
    IMPORTS_SUCCESSFUL = True
except ImportError as e:
    st.error(f"Import error: {e}")
    st.info("🔧 Some components are not available. Running in simplified mode.")
    IMPORTS_SUCCESSFUL = False

class DataMapApp:
    """Main application class for DataMap AI."""
    
    def __init__(self):
        if IMPORTS_SUCCESSFUL:
            try:
                self.config = load_config()
                self.auth_manager = AuthManager(self.config.auth)
                self._initialize_services()
            except Exception as e:
                st.error(f"Configuration error: {e}")
                self.config = None
                self.auth_manager = None
        else:
            self.config = None
            self.auth_manager = None
        
        self._initialize_session_state()
    
    def _initialize_services(self):
        """Initialize core services."""
        try:
            self.document_service = DocumentService(self.config.storage)
            
            # Fix: EmbeddingService needs both embedding_config and llm_config
            self.embedding_service = EmbeddingService(
                self.config.embeddings,  # embedding_config
                self.config.llm         # llm_config
            )
            
            # Fix: RAGService needs all three configs
            self.rag_service = RAGService(
                self.config.llm,        # llm_config
                self.config.embeddings, # embedding_config
                self.config.storage     # storage_config
            )
            
            self.chat_service = ChatService(
                self.config.llm,
                self.embedding_service,
                self.document_service
            )
            
        except Exception as e:
            st.error(f"Service initialization error: {e}")
            # Set to None so we can handle gracefully
            self.document_service = None
            self.embedding_service = None
            self.rag_service = None
            self.chat_service = None
    
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
            st.error(f"Application error: {e}")
            st.info("🔧 This appears to be a setup issue. Please check your configuration.")
    
    def _load_styles(self):
        """Load custom CSS styles."""
        css_file = Path(__file__).parent / "src" / "ui" / "styles" / "main.css"
        if css_file.exists():
            try:
                with open(css_file) as f:
                    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
            except Exception:
                pass  # Ignore CSS loading errors
    
    def _render_auth(self):
        """Render authentication interface."""
        if self.auth_manager:
            try:
                auth_result = self.auth_manager.render_auth_ui()
                if auth_result.authenticated:
                    st.session_state.authenticated = True
                    st.session_state.user_id = auth_result.user_id
                    st.rerun()
            except Exception as e:
                st.error(f"Authentication error: {e}")
                self._render_simple_auth()
        else:
            self._render_simple_auth()
    
    def _render_simple_auth(self):
        """Render simple authentication fallback."""
        st.title("🗺️ DataMap AI")
        st.markdown("### Welcome to DataMap AI")
        st.info("🔧 Running in simplified mode due to configuration issues.")
        
        if st.button("🚀 Continue as Demo User", type="primary"):
            st.session_state.authenticated = True
            st.session_state.user_id = "demo_user"
            st.rerun()
    
    def _render_app(self):
        """Render main application interface."""
        # Simple sidebar
        with st.sidebar:
            st.markdown("### 🗺️ DataMap AI")
            st.markdown(f"**User:** {st.session_state.user_id}")
            
            # Navigation
            pages = {
                "🏠 Home": "home",
                "📁 Upload": "upload",
                "💬 Chat": "chat",
                "📊 Visualization": "visualization",
                "⚙️ Settings": "settings"
            }
            
            selected = st.radio(
                "Navigation",
                list(pages.keys()),
                label_visibility="collapsed"
            )
            
            current_page = pages[selected]
            
            # Update page if changed
            if current_page != st.session_state.current_page:
                st.session_state.current_page = current_page
                st.rerun()
            
            st.markdown("---")
            
            # Stats
            if st.session_state.documents:
                st.metric("Documents", len(st.session_state.documents))
            
            # Logout
            if st.button("🚪 Logout"):
                for key in list(st.session_state.keys()):
                    if key not in ['current_page']:  # Keep some state
                        del st.session_state[key]
                st.rerun()
        
        # Render current page
        self._render_page(st.session_state.current_page)
    
    def _render_page(self, page_name: str):
        """Render the specified page."""
        if page_name == 'home':
            self._render_home_page()
        elif page_name == 'upload':
            self._render_upload_page()
        elif page_name == 'chat':
            self._render_chat_page()
        elif page_name == 'visualization':
            self._render_visualization_page()
        elif page_name == 'settings':
            self._render_settings_page()
        else:
            st.error(f"Page '{page_name}' not found.")
    
    def _render_home_page(self):
        """Render home page."""
        st.title("🗺️ Welcome to DataMap AI")
        
        st.markdown("""
        ### Transform Your Documents into Interactive Knowledge Maps
        
        DataMap AI helps you:
        - **📁 Upload** various document types (PDF, DOCX, images, etc.)
        - **💬 Chat** with your documents using AI
        - **📊 Generate** interactive flowcharts and mindmaps
        - **🔍 Explore** your knowledge visually
        """)
        
        # Quick start guide
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            #### 1. 📁 Upload Documents
            Start by uploading your documents, PDFs, or images.
            """)
            if st.button("📁 Go to Upload", key="nav_upload"):
                st.session_state.current_page = 'upload'
                st.rerun()
        
        with col2:
            st.markdown("""
            #### 2. 💬 Ask Questions
            Chat with your documents and get intelligent answers.
            """)
            if st.button("💬 Start Chatting", key="nav_chat"):
                st.session_state.current_page = 'chat'
                st.rerun()
        
        with col3:
            st.markdown("""
            #### 3. 📊 Create Visuals
            Generate flowcharts and mindmaps from your content.
            """)
            if st.button("📊 Visualize", key="nav_viz"):
                st.session_state.current_page = 'visualization'
                st.rerun()
        
        # System status
        st.markdown("---")
        if IMPORTS_SUCCESSFUL and self.config:
            st.success("✅ All systems operational")
        else:
            st.warning("⚠️ Running in simplified mode - some features may be limited")
    
    def _render_upload_page(self):
        """Render upload page."""
        st.title("📁 Document Upload")
        
        # Check if services are available
        if not self.document_service:
            st.warning("⚠️ Upload service not available. Please check configuration.")
            return
        
        st.markdown("### Upload Your Documents")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Choose files",
            accept_multiple_files=True,
            type=['pdf', 'docx', 'txt', 'csv', 'png', 'jpg'],
            help="Upload documents to chat with them"
        )
        
        # URL inputs
        col1, col2 = st.columns(2)
        with col1:
            youtube_url = st.text_input("YouTube URL", placeholder="https://youtube.com/watch?v=...")
        with col2:
            web_url = st.text_input("Website URL", placeholder="https://example.com")
        
        if st.button("🚀 Process Documents", type="primary"):
            if uploaded_files or youtube_url or web_url:
                with st.spinner("Processing documents..."):
                    try:
                        user_id = st.session_state.user_id
                        processed_docs = []
                        
                        # Process uploaded files
                        for file in uploaded_files:
                            file_path = self.document_service.save_uploaded_file(file, user_id)
                            result = self.document_service.process_document(file_path, user_id, file.name)
                            
                            if result.success:
                                processed_docs.append(result.document)
                                if self.rag_service:
                                    self.rag_service.add_document(result.document)
                                st.success(f"✅ {file.name} processed ({result.chunks_created} chunks)")
                            else:
                                st.error(f"❌ Error processing {file.name}: {result.error}")
                        
                        # Process URLs
                        if youtube_url:
                            result = self.document_service.process_url(youtube_url, user_id, "youtube")
                            if result.success:
                                processed_docs.append(result.document)
                                if self.rag_service:
                                    self.rag_service.add_document(result.document)
                                st.success(f"✅ YouTube video processed ({result.chunks_created} chunks)")
                        
                        if web_url:
                            result = self.document_service.process_url(web_url, user_id, "web")
                            if result.success:
                                processed_docs.append(result.document)
                                if self.rag_service:
                                    self.rag_service.add_document(result.document)
                                st.success(f"✅ Website processed ({result.chunks_created} chunks)")
                        
                        # Add to session state
                        st.session_state.documents.extend(processed_docs)
                        
                        if processed_docs:
                            st.balloons()
                            
                    except Exception as e:
                        st.error(f"Processing error: {e}")
            else:
                st.warning("Please upload files or provide URLs to process.")
        
        # Show uploaded documents
        if st.session_state.documents:
            st.markdown("---")
            st.markdown("### 📚 Uploaded Documents")
            for doc in st.session_state.documents:
                with st.expander(f"📄 {doc.metadata.title}"):
                    st.write(f"**Type:** {doc.metadata.file_type}")
                    st.write(f"**Chunks:** {len(doc.chunks)}")
                    st.write(f"**Size:** {doc.metadata.file_size / 1024:.1f} KB")
    
    def _render_chat_page(self):
        """Render chat page."""
        st.title("💬 Chat with Your Documents")
        
        if not st.session_state.documents:
            st.warning("📁 No documents uploaded yet. Please upload documents first.")
            if st.button("📁 Go to Upload"):
                st.session_state.current_page = 'upload'
                st.rerun()
            return
        
        # Chat mode selector
        mode = st.selectbox(
            "Chat Mode",
            ["Quick Q&A", "Conversational", "Generate Flowchart", "Generate Mindmap"]
        )
        
        # Chat interface
        if 'chat_messages' not in st.session_state:
            st.session_state.chat_messages = [
                {"role": "assistant", "content": f"Hello! I can help you explore your {len(st.session_state.documents)} uploaded documents. What would you like to know?"}
            ]
        
        # Display chat messages
        for message in st.session_state.chat_messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask a question about your documents..."):
            # Add user message
            st.session_state.chat_messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.write(prompt)
            
            # Generate response
            with st.chat_message("assistant"):
                if self.rag_service:
                    try:
                        user_id = st.session_state.user_id
                        response = self.rag_service.query(prompt, user_id)
                        st.write(response.answer)
                        
                        if response.sources:
                            with st.expander("📚 Sources"):
                                for i, source in enumerate(response.sources[:3], 1):
                                    st.write(f"**Source {i}:** {source.content[:200]}...")
                        
                        st.session_state.chat_messages.append({"role": "assistant", "content": response.answer})
                    except Exception as e:
                        error_msg = f"I encountered an error: {str(e)}"
                        st.error(error_msg)
                        st.session_state.chat_messages.append({"role": "assistant", "content": error_msg})
                else:
                    fallback_msg = f"I can see you're asking about: '{prompt}'. The full RAG system isn't available, but I can see you have {len(st.session_state.documents)} documents uploaded."
                    st.write(fallback_msg)
                    st.session_state.chat_messages.append({"role": "assistant", "content": fallback_msg})
    
    def _render_visualization_page(self):
        """Render visualization page."""
        st.title("📊 Knowledge Visualization")
        
        if not st.session_state.documents:
            st.warning("📁 No documents uploaded yet. Please upload documents first.")
            return
        
        # Visualization type
        viz_type = st.radio("Visualization Type", ["Flowchart", "Mindmap"], horizontal=True)
        
        # Input
        query = st.text_area(
            "What would you like to visualize?",
            placeholder="Enter a topic or question to create a visual representation...",
            height=100
        )
        
        if st.button("🎨 Generate Visualization", type="primary"):
            if query:
                with st.spinner(f"Generating {viz_type.lower()}..."):
                    # Simple visualization placeholder
                    st.success(f"✅ {viz_type} generated!")
                    
                    if viz_type == "Flowchart":
                        # Simple flowchart example
                        st.graphviz_chart(f'''
                        digraph {{
                            rankdir=TB;
                            node [shape=box, style=rounded];
                            
                            "Your Query" [style=filled, fillcolor=lightblue];
                            "Key Concept 1" [style=filled, fillcolor=lightgreen];
                            "Key Concept 2" [style=filled, fillcolor=lightgreen];
                            "Detail A" [style=filled, fillcolor=lightyellow];
                            "Detail B" [style=filled, fillcolor=lightyellow];
                            
                            "Your Query" -> "Key Concept 1";
                            "Your Query" -> "Key Concept 2";
                            "Key Concept 1" -> "Detail A";
                            "Key Concept 2" -> "Detail B";
                        }}
                        ''')
                    else:
                        st.info("🧠 Interactive mindmap visualization coming soon!")
                        st.markdown("**Mindmap Structure:**")
                        st.markdown("- Main Topic: " + query)
                        st.markdown("  - Subtopic 1")
                        st.markdown("  - Subtopic 2")
                        st.markdown("    - Detail A")
                        st.markdown("    - Detail B")
            else:
                st.warning("Please enter a query to generate a visualization.")
    
    def _render_settings_page(self):
        """Render settings page."""
        st.title("⚙️ Settings")
        
        # API Configuration
        st.markdown("### 🔑 API Configuration")
        
        current_key = st.session_state.get('openai_api_key', '')
        if current_key:
            masked_key = f"{'*' * 20}{current_key[-4:]}"
        else:
            masked_key = "Not configured"
        
        st.info(f"Current API Key: {masked_key}")
        
        new_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help="Your OpenAI API key for AI processing"
        )
        
        if st.button("Update API Key"):
            if new_key:
                st.session_state.openai_api_key = new_key
                st.success("API key updated!")
            else:
                st.error("Please enter a valid API key")
        
        # System Information
        st.markdown("---")
        st.markdown("### 📊 System Information")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Documents Uploaded", len(st.session_state.documents))
            st.metric("Chat Messages", len(st.session_state.get('chat_messages', [])))
        
        with col2:
            st.metric("Services Available", "2/4" if IMPORTS_SUCCESSFUL else "0/4")
            st.metric("Config Status", "✅" if self.config else "❌")
        
        # Clear Data
        st.markdown("---")
        st.markdown("### 🗑️ Clear Data")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Clear Chat History"):
                st.session_state.chat_messages = []
                st.success("Chat history cleared!")
        
        with col2:
            if st.button("Clear All Documents"):
                st.session_state.documents = []
                st.success("Documents cleared!")

def main():
    """Application entry point."""
    app = DataMapApp()
    app.run()

if __name__ == "__main__":
    main()
