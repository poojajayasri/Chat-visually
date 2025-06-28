# src/ui/pages.py
import streamlit as st
from typing import Optional, List
from datetime import datetime
import uuid

from .components import (
    DocumentUploader, ChatInterface, DocumentManager, 
    VisualizationPanel, StatsPanel, AlertSystem
)
from ..models.document import Document
from ..services.chat_service import ChatService, ChatModeManager
from ..services.document_service import DocumentService
from ..services.rag_service import EmbeddingService

class BasePage:
    """Base class for all pages."""
    
    def __init__(self):
        self.alert = AlertSystem()
    
    def render(self):
        """Render the page. Must be implemented by subclasses."""
        raise NotImplementedError

class HomePage(BasePage):
    """Home page with welcome content and quick start guide."""
    
    def render(self):
        # Hero section
        st.markdown("""
            <div style="text-align: center; padding: 40px 0;">
                <h1 style="font-size: 3.5em; color: #2E4057; margin-bottom: 20px;">
                    🗺️ DataMap AI
                </h1>
                <h2 style="color: #666; font-weight: 300; margin-bottom: 30px;">
                    Transform Your Documents into Interactive Knowledge Maps
                </h2>
                <p style="font-size: 1.2em; color: #888; max-width: 800px; margin: 0 auto;">
                    Upload documents, ask questions, and generate beautiful flowcharts and mindmaps 
                    from your content using advanced AI technology.
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        # Quick start section
        st.markdown("---")
        st.markdown("## 🚀 Quick Start Guide")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
                ### 1. 📁 Upload Documents
                - PDF, DOCX, CSV files
                - Images with text (OCR)
                - YouTube videos
                - Web pages
            """)
            
            if st.button("📁 Go to Upload", key="nav_upload", use_container_width=True):
                st.session_state.current_page = 'upload'
                st.rerun()
        
        with col2:
            st.markdown("""
                ### 2. 💬 Ask Questions
                - Quick Q&A mode
                - Conversational chat
                - Context-aware responses
                - Source citations
            """)
            
            if st.button("💬 Start Chatting", key="nav_chat", use_container_width=True):
                st.session_state.current_page = 'chat'
                st.rerun()
        
        with col3:
            st.markdown("""
                ### 3. 📊 Generate Visuals
                - Interactive flowcharts
                - Dynamic mindmaps
                - Concept relationships
                - Knowledge exploration
            """)
            
            if st.button("📊 Create Visuals", key="nav_viz", use_container_width=True):
                st.session_state.current_page = 'visualization'
                st.rerun()
        
        # Features showcase
        st.markdown("---")
        st.markdown("## ✨ Key Features")
        
        features = [
            {
                "icon": "🤖",
                "title": "Advanced RAG Technology",
                "description": "Retrieval-Augmented Generation for accurate, context-aware responses"
            },
            {
                "icon": "🌐",
                "title": "Multi-Modal Input",
                "description": "Process text, images, videos, and web content seamlessly"
            },
            {
                "icon": "🎨",
                "title": "Visual Knowledge Maps",
                "description": "Transform complex information into interactive diagrams"
            },
            {
                "icon": "💾",
                "title": "Smart Document Processing",
                "description": "Intelligent chunking and embedding for optimal retrieval"
            },
            {
                "icon": "🔒",
                "title": "Privacy Focused",
                "description": "Your documents are processed locally and securely"
            },
            {
                "icon": "⚡",
                "title": "Real-time Processing",
                "description": "Fast response times with efficient vector search"
            }
        ]
        
        for i in range(0, len(features), 3):
            cols = st.columns(3)
            for j, feature in enumerate(features[i:i+3]):
                with cols[j]:
                    st.markdown(f"""
                        <div style="text-align: center; padding: 20px; border: 1px solid #e0e0e0; border-radius: 10px; margin: 10px 0;">
                            <div style="font-size: 2em; margin-bottom: 10px;">{feature['icon']}</div>
                            <h4 style="color: #2E4057; margin-bottom: 10px;">{feature['title']}</h4>
                            <p style="color: #666; font-size: 0.9em;">{feature['description']}</p>
                        </div>
                    """, unsafe_allow_html=True)
        
        # Getting started tips
        if not st.session_state.get('documents'):
            st.markdown("---")
            st.info("""
                **👋 New to DataMap AI?** Start by uploading some documents to see the magic happen! 
                Try uploading a PDF research paper, a CSV dataset, or paste a YouTube educational video URL.
            """)

class UploadPage(BasePage):
    """Document upload and management page."""
    
    def __init__(self, document_service: DocumentService, embedding_service: EmbeddingService):
        super().__init__()
        self.document_service = document_service
        self.embedding_service = embedding_service
        self.uploader = DocumentUploader(document_service, embedding_service)
        self.manager = DocumentManager()
    
    def render(self):
        st.title("📁 Document Upload & Management")
        
        # Upload section
        uploaded_docs = self.uploader.render()
        
        # Add to session state
        if uploaded_docs:
            if 'documents' not in st.session_state:
                st.session_state.documents = []
            st.session_state.documents.extend(uploaded_docs)
        
        st.markdown("---")
        
        # Document management section
        documents = st.session_state.get('documents', [])
        self.manager.render(documents)
        
        # Tips and guidelines
        st.markdown("---")
        with st.expander("💡 Upload Tips & Guidelines"):
            st.markdown("""
            ### 📋 Supported Formats
            - **PDF**: Research papers, reports, manuals
            - **DOCX**: Word documents, articles
            - **CSV**: Data files, spreadsheets
            - **Images**: PNG, JPG with text (OCR enabled)
            - **YouTube**: Educational videos, lectures
            - **Websites**: Articles, documentation
            
            ### 🎯 Best Practices
            - **File Size**: Keep files under 50MB for optimal performance
            - **Quality**: Use clear, well-formatted documents for best results
            - **Language**: English content works best with current models
            - **Structure**: Well-organized content yields better chunks
            
            ### ⚡ Processing Tips
            - Larger documents take more time to process
            - Images require OCR processing (may be slower)
            - YouTube videos are processed via transcript extraction
            - Web pages are crawled and cleaned automatically
            """)

class ChatPage(BasePage):
    """Interactive chat page with multiple modes."""
    
    def __init__(self, chat_service: ChatService):
        super().__init__()
        self.chat_service = chat_service
        self.interface = ChatInterface(chat_service)
    
    def render(self):
        st.title("💬 Interactive Chat")
        
        # Check if documents are available
        if not st.session_state.get('documents'):
            self.alert.warning(
                "No documents uploaded yet. Upload documents first to enable document-based chat."
            )
            if st.button("📁 Go to Upload Page"):
                st.session_state.current_page = 'upload'
                st.rerun()
            return
        
        # Session management
        if 'chat_session_id' not in st.session_state:
            user_id = st.session_state.get('user_id', 'anonymous')
            st.session_state.chat_session_id = self.chat_service.create_session(user_id)
        
        session_id = st.session_state.chat_session_id
        
        # Chat mode information
        with st.expander("ℹ️ Chat Modes Explained"):
            st.markdown("""
            ### 🎯 Available Chat Modes
            
            **Quick Query**: Fast Q&A without conversation memory
            - Best for: Simple questions, fact-finding
            - Features: Direct answers, source citations
            
            **Conversational Chat**: Context-aware dialogue
            - Best for: Complex discussions, follow-up questions
            - Features: Memory of previous messages, natural conversation flow
            
            **Flowchart Generation**: Visual process mapping
            - Best for: Understanding workflows, decision trees
            - Features: Automatic diagram generation, interactive visuals
            
            **Mindmap Generation**: Concept relationship mapping
            - Best for: Learning, knowledge exploration
            - Features: Hierarchical concept visualization, expandable nodes
            """)
        
        # Render chat interface
        self.interface.render(session_id)
        
        # Session controls
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            if st.button("🔄 New Session"):
                user_id = st.session_state.get('user_id', 'anonymous')
                st.session_state.chat_session_id = self.chat_service.create_session(user_id)
                st.rerun()
        
        with col2:
            # Download conversation
            messages = self.chat_service.get_conversation_history(session_id)
            if messages:
                conversation_text = self._export_conversation(messages)
                st.download_button(
                    "💾 Export Chat",
                    data=conversation_text,
                    file_name=f"datamap_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
        
        with col3:
            # Show session stats
            if messages:
                st.info(f"📊 Session: {len(messages)} messages | Started: {messages[0].timestamp.strftime('%H:%M')}")
    
    def _export_conversation(self, messages: List) -> str:
        """Export conversation to text format."""
        lines = [
            "DataMap AI Conversation Export",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 50,
            ""
        ]
        
        for msg in messages:
            role = "User" if msg.type.value == "user" else "Assistant"
            timestamp = msg.timestamp.strftime('%H:%M:%S')
            lines.append(f"[{timestamp}] {role}:")
            lines.append(msg.content)
            lines.append("")
        
        return "\n".join(lines)

class VisualizationPage(BasePage):
    """Dedicated visualization creation and management page."""
    
    def __init__(self, chat_service: ChatService):
        super().__init__()
        self.chat_service = chat_service
        self.viz_panel = VisualizationPanel(chat_service)
    
    def render(self):
        st.title("📊 Knowledge Visualization")
        
        # Check if documents are available
        if not st.session_state.get('documents'):
            self.alert.warning(
                "No documents uploaded yet. Upload documents first to create visualizations."
            )
            if st.button("📁 Go to Upload Page"):
                st.session_state.current_page = 'upload'
                st.rerun()
            return
        
        # Visualization creation panel
        self.viz_panel.render()
        
        # Examples and inspiration
        st.markdown("---")
        with st.expander("💡 Visualization Ideas & Examples"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                ### 🔄 Flowchart Ideas
                - "Show me the process described in the document"
                - "Create a decision tree for the methodology"
                - "Map out the workflow from the manual"
                - "Visualize the algorithm steps"
                - "Show the cause and effect relationships"
                """)
            
            with col2:
                st.markdown("""
                ### 🧠 Mindmap Ideas
                - "Break down the main concepts in this topic"
                - "Show relationships between key terms"
                - "Create a knowledge map of the subject"
                - "Organize the information hierarchically"
                - "Map out the learning objectives"
                """)
        
        # Tips for better visualizations
        with st.expander("🎯 Tips for Better Visualizations"):
            st.markdown("""
            ### 📈 Getting the Best Results
            
            **Be Specific**: Instead of "explain machine learning", try "show the machine learning workflow from data collection to model deployment"
            
            **Use Action Words**: "Map out", "Break down", "Show the process", "Visualize the relationship"
            
            **Reference Your Documents**: "Based on the uploaded research paper, create a flowchart of the methodology"
            
            **Iterative Refinement**: Try different phrasings if the first result isn't perfect
            
            ### 🎨 Visualization Best Practices
            - **Flowcharts**: Best for processes, decisions, and sequential steps
            - **Mindmaps**: Best for concepts, relationships, and knowledge structures
            - **Complexity**: Aim for 5-12 nodes for optimal readability
            - **Clarity**: Simple, clear labels work better than complex phrases
            """)

class SettingsPage(BasePage):
    """Application settings and configuration page."""
    
    def render(self):
        st.title("⚙️ Settings & Configuration")
        
        # API Configuration
        st.markdown("### 🔑 API Configuration")
        with st.expander("OpenAI API Settings"):
            current_key = st.session_state.get('openai_api_key', '')
            masked_key = f"{'*' * 20}{current_key[-4:]}" if current_key else "Not configured"
            
            st.info(f"Current API Key: {masked_key}")
            
            new_key = st.text_input(
                "OpenAI API Key",
                type="password",
                help="Your OpenAI API key for AI processing"
            )
            
            if st.button("Update API Key"):
                if new_key:
                    st.session_state.openai_api_key = new_key
                    st.success("API key updated successfully!")
                else:
                    st.error("Please enter a valid API key")
        
        # Chat Settings
        st.markdown("### 💬 Chat Settings")
        with st.expander("Conversation Settings"):
            max_history = st.slider(
                "Maximum conversation history",
                min_value=5,
                max_value=50,
                value=st.session_state.get('max_chat_history', 10),
                help="Number of previous messages to remember in conversations"
            )
            st.session_state.max_chat_history = max_history
            
            auto_clear = st.checkbox(
                "Auto-clear conversations after 1 hour",
                value=st.session_state.get('auto_clear_chat', False)
            )
            st.session_state.auto_clear_chat = auto_clear
        
        # Processing Settings
        st.markdown("### ⚡ Processing Settings")
        with st.expander("Document Processing"):
            chunk_size = st.slider(
                "Text chunk size",
                min_value=500,
                max_value=2000,
                value=st.session_state.get('chunk_size', 1000),
                help="Size of text chunks for processing (affects accuracy vs speed)"
            )
            st.session_state.chunk_size = chunk_size
            
            chunk_overlap = st.slider(
                "Chunk overlap",
                min_value=50,
                max_value=500,
                value=st.session_state.get('chunk_overlap', 200),
                help="Overlap between chunks (helps maintain context)"
            )
            st.session_state.chunk_overlap = chunk_overlap
        
        # Statistics and Usage
        st.markdown("### 📊 Statistics & Usage")
        stats_panel = StatsPanel()
        stats_panel.render(self.chat_service)
        
        # Data Management
        st.markdown("### 🗂️ Data Management")
        with st.expander("Clear Application Data"):
            st.warning("⚠️ These actions cannot be undone!")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("🗑️ Clear Chat History", type="secondary"):
                    if 'chat_session_id' in st.session_state:
                        self.chat_service.clear_conversation(st.session_state.chat_session_id)
                    st.success("Chat history cleared!")
            
            with col2:
                if st.button("📁 Clear Documents", type="secondary"):
                    st.session_state.documents = []
                    st.success("Documents cleared!")
            
            with col3:
                if st.button("🔄 Reset All Data", type="secondary"):
                    # Clear all session state except authentication
                    keys_to_keep = ['authenticated', 'user_id']
                    keys_to_delete = [k for k in st.session_state.keys() if k not in keys_to_keep]
                    
                    for key in keys_to_delete:
                        del st.session_state[key]
                    
                    st.success("All data reset!")
                    st.rerun()
        
        # Export Settings
        st.markdown("### 💾 Export & Backup")
        with st.expander("Export Application Data"):
            if st.session_state.get('documents'):
                # Export document metadata
                doc_metadata = []
                for doc in st.session_state.documents:
                    doc_metadata.append({
                        'title': doc.metadata.title,
                        'file_type': doc.metadata.file_type,
                        'created_at': doc.metadata.created_at.isoformat(),
                        'chunks_count': len(doc.chunks)
                    })
                
                metadata_json = json.dumps(doc_metadata, indent=2)
                st.download_button(
                    "📋 Export Document Metadata",
                    data=metadata_json,
                    file_name=f"datamap_documents_{datetime.now().strftime('%Y%m%d')}.json",
                    mime="application/json"
                )
            
            # Export chat sessions
            if hasattr(self, 'chat_service') and self.chat_service.memory.conversations:
                chat_export = {}
                for session_id, messages in self.chat_service.memory.conversations.items():
                    chat_export[session_id] = [
                        {
                            'content': msg.content,
                            'type': msg.type.value,
                            'timestamp': msg.timestamp.isoformat()
                        }
                        for msg in messages
                    ]
                
                chat_json = json.dumps(chat_export, indent=2)
                st.download_button(
                    "💬 Export Chat History",
                    data=chat_json,
                    file_name=f"datamap_chats_{datetime.now().strftime('%Y%m%d')}.json",
                    mime="application/json"
                )
        
        # About and Help
        st.markdown("### ℹ️ About DataMap AI")
        with st.expander("Application Information"):
            st.markdown("""
            **DataMap AI** - Version 2.0
            
            A modern RAG (Retrieval-Augmented Generation) application that transforms 
            documents into interactive knowledge visualizations.
            
            **Key Technologies:**
            - OpenAI GPT models for natural language processing
            - ChromaDB for vector storage and similarity search
            - Streamlit for the user interface
            - NetworkX and Plotly for visualizations
            
            **Features:**
            - Multi-modal document processing (PDF, DOCX, images, videos, web)
            - Advanced RAG with conversation memory
            - Interactive flowchart and mindmap generation
            - Real-time chat with document context
            - Secure local processing
            
            **Privacy & Security:**
            - Documents are processed locally
            - No data is shared with third parties
            - API keys are not stored permanently
            - All processing happens in your browser session
            """)
        
        # Help and Documentation
        with st.expander("🆘 Help & Troubleshooting"):
            st.markdown("""
            ### 🔧 Common Issues & Solutions
            
            **Problem: "No relevant information found"**
            - Solution: Try rephrasing your question or upload more relevant documents
            
            **Problem: Slow processing**
            - Solution: Reduce chunk size in settings or use smaller documents
            
            **Problem: Visualization not generating**
            - Solution: Be more specific in your visualization request
            
            **Problem: API errors**
            - Solution: Check your OpenAI API key and ensure sufficient credits
            
            ### 📚 Best Practices
            1. Upload high-quality, well-formatted documents
            2. Use specific, clear questions for better results
            3. Try different chat modes for different use cases
            4. Keep documents under 50MB for optimal performance
            
            ### 🔗 Useful Links
            - [OpenAI API Documentation](https://platform.openai.com/docs)
            - [Streamlit Documentation](https://docs.streamlit.io)
            - [DataMap AI GitHub Repository](#) (Coming soon)
            """)

    def __init__(self, chat_service: ChatService):
        super().__init__()
        self.chat_service = chat_service
