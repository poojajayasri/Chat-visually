# main.py - FINAL FIXED VERSION
import streamlit as st
from pathlib import Path

# Configure Streamlit first
st.set_page_config(
    page_title="DataMap AI",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Import with error handling
try:
    from src.config import load_config
    from src.auth.auth_manager import AuthManager
    from src.services.document_service import DocumentService
    from src.services.embedding_service import EmbeddingService
    from src.services.rag_service import RAGService, VisualizationService
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
        """Initialize core services with proper parameter matching."""
        try:
            # Document service
            self.document_service = DocumentService(self.config.storage)
            
            # Embedding service - takes (embedding_config, llm_config)
            self.embedding_service = EmbeddingService(
                self.config.embeddings,  # embedding_config
                self.config.llm         # llm_config
            )
            
            # RAG service - takes (llm_config, embedding_config, storage_config)
            self.rag_service = RAGService(
                self.config.llm,        # llm_config
                self.config.embeddings, # embedding_config
                self.config.storage     # storage_config
            )
            
            # Visualization service
            self.visualization_service = VisualizationService(self.rag_service)
            
        except Exception as e:
            st.error(f"Service initialization error: {e}")
            # Set to None so we can handle gracefully
            self.document_service = None
            self.embedding_service = None
            self.rag_service = None
            self.visualization_service = None
    
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
            # Handle authentication
            if not st.session_state.authenticated:
                self._render_auth()
                return
            
            # Render main application
            self._render_app()
            
        except Exception as e:
            st.error(f"Application error: {e}")
            st.info("🔧 This appears to be a setup issue. Please check your configuration.")
    
    def _render_auth(self):
        """Render authentication interface."""
        st.title("🗺️ DataMap AI")
        st.markdown("### Welcome to DataMap AI")
        
        # Simple demo login for now
        st.info("🚀 Demo Mode - Click below to continue")
        
        if st.button("🔓 Enter as Demo User", type="primary"):
            st.session_state.authenticated = True
            st.session_state.user_id = "demo_user"
            st.rerun()
        
        # Try full auth if available
        if self.auth_manager:
            st.markdown("---")
            try:
                auth_result = self.auth_manager.render_auth_ui()
                if auth_result.authenticated:
                    st.session_state.authenticated = True
                    st.session_state.user_id = auth_result.user_id
                    st.rerun()
            except Exception as e:
                st.error(f"Authentication system error: {e}")
    
    def _render_app(self):
        """Render main application interface."""
        # Sidebar
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
            
            # Quick stats
            if st.session_state.documents:
                st.metric("Documents", len(st.session_state.documents))
                total_chunks = sum(len(doc.chunks) for doc in st.session_state.documents)
                st.metric("Chunks", total_chunks)
            
            # System status
            services_available = sum([
                1 for service in [self.document_service, self.embedding_service, self.rag_service]
                if service is not None
            ])
            st.metric("Services", f"{services_available}/3")
            
            # Logout
            if st.button("🚪 Logout"):
                for key in list(st.session_state.keys()):
                    if key not in ['current_page']:
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
    
    def _render_home_page(self):
        """Render home page."""
        st.title("🗺️ Welcome to DataMap AI")
        
        st.markdown("""
        ### Transform Your Documents into Interactive Knowledge Maps
        
        Upload documents, ask questions, and generate beautiful visualizations from your content.
        """)
        
        # Quick start guide
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 1. 📁 Upload Documents")
            st.write("Start by uploading PDFs, documents, or other files.")
            if st.button("📁 Go to Upload", key="nav_upload"):
                st.session_state.current_page = 'upload'
                st.rerun()
        
        with col2:
            st.markdown("#### 2. 💬 Ask Questions")
            st.write("Chat with your documents using AI.")
            if st.button("💬 Start Chatting", key="nav_chat"):
                st.session_state.current_page = 'chat'
                st.rerun()
        
        with col3:
            st.markdown("#### 3. 📊 Create Visuals")
            st.write("Generate flowcharts and mindmaps.")
            if st.button("📊 Visualize", key="nav_viz"):
                st.session_state.current_page = 'visualization'
                st.rerun()
        
        # System status
        st.markdown("---")
        st.markdown("### 📊 System Status")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            status = "✅ Ready" if IMPORTS_SUCCESSFUL and self.config else "⚠️ Limited"
            st.metric("System", status)
        
        with col2:
            docs_count = len(st.session_state.documents)
            st.metric("Documents", docs_count)
        
        with col3:
            api_status = "✅ Set" if self.config and self.config.llm.openai_api_key else "❌ Missing"
            st.metric("API Key", api_status)
    
    def _render_upload_page(self):
        """Render upload page."""
        st.title("📁 Document Upload")
        
        if not self.document_service:
            st.warning("⚠️ Upload service not available. Please check configuration.")
            return
        
        st.markdown("### Upload Your Documents")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Choose files",
            accept_multiple_files=True,
            type=['pdf', 'docx', 'txt', 'csv', 'png', 'jpg'],
            help="Upload documents to analyze and chat with"
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
                            try:
                                file_path = self.document_service.save_uploaded_file(file, user_id)
                                result = self.document_service.process_document(file_path, user_id, file.name)
                                
                                if result.success:
                                    processed_docs.append(result.document)
                                    if self.rag_service:
                                        self.rag_service.add_document(result.document)
                                    st.success(f"✅ {file.name} processed ({result.chunks_created} chunks)")
                                else:
                                    st.error(f"❌ Error processing {file.name}: {result.error}")
                            except Exception as e:
                                st.error(f"❌ Error with {file.name}: {str(e)}")
                        
                        # Process URLs
                        if youtube_url:
                            try:
                                result = self.document_service.process_url(youtube_url, user_id, "youtube")
                                if result.success:
                                    processed_docs.append(result.document)
                                    if self.rag_service:
                                        self.rag_service.add_document(result.document)
                                    st.success(f"✅ YouTube video processed ({result.chunks_created} chunks)")
                                else:
                                    st.error(f"❌ Error processing YouTube: {result.error}")
                            except Exception as e:
                                st.error(f"❌ Error with YouTube URL: {str(e)}")
                        
                        if web_url:
                            try:
                                result = self.document_service.process_url(web_url, user_id, "web")
                                if result.success:
                                    processed_docs.append(result.document)
                                    if self.rag_service:
                                        self.rag_service.add_document(result.document)
                                    st.success(f"✅ Website processed ({result.chunks_created} chunks)")
                                else:
                                    st.error(f"❌ Error processing website: {result.error}")
                            except Exception as e:
                                st.error(f"❌ Error with website: {str(e)}")
                        
                        # Add to session state
                        st.session_state.documents.extend(processed_docs)
                        
                        if processed_docs:
                            st.balloons()
                            st.success(f"🎉 Successfully processed {len(processed_docs)} documents!")
                            
                    except Exception as e:
                        st.error(f"Processing error: {e}")
            else:
                st.warning("Please upload files or provide URLs to process.")
        
        # Show uploaded documents
        if st.session_state.documents:
            st.markdown("---")
            st.markdown("### 📚 Uploaded Documents")
            for i, doc in enumerate(st.session_state.documents):
                with st.expander(f"📄 {doc.metadata.title}"):
                    col1, col2, col3 = st.columns([2, 2, 1])
                    
                    with col1:
                        st.write(f"**Type:** {doc.metadata.file_type}")
                        st.write(f"**Size:** {doc.metadata.file_size / 1024:.1f} KB")
                    
                    with col2:
                        st.write(f"**Chunks:** {len(doc.chunks)}")
                        st.write(f"**Created:** {doc.metadata.created_at.strftime('%Y-%m-%d %H:%M')}")
                    
                    with col3:
                        if st.button(f"🗑️ Delete", key=f"delete_{i}"):
                            st.session_state.documents.remove(doc)
                            st.success(f"Deleted {doc.metadata.title}")
                            st.rerun()
    
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
            ["Quick Q&A", "Conversational", "Generate Flowchart", "Generate Mindmap"],
            help="Choose how you want to interact with your documents"
        )
        
        # Initialize chat history
        if 'chat_messages' not in st.session_state:
            st.session_state.chat_messages = [
                {
                    "role": "assistant", 
                    "content": f"Hello! I can help you explore your {len(st.session_state.documents)} uploaded documents. What would you like to know?"
                }
            ]
        
        # Display chat messages
        for message in st.session_state.chat_messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                
                # Show sources if available
                if message.get("sources"):
                    with st.expander("📚 Sources"):
                        for i, source in enumerate(message["sources"], 1):
                            st.write(f"**Source {i}:** {source[:150]}...")
        
        # Chat input
        if prompt := st.chat_input("Ask a question about your documents..."):
            # Add user message
            st.session_state.chat_messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.write(prompt)
            
            # Generate response
            with st.chat_message("assistant"):
                if self.rag_service and mode in ["Quick Q&A", "Conversational"]:
                    try:
                        user_id = st.session_state.user_id
                        response = self.rag_service.query(prompt, user_id)
                        
                        st.write(response.answer)
                        
                        # Prepare sources for storage
                        sources = []
                        if response.sources:
                            with st.expander("📚 Sources"):
                                for i, source in enumerate(response.sources[:3], 1):
                                    source_text = source.content[:200] + "..." if len(source.content) > 200 else source.content
                                    st.write(f"**Source {i}:** {source_text}")
                                    sources.append(source.content)
                        
                        # Show processing time
                        st.caption(f"⏱️ Processed in {response.processing_time:.2f} seconds")
                        
                        st.session_state.chat_messages.append({
                            "role": "assistant", 
                            "content": response.answer,
                            "sources": sources
                        })
                        
                    except Exception as e:
                        error_msg = f"I encountered an error: {str(e)}"
                        st.error(error_msg)
                        st.session_state.chat_messages.append({"role": "assistant", "content": error_msg})
                
                elif self.visualization_service and mode in ["Generate Flowchart", "Generate Mindmap"]:
                    try:
                        user_id = st.session_state.user_id
                        
                        if mode == "Generate Flowchart":
                            answer, flowchart_dot = self.visualization_service.generate_flowchart(prompt, user_id)
                            
                            if flowchart_dot:
                                st.write(f"Here's a flowchart for: {prompt}")
                                st.graphviz_chart(flowchart_dot)
                                st.session_state.chat_messages.append({
                                    "role": "assistant", 
                                    "content": f"Generated flowchart for: {prompt}\n\n{answer}"
                                })
                            else:
                                error_msg = "I couldn't generate a flowchart for this topic."
                                st.error(error_msg)
                                st.session_state.chat_messages.append({"role": "assistant", "content": error_msg})
                        
                        else:  # Generate Mindmap
                            answer, mindmap_connections = self.visualization_service.generate_mindmap(prompt, user_id)
                            
                            if mindmap_connections:
                                st.write(f"Here's a mindmap for: {prompt}")
                                
                                # Simple mindmap display
                                st.markdown("**Mindmap Structure:**")
                                for connection in mindmap_connections[:10]:  # Limit display
                                    source = connection.get('source', 'Unknown')
                                    target = connection.get('target', 'Unknown')
                                    st.markdown(f"- {source} → {target}")
                                
                                st.session_state.chat_messages.append({
                                    "role": "assistant", 
                                    "content": f"Generated mindmap for: {prompt}\n\n{answer}"
                                })
                            else:
                                error_msg = "I couldn't generate a mindmap for this topic."
                                st.error(error_msg)
                                st.session_state.chat_messages.append({"role": "assistant", "content": error_msg})
                                
                    except Exception as e:
                        error_msg = f"Visualization error: {str(e)}"
                        st.error(error_msg)
                        st.session_state.chat_messages.append({"role": "assistant", "content": error_msg})
                
                else:
                    # Fallback response
                    fallback_msg = f"I can see you're asking about: '{prompt}'. I have access to {len(st.session_state.documents)} documents, but some services may not be fully available."
                    st.write(fallback_msg)
                    st.session_state.chat_messages.append({"role": "assistant", "content": fallback_msg})
        
        # Chat controls
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🗑️ Clear Chat"):
                st.session_state.chat_messages = []
                st.rerun()
        
        with col2:
            if st.button("💾 Export Chat"):
                chat_text = "\n".join([f"{msg['role'].title()}: {msg['content']}" for msg in st.session_state.chat_messages])
                st.download_button(
                    "Download Chat History",
                    data=chat_text,
                    file_name="datamap_chat.txt",
                    mime="text/plain"
                )
    
    def _render_visualization_page(self):
        """Render visualization page."""
        st.title("📊 Knowledge Visualization")
        
        if not st.session_state.documents:
            st.warning("📁 No documents uploaded yet. Please upload documents first.")
            return
        
        st.markdown("### Create Visual Knowledge Maps")
        
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
                    try:
                        user_id = st.session_state.user_id
                        
                        if viz_type == "Flowchart" and self.visualization_service:
                            answer, flowchart_dot = self.visualization_service.generate_flowchart(query, user_id)
                            
                            if flowchart_dot:
                                st.success("✅ Flowchart generated successfully!")
                                st.graphviz_chart(flowchart_dot)
                                
                                with st.expander("📝 Description"):
                                    st.write(answer)
                            else:
                                st.error("Could not generate flowchart. Try a more specific query.")
                        
                        elif viz_type == "Mindmap" and self.visualization_service:
                            answer, mindmap_connections = self.visualization_service.generate_mindmap(query, user_id)
                            
                            if mindmap_connections:
                                st.success("✅ Mindmap generated successfully!")
                                
                                # Display mindmap as structured text
                                st.markdown("### 🧠 Mindmap Structure")
                                for connection in mindmap_connections:
                                    source = connection.get('source', 'Unknown')
                                    target = connection.get('target', 'Unknown')
                                    st.markdown(f"**{source}** → {target}")
                                
                                with st.expander("📝 Description"):
                                    st.write(answer)
                            else:
                                st.error("Could not generate mindmap. Try a more specific query.")
                        
                        else:
                            # Fallback visualization
                            st.success("✅ Simple visualization generated!")
                            
                            if viz_type == "Flowchart":
                                st.graphviz_chart(f'''
                                digraph {{
                                    rankdir=TB;
                                    node [shape=box, style=rounded];
                                    
                                    "Query: {query[:30]}..." [style=filled, fillcolor=lightblue];
                                    "Concept 1" [style=filled, fillcolor=lightgreen];
                                    "Concept 2" [style=filled, fillcolor=lightgreen];
                                    "Detail A" [style=filled, fillcolor=lightyellow];
                                    "Detail B" [style=filled, fillcolor=lightyellow];
                                    
                                    "Query: {query[:30]}..." -> "Concept 1";
                                    "Query: {query[:30]}..." -> "Concept 2";
                                    "Concept 1" -> "Detail A";
                                    "Concept 2" -> "Detail B";
                                }}
                                ''')
                            else:
                                st.markdown("### 🧠 Sample Mindmap Structure")
                                st.markdown(f"- **Main Topic:** {query}")
                                st.markdown("  - Subtopic 1")
                                st.markdown("  - Subtopic 2")
                                st.markdown("    - Detail A")
                                st.markdown("    - Detail B")
                        
                    except Exception as e:
                        st.error(f"Visualization error: {e}")
            else:
                st.warning("Please enter a query to generate a visualization.")
    
    def _render_settings_page(self):
        """Render settings page."""
        st.title("⚙️ Settings & Configuration")
        
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
                st.info("⚠️ Restart the app to use the new API key.")
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
            services_count = sum([
                1 for service in [self.document_service, self.embedding_service, self.rag_service]
                if service is not None
            ])
            st.metric("Services Available", f"{services_count}/3")
            st.metric("Import Status", "✅" if IMPORTS_SUCCESSFUL else "❌")
        
        # RAG Statistics
        if self.rag_service:
            try:
                stats = self.rag_service.get_stats()
                st.markdown("### 📈 RAG System Statistics")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Chunks", stats.get('total_chunks', 0))
                with col2:
                    st.metric("Index Size", stats.get('index_size', 0))
            except Exception:
                pass
        
        # Clear Data
        st.markdown("---")
        st.markdown("### 🗑️ Data Management")
        
        st.warning("⚠️ These actions cannot be undone!")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Clear Chat History"):
                st.session_state.chat_messages = []
                st.success("Chat history cleared!")
        
        with col2:
            if st.button("Clear Documents"):
                st.session_state.documents = []
                st.success("Documents cleared!")
        
        with col3:
            if st.button("Reset All Data"):
                keys_to_keep = ['authenticated', 'user_id', 'current_page']
                keys_to_delete = [k for k in st.session_state.keys() if k not in keys_to_keep]
                
                for key in keys_to_delete:
                    del st.session_state[key]
                
                st.success("All data reset!")
                st.rerun()

def main():
    """Application entry point."""
    app = DataMapApp()
    app.run()

if __name__ == "__main__":
    main()
