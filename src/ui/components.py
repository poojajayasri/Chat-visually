# src/ui/components.py
import streamlit as st
import graphviz
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
from typing import List, Dict, Optional, Any
from datetime import datetime
import json

from ..models.chat import ChatMessage, MessageType
from ..models.document import Document, DocumentMetadata
from ..services.chat_service import ChatModeManager

class Sidebar:
    """Modern sidebar navigation component."""
    
    def render(self) -> str:
        """Render sidebar and return selected page."""
        with st.sidebar:
            # App logo and title
            st.markdown("""
                <div style="text-align: center; padding: 20px 0;">
                    <h1 style="color: #2E4057; margin: 0;">🗺️ DataMap AI</h1>
                    <p style="color: #666; margin: 5px 0;">Visual Knowledge Explorer</p>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Navigation menu
            pages = {
                "🏠 Home": "home",
                "📁 Upload Documents": "upload", 
                "💬 Chat": "chat",
                "📊 Visualization": "visualization",
                "⚙️ Settings": "settings"
            }
            
            selected_page = st.radio(
                "Navigation",
                options=list(pages.keys()),
                index=0,
                label_visibility="collapsed"
            )
            
            st.markdown("---")
            
            # Quick stats if user has documents
            if 'documents' in st.session_state and st.session_state.documents:
                st.markdown("### 📈 Quick Stats")
                st.metric("Documents", len(st.session_state.documents))
                
                total_chunks = sum(len(doc.chunks) for doc in st.session_state.documents)
                st.metric("Text Chunks", total_chunks)
            
            st.markdown("---")
            
            # User info and logout
            if st.session_state.get('authenticated', False):
                user_id = st.session_state.get('user_id', 'User')
                st.markdown(f"**Logged in as:** {user_id}")
                
                if st.button("🚪 Logout", use_container_width=True):
                    for key in list(st.session_state.keys()):
                        del st.session_state[key]
                    st.rerun()
            
            return pages[selected_page]

class DocumentUploader:
    """Document upload component with progress tracking."""
    
    def __init__(self, document_service, embedding_service):
        self.document_service = document_service
        self.embedding_service = embedding_service
    
    def render(self) -> List[Document]:
        """Render document upload interface."""
        uploaded_docs = []
        
        st.markdown("### 📁 Upload Documents")
        st.markdown("Support formats: PDF, DOCX, CSV, Images (PNG, JPG), URLs")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Choose files",
            accept_multiple_files=True,
            type=['pdf', 'docx', 'csv', 'png', 'jpg', 'jpeg'],
            help="Upload multiple files at once"
        )
        
        # URL inputs
        col1, col2 = st.columns(2)
        with col1:
            youtube_url = st.text_input(
                "YouTube URL",
                placeholder="https://youtube.com/watch?v=..."
            )
        
        with col2:
            web_url = st.text_input(
                "Website URL", 
                placeholder="https://example.com"
            )
        
        # Process uploads
        if st.button("🚀 Process Documents", type="primary", use_container_width=True):
            user_id = st.session_state.get('user_id', 'anonymous')
            
            # Create progress bar
            total_items = len(uploaded_files) + (1 if youtube_url else 0) + (1 if web_url else 0)
            if total_items == 0:
                st.warning("Please upload files or provide URLs to process.")
                return uploaded_docs
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            processed = 0
            
            # Process files
            for file in uploaded_files:
                status_text.text(f"Processing {file.name}...")
                
                # Validate file
                is_valid, error_msg = self.document_service.validate_file(file)
                if not is_valid:
                    st.error(f"Error with {file.name}: {error_msg}")
                    continue
                
                try:
                    # Save and process file
                    file_path = self.document_service.save_uploaded_file(file, user_id)
                    result = self.document_service.process_document(file_path, user_id, file.name)
                    
                    if result.success:
                        # Add to RAG system
                        self.embedding_service.add_document(result.document)
                        uploaded_docs.append(result.document)
                        st.success(f"✅ {file.name} processed ({result.chunks_created} chunks)")
                    else:
                        st.error(f"❌ Error processing {file.name}: {result.error}")
                
                except Exception as e:
                    st.error(f"❌ Unexpected error with {file.name}: {str(e)}")
                
                processed += 1
                progress_bar.progress(processed / total_items)
            
            # Process URLs
            if youtube_url:
                status_text.text("Processing YouTube video...")
                try:
                    result = self.document_service.process_url(youtube_url, user_id, "youtube")
                    if result.success:
                        self.embedding_service.add_document(result.document)
                        uploaded_docs.append(result.document)
                        st.success(f"✅ YouTube video processed ({result.chunks_created} chunks)")
                    else:
                        st.error(f"❌ Error processing YouTube URL: {result.error}")
                except Exception as e:
                    st.error(f"❌ Error with YouTube URL: {str(e)}")
                
                processed += 1
                progress_bar.progress(processed / total_items)
            
            if web_url:
                status_text.text("Processing website...")
                try:
                    result = self.document_service.process_url(web_url, user_id, "web")
                    if result.success:
                        self.embedding_service.add_document(result.document)
                        uploaded_docs.append(result.document)
                        st.success(f"✅ Website processed ({result.chunks_created} chunks)")
                    else:
                        st.error(f"❌ Error processing website: {result.error}")
                except Exception as e:
                    st.error(f"❌ Error with website URL: {str(e)}")
                
                processed += 1
                progress_bar.progress(processed / total_items)
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            if uploaded_docs:
                st.balloons()
                st.success(f"🎉 Successfully processed {len(uploaded_docs)} documents!")
        
        return uploaded_docs

class ChatInterface:
    """Modern chat interface component."""
    
    def __init__(self, chat_service):
        self.chat_service = chat_service
    
    def render(self, session_id: str):
        """Render chat interface."""
        user_id = st.session_state.get('user_id', 'anonymous')
        
        # Chat mode selector
        st.markdown("### 💬 Chat Mode")
        mode_options = {
            info['name']: mode for mode, info in ChatModeManager.get_all_modes().items()
        }
        
        selected_mode_name = st.selectbox(
            "Choose chat mode",
            options=list(mode_options.keys()),
            help="Different modes offer different capabilities"
        )
        
        selected_mode = mode_options[selected_mode_name]
        mode_info = ChatModeManager.get_mode_info(selected_mode)
        
        st.info(f"**{mode_info['name']}**: {mode_info['description']}")
        
        # Chat container
        chat_container = st.container()
        
        # Display conversation history
        with chat_container:
            messages = self.chat_service.get_conversation_history(session_id)
            
            for message in messages:
                self._render_message(message)
        
        # Chat input
        with st.container():
            col1, col2 = st.columns([6, 1])
            
            with col1:
                user_input = st.text_input(
                    "Message",
                    placeholder=f"Ask a question in {mode_info['name']} mode...",
                    key=f"chat_input_{session_id}",
                    label_visibility="collapsed"
                )
            
            with col2:
                send_button = st.button("Send", type="primary", use_container_width=True)
            
            # Clear conversation button
            if st.button("🗑️ Clear Conversation"):
                self.chat_service.clear_conversation(session_id)
                st.rerun()
        
        # Process user input
        if send_button and user_input.strip():
            with st.spinner("Processing..."):
                response = self.chat_service.process_message(
                    session_id=session_id,
                    user_id=user_id,
                    message=user_input,
                    message_type=mode_info['default_type']
                )
                
                # Clear input and refresh
                st.session_state[f"chat_input_{session_id}"] = ""
                st.rerun()
    
    def _render_message(self, message: ChatMessage):
        """Render a single chat message."""
        if message.type == MessageType.USER:
            with st.chat_message("user"):
                st.write(message.content)
        
        elif message.type == MessageType.TEXT:
            with st.chat_message("assistant"):
                st.write(message.content)
                
                # Show sources if available
                if message.metadata and message.metadata.get('sources'):
                    with st.expander("📚 Sources"):
                        for source in message.metadata['sources']:
                            st.write(f"• {source}")
        
        elif message.type == MessageType.FLOWCHART:
            with st.chat_message("assistant"):
                st.write(message.content)
                
                # Render flowchart
                if message.metadata and message.metadata.get('visualization_data'):
                    viz_data = message.metadata['visualization_data']
                    if viz_data.get('dot_notation'):
                        try:
                            st.graphviz_chart(viz_data['dot_notation'])
                        except Exception as e:
                            st.error(f"Error rendering flowchart: {e}")
        
        elif message.type == MessageType.MINDMAP:
            with st.chat_message("assistant"):
                st.write(message.content)
                
                # Render mindmap
                if message.metadata and message.metadata.get('visualization_data'):
                    viz_data = message.metadata['visualization_data']
                    connections = viz_data.get('connections', [])
                    if connections:
                        self._render_mindmap(connections)
        
        elif message.type == MessageType.ERROR:
            with st.chat_message("assistant"):
                st.error(message.content)
    
    def _render_mindmap(self, connections: List[Dict]):
        """Render mindmap using plotly."""
        try:
            # Create network graph
            G = nx.Graph()
            for conn in connections:
                G.add_edge(conn['source'], conn['target'])
            
            # Calculate layout
            pos = nx.spring_layout(G, k=1, iterations=50)
            
            # Create plotly figure
            edge_x = []
            edge_y = []
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
            
            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=2, color='#888'),
                hoverinfo='none',
                mode='lines'
            )
            
            node_x = []
            node_y = []
            node_text = []
            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                node_text.append(node)
            
            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                hoverinfo='text',
                text=node_text,
                textposition="middle center",
                marker=dict(
                    size=20,
                    color='lightblue',
                    line=dict(width=2, color='black')
                )
            )
            
            fig = go.Figure(data=[edge_trace, node_trace],
                          layout=go.Layout(
                              showlegend=False,
                              hovermode='closest',
                              margin=dict(b=20,l=5,r=5,t=40),
                              annotations=[ dict(
                                  text="Mindmap",
                                  showarrow=False,
                                  xref="paper", yref="paper",
                                  x=0.005, y=-0.002,
                                  xanchor="left", yanchor="bottom",
                                  font=dict(color="black", size=12)
                              )],
                              xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                              yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                              height=400
                          ))
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error rendering mindmap: {e}")

class DocumentManager:
    """Component for managing uploaded documents."""
    
    def render(self, documents: List[Document]):
        """Render document management interface."""
        if not documents:
            st.info("No documents uploaded yet. Go to the Upload page to add documents.")
            return
        
        st.markdown("### 📚 Document Library")
        
        # Document list
        for i, doc in enumerate(documents):
            with st.expander(f"📄 {doc.metadata.title}"):
                col1, col2, col3 = st.columns([2, 2, 1])
                
                with col1:
                    st.write(f"**Type:** {doc.metadata.file_type}")
                    st.write(f"**Size:** {doc.metadata.file_size / 1024:.1f} KB")
                
                with col2:
                    st.write(f"**Chunks:** {len(doc.chunks)}")
                    st.write(f"**Created:** {doc.metadata.created_at.strftime('%Y-%m-%d %H:%M')}")
                
                with col3:
                    if st.button(f"🗑️ Delete", key=f"delete_{doc.metadata.id}"):
                        # Remove from session state
                        st.session_state.documents = [
                            d for d in st.session_state.documents 
                            if d.metadata.id != doc.metadata.id
                        ]
                        st.success(f"Deleted {doc.metadata.title}")
                        st.rerun()

class VisualizationPanel:
    """Panel for creating and managing visualizations."""
    
    def __init__(self, chat_service):
        self.chat_service = chat_service
    
    def render(self):
        """Render visualization creation panel."""
        st.markdown("### 📊 Create Visualizations")
        
        # Visualization type selector
        viz_type = st.radio(
            "Visualization Type",
            ["Flowchart", "Mindmap"],
            horizontal=True
        )
        
        # Input for visualization query
        query = st.text_area(
            "What would you like to visualize?",
            placeholder="Enter a topic or question to create a visual representation...",
            height=100
        )
        
        # Generate button
        if st.button("🎨 Generate Visualization", type="primary"):
            if not query.strip():
                st.warning("Please enter a query to generate a visualization.")
                return
            
            user_id = st.session_state.get('user_id', 'anonymous')
            session_id = f"viz_{user_id}"
            
            with st.spinner(f"Generating {viz_type.lower()}..."):
                if viz_type == "Flowchart":
                    message_type = MessageType.FLOWCHART
                else:
                    message_type = MessageType.MINDMAP
                
                response = self.chat_service.process_message(
                    session_id=session_id,
                    user_id=user_id,
                    message=query,
                    message_type=message_type
                )
                
                if response.error:
                    st.error(f"Error generating {viz_type.lower()}: {response.error}")
                else:
                    st.success(f"{viz_type} generated successfully!")
                    
                    # Display the visualization
                    if response.visualization_data:
                        if viz_type == "Flowchart" and response.visualization_data.get('dot_notation'):
                            st.graphviz_chart(response.visualization_data['dot_notation'])
                        elif viz_type == "Mindmap" and response.visualization_data.get('connections'):
                            chat_interface = ChatInterface(self.chat_service)
                            chat_interface._render_mindmap(response.visualization_data['connections'])
                    
                    # Show description
                    with st.expander("📝 Description"):
                        st.write(response.message)
                    
                    # Show performance metrics
                    st.info(f"⏱️ Generated in {response.processing_time:.2f} seconds")

class StatsPanel:
    """Panel for displaying application statistics."""
    
    def render(self, chat_service):
        """Render statistics panel."""
        st.markdown("### 📊 Application Statistics")
        
        # Get stats
        rag_stats = chat_service.get_rag_stats()
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Total Documents",
                len(st.session_state.get('documents', []))
            )
        
        with col2:
            st.metric(
                "Text Chunks",
                rag_stats.get('total_chunks', 0)
            )
        
        with col3:
            sessions = len(set(
                msg.session_id for msg in chat_service.memory.conversations.values()
                for msg in msg if hasattr(msg, 'session_id')
            ))
            st.metric("Chat Sessions", sessions)
        
        # Document type breakdown
        if st.session_state.get('documents'):
            st.markdown("#### 📁 Document Types")
            
            type_counts = {}
            for doc in st.session_state.documents:
                file_type = doc.metadata.file_type
                type_counts[file_type] = type_counts.get(file_type, 0) + 1
            
            # Create pie chart
            fig = px.pie(
                values=list(type_counts.values()),
                names=list(type_counts.keys()),
                title="Documents by Type"
            )
            st.plotly_chart(fig, use_container_width=True)

class AlertSystem:
    """System for displaying user alerts and notifications."""
    
    @staticmethod
    def success(message: str):
        """Display success alert."""
        st.success(f"✅ {message}")
    
    @staticmethod
    def error(message: str):
        """Display error alert."""
        st.error(f"❌ {message}")
    
    @staticmethod
    def warning(message: str):
        """Display warning alert."""
        st.warning(f"⚠️ {message}")
    
    @staticmethod
    def info(message: str):
        """Display info alert."""
        st.info(f"ℹ️ {message}")
    
    @staticmethod
    def show_processing(message: str = "Processing..."):
        """Show processing spinner."""
        return st.spinner(message)
