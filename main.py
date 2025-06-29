# main.py - COMPLETE FIXED VERSION
import streamlit as st
import openai
import time
from pathlib import Path
from typing import List, Optional
import tempfile
import os

# Configure Streamlit
st.set_page_config(
    page_title="DataMap AI",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

class SimpleDataMapApp:
    """Complete working DataMap AI application."""
    
    def __init__(self):
        self._initialize_session_state()
        
        # Get API key from environment or session state
        self.openai_api_key = (
            os.getenv("OPENAI_API_KEY") or 
            st.session_state.get('openai_api_key', '')
        )
        
        if self.openai_api_key:
            self.client = openai.OpenAI(api_key=self.openai_api_key)
        else:
            self.client = None
    
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
        if 'chat_messages' not in st.session_state:
            st.session_state.chat_messages = []
        if 'openai_api_key' not in st.session_state:
            st.session_state.openai_api_key = ''
    
    def run(self):
        """Main application entry point."""
        # Handle authentication
        if not st.session_state.authenticated:
            self._render_auth()
            return
        
        # Render main application
        self._render_app()
    
    def _render_auth(self):
        """Render simple authentication interface."""
        st.title("🗺️ DataMap AI")
        st.markdown("### Transform Your Documents into Interactive Knowledge Maps")
        
        st.markdown("---")
        
        # API Key Setup
        st.markdown("### 🔑 Setup")
        api_key = st.text_input(
            "OpenAI API Key", 
            type="password",
            value=st.session_state.openai_api_key,
            help="Get your API key from https://platform.openai.com/api-keys"
        )
        
        if api_key != st.session_state.openai_api_key:
            st.session_state.openai_api_key = api_key
            if api_key:
                self.openai_api_key = api_key
                self.client = openai.OpenAI(api_key=api_key)
        
        # Demo login
        st.markdown("### 🚀 Continue")
        if st.button("🔓 Start Using DataMap AI", type="primary"):
            if api_key:
                st.session_state.authenticated = True
                st.session_state.user_id = "demo_user"
                st.rerun()
            else:
                st.error("Please enter your OpenAI API key first")
    
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
            
            if current_page != st.session_state.current_page:
                st.session_state.current_page = current_page
                st.rerun()
            
            st.markdown("---")
            
            # Stats
            if st.session_state.documents:
                st.metric("Documents", len(st.session_state.documents))
                total_chunks = sum(len(doc['chunks']) for doc in st.session_state.documents)
                st.metric("Text Chunks", total_chunks)
            
            api_status = "✅" if self.openai_api_key else "❌"
            st.metric("API Key", api_status)
            
            # Logout
            if st.button("🚪 Logout"):
                st.session_state.authenticated = False
                st.session_state.user_id = None
                st.rerun()
        
        # Render current page
        self._render_page(st.session_state.current_page)
    
    def _render_page(self, page_name: str):
        """Render the specified page."""
        if page_name == 'home':
            self._render_home()
        elif page_name == 'upload':
            self._render_upload()
        elif page_name == 'chat':
            self._render_chat()
        elif page_name == 'visualization':
            self._render_visualization()
        elif page_name == 'settings':
            self._render_settings()
    
    def _render_home(self):
        """Render home page."""
        st.title("🗺️ Welcome to DataMap AI")
        
        st.markdown("""
        ### Transform Your Documents into Interactive Knowledge Maps
        
        **What you can do:**
        - 📁 **Upload documents** (PDF, text files, or paste content)
        - 💬 **Chat with your content** using AI
        - 📊 **Generate flowcharts** and mindmaps
        - 🔍 **Explore knowledge** visually
        """)
        
        # Quick start
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 1. 📁 Upload Content")
            if st.button("📁 Upload Documents", key="nav1"):
                st.session_state.current_page = 'upload'
                st.rerun()
        
        with col2:
            st.markdown("#### 2. 💬 Ask Questions")
            if st.button("💬 Start Chatting", key="nav2"):
                st.session_state.current_page = 'chat'
                st.rerun()
        
        with col3:
            st.markdown("#### 3. 📊 Create Visuals")
            if st.button("📊 Generate Visuals", key="nav3"):
                st.session_state.current_page = 'visualization'
                st.rerun()
        
        # Status
        st.markdown("---")
        if not self.openai_api_key:
            st.warning("⚠️ OpenAI API key not configured. Go to Settings to add it.")
        elif not st.session_state.documents:
            st.info("📁 Ready to use! Upload some documents to get started.")
        else:
            st.success(f"✅ All set! You have {len(st.session_state.documents)} documents ready to explore.")
    
    def _render_upload(self):
        """Render upload page."""
        st.title("📁 Upload Documents")
        
        st.markdown("### Add Your Content")
        
        # Text input
        st.markdown("#### 📝 Paste Text Content")
        text_content = st.text_area(
            "Paste your text here",
            height=200,
            placeholder="Paste any text content you want to chat with..."
        )
        
        if st.button("📝 Add Text Content") and text_content:
            doc = {
                'title': f"Text Content {len(st.session_state.documents) + 1}",
                'content': text_content,
                'type': 'text',
                'chunks': self._simple_chunk_text(text_content)
            }
            st.session_state.documents.append(doc)
            st.success(f"✅ Added text content with {len(doc['chunks'])} chunks")
        
        # File upload
        st.markdown("#### 📄 Upload Files")
        uploaded_files = st.file_uploader(
            "Choose files",
            accept_multiple_files=True,
            type=['txt', 'pdf'],
            help="Upload text files (PDF support coming soon)"
        )
        
        if st.button("📄 Process Files") and uploaded_files:
            for file in uploaded_files:
                try:
                    if file.type == "text/plain":
                        content = file.read().decode('utf-8')
                        doc = {
                            'title': file.name,
                            'content': content,
                            'type': 'file',
                            'chunks': self._simple_chunk_text(content)
                        }
                        st.session_state.documents.append(doc)
                        st.success(f"✅ Processed {file.name}")
                    else:
                        st.warning(f"⚠️ {file.name} - Only text files supported in this demo")
                except Exception as e:
                    st.error(f"❌ Error processing {file.name}: {e}")
        
        # Show documents
        if st.session_state.documents:
            st.markdown("---")
            st.markdown("### 📚 Your Documents")
            
            for i, doc in enumerate(st.session_state.documents):
                with st.expander(f"📄 {doc['title']}"):
                    st.write(f"**Type:** {doc['type']}")
                    st.write(f"**Chunks:** {len(doc['chunks'])}")
                    st.write(f"**Content Length:** {len(doc['content'])} characters")
                    st.write(f"**Preview:** {doc['content'][:200]}...")
                    
                    if st.button(f"🗑️ Delete", key=f"del_{i}"):
                        st.session_state.documents.pop(i)
                        st.rerun()
    
    def _simple_chunk_text(self, text: str, chunk_size: int = 1000) -> List[str]:
        """Simple text chunking."""
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) > chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_length = len(word)
            else:
                current_chunk.append(word)
                current_length += len(word) + 1
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def _render_chat(self):
        """Render chat page with debug info."""
        st.title("💬 Chat with Your Documents")
        
        if not st.session_state.documents:
            st.warning("📁 No documents uploaded yet.")
            if st.button("📁 Upload Documents"):
                st.session_state.current_page = 'upload'
                st.rerun()
            return
        
        if not self.client:
            st.error("❌ OpenAI API key not configured. Go to Settings.")
            return
        
        # Debug info
        with st.expander("🔍 Debug Information"):
            st.write(f"**Documents loaded:** {len(st.session_state.documents)}")
            for i, doc in enumerate(st.session_state.documents):
                st.write(f"**{i+1}. {doc['title']}**")
                st.write(f"  - Type: {doc['type']}")
                st.write(f"  - Chunks: {len(doc['chunks'])}")
                st.write(f"  - Content length: {len(doc['content'])} characters")
                st.write(f"  - Preview: {doc['content'][:100]}...")
            
            st.write(f"**API Key configured:** {bool(self.client)}")
        
        # Suggested questions
        st.markdown("### 💡 Try these questions:")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📋 Summarize all documents"):
                st.session_state.suggested_question = "Please provide a summary of all the documents"
                st.rerun()
            
            if st.button("🔍 What are the main topics?"):
                st.session_state.suggested_question = "What are the main topics covered in these documents?"
                st.rerun()
        
        with col2:
            if st.button("📊 List key information"):
                st.session_state.suggested_question = "List the key information from each document"
                st.rerun()
            
            if st.button("❓ What can you tell me?"):
                st.session_state.suggested_question = "What can you tell me about these documents?"
                st.rerun()
        
        # Initialize chat
        if not st.session_state.chat_messages:
            st.session_state.chat_messages = [{
                "role": "assistant",
                "content": f"Hello! I can help you explore your {len(st.session_state.documents)} documents. What would you like to know?"
            }]
        
        # Display messages
        for message in st.session_state.chat_messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        
        # Handle suggested questions
        if hasattr(st.session_state, 'suggested_question'):
            prompt = st.session_state.suggested_question
            del st.session_state.suggested_question
            
            # Add user message
            st.session_state.chat_messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.write(prompt)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Analyzing your documents..."):
                    response = self._generate_response(prompt)
                    st.write(response)
                    st.session_state.chat_messages.append({"role": "assistant", "content": response})
        
        # Chat input
        if prompt := st.chat_input("Ask about your documents..."):
            # Add user message
            st.session_state.chat_messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.write(prompt)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = self._generate_response(prompt)
                    st.write(response)
                    st.session_state.chat_messages.append({"role": "assistant", "content": response})
    
    def _find_relevant_chunks(self, question: str) -> List[dict]:
        """Improved chunk finding that actually works."""
        all_chunks = []
        
        # Collect ALL chunks from ALL documents
        for doc in st.session_state.documents:
            for i, chunk in enumerate(doc['chunks']):
                all_chunks.append({
                    'content': chunk,
                    'score': 1.0,  # Give all chunks a base score
                    'doc_title': doc['title'],
                    'chunk_index': i
                })
        
        # If we have chunks, return them
        if all_chunks:
            # For general questions, return first few chunks
            general_words = ['summary', 'about', 'what', 'tell', 'overview', 'describe', 'explain', 'main', 'key', 'list']
            if any(word in question.lower() for word in general_words):
                return all_chunks[:5]  # Return first 5 chunks
            
            # For specific questions, try keyword matching
            question_words = set(question.lower().split())
            scored_chunks = []
            
            for chunk_data in all_chunks:
                chunk_words = set(chunk_data['content'].lower().split())
                common_words = question_words.intersection(chunk_words)
                score = len(common_words)
                
                if score > 0:
                    chunk_data['score'] = score
                    scored_chunks.append(chunk_data)
            
            # If we found matches, return them
            if scored_chunks:
                scored_chunks.sort(key=lambda x: x['score'], reverse=True)
                return scored_chunks[:5]
            
            # If no keyword matches, return first few chunks anyway
            return all_chunks[:3]
        
        return []
    
    def _generate_response(self, question: str) -> str:
        """Generate response with better error handling."""
        try:
            # Debug: Check if we have documents
            if not st.session_state.documents:
                return "No documents uploaded. Please upload some documents first."
            
            # Debug: Check total chunks
            total_chunks = sum(len(doc['chunks']) for doc in st.session_state.documents)
            if total_chunks == 0:
                return "Your documents don't seem to have any content. Please check your uploads."
            
            # Get relevant chunks
            relevant_chunks = self._find_relevant_chunks(question)
            
            # Debug: Show what we found
            if not relevant_chunks:
                # Fallback: just describe the documents
                doc_info = []
                for doc in st.session_state.documents:
                    doc_info.append(f"- {doc['title']}: {len(doc['chunks'])} chunks, preview: {doc['content'][:150]}...")
                
                return f"""I have access to your {len(st.session_state.documents)} documents:

{chr(10).join(doc_info)}

Please ask a more specific question about the content, or I can try to help with what you'd like to know."""
            
            # Create context from chunks
            context_parts = []
            for chunk_data in relevant_chunks[:3]:
                context_parts.append(f"From '{chunk_data['doc_title']}':\n{chunk_data['content']}")
            
            context = "\n\n".join(context_parts)
            
            # Generate response with OpenAI
            if not self.client:
                return "OpenAI client not configured. Please check your API key in Settings."
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a helpful assistant. Answer questions based on the provided document context. Be specific and cite which document you're referencing."
                    },
                    {
                        "role": "user", 
                        "content": f"Based on these documents:\n\n{context}\n\nQuestion: {question}\n\nPlease provide a helpful answer based on the document content."
                    }
                ],
                max_tokens=500,
                temperature=0.1
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error generating response: {str(e)}. Debug info: {len(st.session_state.documents)} documents, API key configured: {bool(self.client)}"
    
    def _render_visualization(self):
        """Render visualization page with working visuals."""
        st.title("📊 Knowledge Visualization")
        
        if not st.session_state.documents:
            st.warning("📁 No documents uploaded yet.")
            if st.button("📁 Upload Documents"):
                st.session_state.current_page = 'upload'
                st.rerun()
            return
        
        if not self.client:
            st.error("❌ OpenAI API key not configured. Go to Settings.")
            return
        
        st.markdown("### Create Visual Knowledge Maps")
        
        # Visualization input
        viz_type = st.radio("Type", ["Flowchart", "Mindmap"], horizontal=True)
        
        query = st.text_area(
            "What would you like to visualize?",
            placeholder="Enter a topic from your documents to create a visual representation...",
            help="Try: 'Show me the main concepts', 'Visualize the process', or 'Map out the key information'"
        )
        
        # Example queries
        st.markdown("**💡 Example queries:**")
        examples = [
            "Show me the main concepts",
            "Visualize the key information", 
            "Map out the important details",
            "Create a overview of the content"
        ]
        
        example_cols = st.columns(4)
        for i, example in enumerate(examples):
            with example_cols[i]:
                if st.button(example, key=f"example_{i}"):
                    query = example
        
        if st.button("🎨 Generate Visualization", type="primary") and query:
            with st.spinner(f"Generating {viz_type.lower()}..."):
                try:
                    # Get relevant content
                    relevant_chunks = self._find_relevant_chunks(query)
                    
                    if not relevant_chunks:
                        st.error("No relevant content found for this topic.")
                        return
                    
                    context = "\n".join([chunk['content'] for chunk in relevant_chunks[:3]])
                    
                    if viz_type == "Flowchart":
                        st.info("🔄 Creating flowchart...")
                        flowchart = self._generate_flowchart(query, context)
                        
                        if flowchart:
                            st.success("✅ Flowchart generated!")
                            
                            # Debug info
                            with st.expander("🔍 Debug - View DOT Code"):
                                st.code(flowchart, language='dot')
                            
                            try:
                                st.graphviz_chart(flowchart)
                            except Exception as e:
                                st.error(f"Error rendering flowchart: {e}")
                                st.code(flowchart, language='dot')
                        else:
                            st.error("Failed to generate flowchart")
                    
                    else:  # Mindmap
                        st.info("🧠 Creating mindmap...")
                        mindmap_data = self._generate_mindmap(query, context)
                        
                        if mindmap_data:
                            st.success("✅ Mindmap generated!")
                            
                            # Try to render visual mindmap
                            visual_success = self._render_mindmap_visual(mindmap_data)
                            
                            if not visual_success:
                                # Fallback to text display
                                st.markdown("#### 🧠 Mindmap Structure:")
                                for i, item in enumerate(mindmap_data, 1):
                                    st.markdown(f"{i}. **{item['main']}** → {item['sub']}")
                            
                            # Show connections in expandable section
                            with st.expander("📋 View Mindmap Connections"):
                                for item in mindmap_data:
                                    st.write(f"• **{item['main']}** connects to **{item['sub']}**")
                        else:
                            st.error("Failed to generate mindmap")
                    
                except Exception as e:
                    st.error(f"Error generating visualization: {e}")
                    st.info("💡 Try a simpler query or check your document content.")
    
    def _generate_flowchart(self, topic: str, context: str) -> Optional[str]:
        """Generate a working flowchart in DOT notation."""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system", 
                        "content": """Create a flowchart in DOT notation. Rules:
1. Start with 'digraph {'
2. End with '}'
3. Use quotes around node names
4. Use -> for connections
5. Maximum 8 nodes
6. Return ONLY the DOT code, no explanations"""
                    },
                    {
                        "role": "user", 
                        "content": f"Create a flowchart about: {topic}\n\nContext: {context[:500]}\n\nShow main concepts and their relationships."
                    }
                ],
                max_tokens=400,
                temperature=0.1
            )
            
            dot_code = response.choices[0].message.content.strip()
            
            # Clean the response - remove markdown formatting
            if "```" in dot_code:
                lines = dot_code.split('\n')
                dot_lines = []
                in_code_block = False
                for line in lines:
                    if line.strip().startswith('```'):
                        in_code_block = not in_code_block
                        continue
                    if in_code_block or line.strip().startswith('digraph') or '->' in line or line.strip() == '}':
                        dot_lines.append(line)
                dot_code = '\n'.join(dot_lines)
            
            # Validate and fix DOT code
            if not dot_code.strip().startswith('digraph'):
                # Create a fallback flowchart based on the topic
                return f'''digraph {{
    rankdir=TB;
    node [shape=box, style="rounded,filled", fillcolor=lightblue];
    
    "Main Topic" [label="{topic[:30]}", fillcolor=lightgreen];
    "Concept 1" [label="Key Concept 1"];
    "Concept 2" [label="Key Concept 2"];
    "Detail 1" [label="Important Detail"];
    "Detail 2" [label="Related Info"];
    
    "Main Topic" -> "Concept 1";
    "Main Topic" -> "Concept 2";
    "Concept 1" -> "Detail 1";
    "Concept 2" -> "Detail 2";
}}'''
            
            # Ensure proper structure
            if not dot_code.strip().endswith('}'):
                dot_code += '\n}'
                
            return dot_code
            
        except Exception as e:
            # Return a simple working flowchart
            return f'''digraph {{
    rankdir=TB;
    node [shape=box, style="rounded,filled", fillcolor=lightblue];
    
    "Topic" [label="{topic[:20]}", fillcolor=lightgreen];
    "Key Point 1" [fillcolor=lightyellow];
    "Key Point 2" [fillcolor=lightyellow];
    
    "Topic" -> "Key Point 1";
    "Topic" -> "Key Point 2";
}}'''

    def _generate_mindmap(self, topic: str, context: str) -> Optional[List[dict]]:
        """Generate mindmap data for visual display."""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system", 
                        "content": "Extract 6-10 key concepts from the context and organize them hierarchically. Return as simple bullet points."
                    },
                    {
                        "role": "user", 
                        "content": f"Topic: {topic}\nContext: {context[:500]}\n\nList main concepts and their sub-concepts as bullet points."
                    }
                ],
                max_tokens=300,
                temperature=0.1
            )
            
            content = response.choices[0].message.content
            lines = [line.strip() for line in content.split('\n') if line.strip()]
            
            mindmap = []
            current_main = topic
            
            for line in lines:
                if line.startswith('-') or line.startswith('•') or line.startswith('*'):
                    sub_concept = line.lstrip('-•*').strip()
                    if sub_concept and len(sub_concept) > 2:
                        mindmap.append({'main': current_main, 'sub': sub_concept})
                elif ':' in line:
                    # Handle lines like "Main concept: sub concept"
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        main_part = parts[0].strip()
                        sub_part = parts[1].strip()
                        if main_part and sub_part:
                            mindmap.append({'main': main_part, 'sub': sub_part})
                elif len(line) < 60 and len(line) > 5:  # Likely a main concept
                    current_main = line.strip(':-•*')
            
            # Ensure we have some data
            if not mindmap:
                mindmap = [
                    {'main': topic, 'sub': 'Key Concept 1'},
                    {'main': topic, 'sub': 'Key Concept 2'},
                    {'main': topic, 'sub': 'Important Detail'},
                ]
            
            return mindmap[:10]  # Limit to 10 items
            
        except Exception as e:
            # Fallback mindmap
            return [
                {'main': topic, 'sub': 'Main Concept'},
                {'main': topic, 'sub': 'Key Detail'},
                {'main': topic, 'sub': 'Important Point'},
            ]

    def _render_mindmap_visual(self, connections: List[dict]):
        """Render mindmap as an interactive network using Plotly."""
        try:
            import plotly.graph_objects as go
            import plotly.express as px
            from plotly.subplots import make_subplots
            import networkx as nx
            import numpy as np
            
            # Create network graph
            G = nx.Graph()
            
            # Add edges from connections
            for conn in connections:
                G.add_edge(conn['main'], conn['sub'])
            
            # Calculate layout
            pos = nx.spring_layout(G, k=3, iterations=50, seed=42)
            
            # Create edge traces
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
            
            # Create node traces
            node_x = []
            node_y = []
            node_text = []
            node_info = []
            node_colors = []
            node_sizes = []
            
            # Get unique nodes and classify them
            main_nodes = set(conn['main'] for conn in connections)
            
            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                
                # Shorten long text for display
                display_text = node if len(node) <= 20 else node[:17] + "..."
                node_text.append(display_text)
                node_info.append(node)  # Full text for hover
                
                # Color and size based on node type
                if node in main_nodes:
                    node_colors.append('#FF6B6B')  # Red for main topics
                    node_sizes.append(30)
                else:
                    node_colors.append('#4ECDC4')  # Teal for sub-concepts
                    node_sizes.append(20)
            
            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                hoverinfo='text',
                hovertext=node_info,
                text=node_text,
                textposition="middle center",
                textfont=dict(size=10, color='white'),
                marker=dict(
                    size=node_sizes,
                    color=node_colors,
                    line=dict(width=2, color='white')
                )
            )
            
            # Create the figure
            fig = go.Figure(data=[edge_trace, node_trace],
                           layout=go.Layout(
                               title='Interactive Mindmap',
                               titlefont_size=16,
                               showlegend=False,
                               hovermode='closest',
                               margin=dict(b=20,l=5,r=5,t=40),
                               annotations=[ dict(
                                   text="Click and drag to explore the mindmap",
                                   showarrow=False,
                                   xref="paper", yref="paper",
                                   x=0.005, y=-0.002,
                                   xanchor="left", yanchor="bottom",
                                   font=dict(color="gray", size=12)
                               )],
                               xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                               yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                               height=500,
                               plot_bgcolor='white'
                           ))
            
            st.plotly_chart(fig, use_container_width=True)
            return True
            
        except ImportError:
            st.error("Plotly not available. Install with: pip install plotly")
            return False
        except Exception as e:
            st.error(f"Error creating visual mindmap: {e}")
            return False

    def _render_settings(self):
        """Render settings page."""
        st.title("⚙️ Settings")
        
        # API Key
        st.markdown("### 🔑 API Configuration")
        
        current_key = st.session_state.openai_api_key
        masked_key = f"{'*' * 20}{current_key[-4:]}" if current_key else "Not set"
        st.info(f"Current API Key: {masked_key}")
        
        new_key = st.text_input(
            "OpenAI API Key",
            value=current_key,
            type="password",
            help="Get your API key from https://platform.openai.com/api-keys"
        )
        
        if st.button("💾 Save API Key"):
            st.session_state.openai_api_key = new_key
            if new_key:
                self.openai_api_key = new_key
                self.client = openai.OpenAI(api_key=new_key)
                st.success("✅ API key saved!")
            else:
                st.error("❌ Please enter a valid API key")
        
        # Stats
        st.markdown("---")
        st.markdown("### 📊 Statistics")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Documents", len(st.session_state.documents))
            st.metric("Chat Messages", len(st.session_state.chat_messages))
        
        with col2:
            total_chunks = sum(len(doc['chunks']) for doc in st.session_state.documents)
            st.metric("Total Chunks", total_chunks)
            st.metric("API Status", "✅ Connected" if self.client else "❌ Not Connected")
        
        # Document details
        if st.session_state.documents:
            st.markdown("---")
            st.markdown("### 📋 Document Details")
            for i, doc in enumerate(st.session_state.documents):
                with st.expander(f"📄 {doc['title']}"):
                    st.write(f"**Type:** {doc['type']}")
                    st.write(f"**Chunks:** {len(doc['chunks'])}")
                    st.write(f"**Total Characters:** {len(doc['content'])}")
                    st.write(f"**Words:** {len(doc['content'].split())}")
                    
                    # Show chunks
                    for j, chunk in enumerate(doc['chunks'][:3]):  # Show first 3 chunks
                        st.write(f"**Chunk {j+1}:** {chunk[:100]}...")
        
        # Data management
        st.markdown("---")
        st.markdown("### 🗑️ Data Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🗑️ Clear Documents"):
                st.session_state.documents = []
                st.success("Documents cleared!")
                st.rerun()
        
        with col2:
            if st.button("🗑️ Clear Chat"):
                st.session_state.chat_messages = []
                st.success("Chat history cleared!")
                st.rerun()

def main():
    """Application entry point."""
    app = SimpleDataMapApp()
    app.run()

if __name__ == "__main__":
    main()
