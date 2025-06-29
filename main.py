# main.py - MINIMAL WORKING VERSION (No Import Conflicts)
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
    """Minimal working DataMap AI application."""
    
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
        else:
            st.success("✅ Ready to use! Upload some documents to get started.")
    
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
        """Render chat page."""
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
    
    def _generate_response(self, question: str) -> str:
        """Generate response using simple RAG."""
        try:
            # Get relevant chunks
            relevant_chunks = self._find_relevant_chunks(question)
            
            if not relevant_chunks:
                return "I couldn't find relevant information in your documents to answer that question."
            
            # Create context
            context = "\n\n".join([f"Document: {chunk['doc_title']}\nContent: {chunk['content']}" 
                                 for chunk in relevant_chunks[:3]])
            
            # Generate response
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. Answer questions based on the provided context. If the context doesn't contain enough information, say so."},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"}
                ],
                max_tokens=500,
                temperature=0.1
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def _find_relevant_chunks(self, question: str) -> List[dict]:
        """Simple keyword-based chunk finding."""
        question_words = set(question.lower().split())
        scored_chunks = []
        
        for doc in st.session_state.documents:
            for chunk in doc['chunks']:
                chunk_words = set(chunk.lower().split())
                score = len(question_words.intersection(chunk_words))
                
                if score > 0:
                    scored_chunks.append({
                        'content': chunk,
                        'score': score,
                        'doc_title': doc['title']
                    })
        
        # Sort by score and return top chunks
        scored_chunks.sort(key=lambda x: x['score'], reverse=True)
        return scored_chunks[:5]
    
    def _render_visualization(self):
        """Render visualization page."""
        st.title("📊 Knowledge Visualization")
        
        if not st.session_state.documents:
            st.warning("📁 No documents uploaded yet.")
            return
        
        if not self.client:
            st.error("❌ OpenAI API key not configured.")
            return
        
        st.markdown("### Create Visual Knowledge Maps")
        
        # Visualization input
        viz_type = st.radio("Type", ["Flowchart", "Mindmap"], horizontal=True)
        
        query = st.text_area(
            "What would you like to visualize?",
            placeholder="Enter a topic from your documents to create a visual representation..."
        )
        
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
                        flowchart = self._generate_flowchart(query, context)
                        if flowchart:
                            st.success("✅ Flowchart generated!")
                            st.graphviz_chart(flowchart)
                    else:
                        mindmap = self._generate_mindmap(query, context)
                        if mindmap:
                            st.success("✅ Mindmap generated!")
                            for item in mindmap:
                                st.markdown(f"- **{item['main']}** → {item['sub']}")
                
                except Exception as e:
                    st.error(f"Error generating visualization: {e}")
    
    def _generate_flowchart(self, topic: str, context: str) -> Optional[str]:
        """Generate a simple flowchart."""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Create a simple flowchart in DOT notation. Return only the DOT code, no explanations."},
                    {"role": "user", "content": f"Topic: {topic}\nContext: {context}\n\nCreate a flowchart with maximum 6 nodes showing the main concepts and their relationships."}
                ],
                max_tokens=300
            )
            
            dot_code = response.choices[0].message.content.strip()
            
            # Basic validation
            if "digraph" in dot_code or "->" in dot_code:
                return dot_code
            else:
                # Fallback simple flowchart
                return f'''
                digraph {{
                    rankdir=TB;
                    "Topic: {topic[:20]}..." -> "Key Concept 1";
                    "Topic: {topic[:20]}..." -> "Key Concept 2";
                    "Key Concept 1" -> "Detail A";
                    "Key Concept 2" -> "Detail B";
                }}
                '''
        except:
            return None
    
    def _generate_mindmap(self, topic: str, context: str) -> Optional[List[dict]]:
        """Generate a simple mindmap structure."""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Extract main concepts and their relationships from the given context. Return a simple list."},
                    {"role": "user", "content": f"Topic: {topic}\nContext: {context}\n\nList the main concepts and sub-concepts related to this topic."}
                ],
                max_tokens=200
            )
            
            content = response.choices[0].message.content
            
            # Simple parsing - convert to structure
            lines = [line.strip() for line in content.split('\n') if line.strip()]
            mindmap = []
            
            current_main = topic
            for line in lines[:8]:  # Limit to 8 items
                if line.startswith('-') or line.startswith('•'):
                    sub_concept = line.lstrip('-•').strip()
                    mindmap.append({'main': current_main, 'sub': sub_concept})
                elif len(line) < 50:  # Likely a main concept
                    current_main = line
            
            return mindmap if mindmap else [{'main': topic, 'sub': 'Concept 1'}, {'main': topic, 'sub': 'Concept 2'}]
            
        except:
            return None
    
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
        
        # Data management
        st.markdown("---")
        st.markdown("### 🗑️ Data Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🗑️ Clear Documents"):
                st.session_state.documents = []
                st.success("Documents cleared!")
        
        with col2:
            if st.button("🗑️ Clear Chat"):
                st.session_state.chat_messages = []
                st.success("Chat history cleared!")

def main():
    """Application entry point."""
    app = SimpleDataMapApp()
    app.run()

if __name__ == "__main__":
    main()
