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
            
            st.mar
