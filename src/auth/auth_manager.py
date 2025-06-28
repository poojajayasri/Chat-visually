# src/auth/auth_manager.py
import streamlit as st
from dataclasses import dataclass
from typing import Optional
import hashlib
import json
from pathlib import Path

@dataclass
class AuthResult:
    """Result of authentication attempt."""
    authenticated: bool
    user_id: Optional[str] = None
    error: Optional[str] = None

class AuthManager:
    """Simple authentication manager for DataMap AI."""
    
    def __init__(self, config):
        self.config = config
        self.users_file = Path("data/users.json")
        self.users_file.parent.mkdir(exist_ok=True)
        
        # Create default users file if it doesn't exist
        if not self.users_file.exists():
            self._create_default_users()
    
    def _create_default_users(self):
        """Create default users file."""
        default_users = {
            "demo@datamap.ai": {
                "password_hash": self._hash_password("demo123"),
                "user_id": "demo_user",
                "created_at": "2024-01-01T00:00:00"
            }
        }
        
        with open(self.users_file, 'w') as f:
            json.dump(default_users, f, indent=2)
    
    def _hash_password(self, password: str) -> str:
        """Hash password using SHA256."""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def _load_users(self) -> dict:
        """Load users from file."""
        try:
            with open(self.users_file, 'r') as f:
                return json.load(f)
        except Exception:
            return {}
    
    def _save_users(self, users: dict):
        """Save users to file."""
        with open(self.users_file, 'w') as f:
            json.dump(users, f, indent=2)
    
    def authenticate(self, email: str, password: str) -> AuthResult:
        """Authenticate user with email and password."""
        users = self._load_users()
        
        if email not in users:
            return AuthResult(authenticated=False, error="User not found")
        
        password_hash = self._hash_password(password)
        if users[email]["password_hash"] != password_hash:
            return AuthResult(authenticated=False, error="Invalid password")
        
        return AuthResult(
            authenticated=True,
            user_id=users[email]["user_id"]
        )
    
    def register(self, email: str, password: str, user_id: str) -> AuthResult:
        """Register new user."""
        users = self._load_users()
        
        if email in users:
            return AuthResult(authenticated=False, error="User already exists")
        
        users[email] = {
            "password_hash": self._hash_password(password),
            "user_id": user_id,
            "created_at": "2024-01-01T00:00:00"
        }
        
        self._save_users(users)
        
        return AuthResult(authenticated=True, user_id=user_id)
    
    def render_auth_ui(self) -> AuthResult:
        """Render authentication UI."""
        st.title("🗺️ DataMap AI - Login")
        
        tab1, tab2 = st.tabs(["Login", "Register"])
        
        with tab1:
            st.markdown("### 🔐 Login to Your Account")
            
            # Demo credentials info
            st.info("**Demo Account**: email: `demo@datamap.ai`, password: `demo123`")
            
            email = st.text_input("Email", key="login_email")
            password = st.text_input("Password", type="password", key="login_password")
            
            if st.button("Login", type="primary", use_container_width=True):
                if email and password:
                    result = self.authenticate(email, password)
                    if result.authenticated:
                        st.success("✅ Login successful!")
                        return result
                    else:
                        st.error(f"❌ {result.error}")
                else:
                    st.error("Please enter both email and password")
        
        with tab2:
            st.markdown("### 📝 Create New Account")
            
            new_email = st.text_input("Email", key="register_email")
            new_password = st.text_input("Password", type="password", key="register_password")
            confirm_password = st.text_input("Confirm Password", type="password", key="confirm_password")
            user_id = st.text_input("Username", key="register_username")
            
            if st.button("Register", type="primary", use_container_width=True):
                if new_email and new_password and user_id:
                    if new_password != confirm_password:
                        st.error("Passwords don't match")
                    elif len(new_password) < 6:
                        st.error("Password must be at least 6 characters")
                    else:
                        result = self.register(new_email, new_password, user_id)
                        if result.authenticated:
                            st.success("✅ Account created successfully!")
                            return result
                        else:
                            st.error(f"❌ {result.error}")
                else:
                    st.error("Please fill in all fields")
        
        return AuthResult(authenticated=False)
