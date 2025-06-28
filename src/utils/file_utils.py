# src/utils/file_utils.py
import hashlib
import mimetypes
from pathlib import Path
from typing import Tuple, Optional
import tempfile
import os

def get_file_hash(file_path: Path) -> str:
    """Generate MD5 hash of a file."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def get_file_info(file_path: Path) -> Tuple[str, Optional[str], int]:
    """Get file information: extension, MIME type, size."""
    extension = file_path.suffix.lower()
    mime_type, _ = mimetypes.guess_type(file_path)
    size = file_path.stat().st_size
    return extension, mime_type, size

def create_temp_file(content: bytes, suffix: str = "") -> Path:
    """Create a temporary file with given content."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(content)
        return Path(tmp_file.name)

def cleanup_temp_files(file_paths: List[Path]):
    """Clean up temporary files."""
    for file_path in file_paths:
        try:
            if file_path.exists():
                os.unlink(file_path)
        except Exception:
            pass  # Ignore cleanup errors

def validate_file_extension(file_path: Path, allowed_extensions: List[str]) -> bool:
    """Validate if file extension is allowed."""
    return file_path.suffix.lower() in allowed_extensions

def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format."""
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"

