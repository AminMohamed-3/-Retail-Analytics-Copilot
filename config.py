"""Configuration for DSPy and database connections."""
import dspy
import sqlite3
import os

DB_PATH = os.path.join("data", "northwind.sqlite")

# Student model (for inference)
STUDENT_MODEL = "phi3.5:3.8b-mini-instruct-q4_K_M"

# Teacher model (for optimization - larger model gives better guidance)
TEACHER_MODEL = "qwen3:4b"

_current_lm = None
_teacher_lm = None


def setup_dspy(use_cache: bool = True):
    """Configure DSPy with Ollama student model (Phi-3.5).
    
    Args:
        use_cache: Whether to use DSPy's caching (default True)
    """
    global _current_lm
    os.environ["OLLAMA_API_BASE"] = "http://localhost:11434"
    
    # Disable cache if requested (for benchmarking)
    if not use_cache:
        os.environ["DSP_CACHEBOOL"] = "false"
        os.environ["LITELLM_CACHE"] = "false"
        # Clear existing caches
        clear_cache(silent=True)
    
    try:
        phi3 = dspy.LM(
            model=f"ollama/{STUDENT_MODEL}",
            api_base="http://localhost:11434",
            model_type="chat",
            max_tokens=1000,
            temperature=0.0,
            keep_alive=0,  # Prevent hallucinations from context caching
        )
    except Exception:
        phi3 = dspy.LM(
            model=f"ollama/{STUDENT_MODEL}",
            max_tokens=1000,
            temperature=0.0,
            keep_alive=0,
        )
    
    dspy.settings.configure(lm=phi3)
    _current_lm = phi3
    return phi3


def get_teacher_lm():
    """Get the teacher language model (Qwen3 4B) for optimization.
    
    Teacher-student optimization uses a larger/better model to guide
    the optimization process, while the smaller student model is used
    for actual inference.
    """
    global _teacher_lm
    if _teacher_lm is None:
        try:
            _teacher_lm = dspy.LM(
                model=f"ollama/{TEACHER_MODEL}",
                api_base="http://localhost:11434",
                model_type="chat",
                max_tokens=2000,
                temperature=0.7,  # Higher temperature for more diverse proposals
                keep_alive=0,  # Prevent context caching issues
            )
        except Exception:
            _teacher_lm = dspy.LM(
                model=f"ollama/{TEACHER_MODEL}",
                max_tokens=2000,
                temperature=0.7,
                keep_alive=0,
            )
    return _teacher_lm


def clear_cache(silent: bool = False):
    """Clear ALL DSPy and LiteLLM caches to ensure fresh LLM calls."""
    import shutil
    from pathlib import Path
    
    cache_paths = [
        Path.home() / ".dspy_cache",
        Path.home() / ".litellm_cache", 
        Path.home() / ".cache" / "litellm",
        Path("/teamspace/studios/this_studio/.dspy_cache"),
        Path(".dspy_cache"),
    ]
    
    cleared = False
    for cache_path in cache_paths:
        if cache_path.exists():
            try:
                shutil.rmtree(cache_path)
                if not silent:
                    print(f"  Cleared cache: {cache_path}")
                cleared = True
            except Exception:
                pass
    
    # Force disable caching via environment
    os.environ["DSP_CACHEBOOL"] = "false"
    os.environ["LITELLM_CACHE"] = "false"
    
    if not cleared and not silent:
        print("  No cache found to clear")
    
    return cleared


def get_lm():
    """Get the current (student) language model instance."""
    global _current_lm
    if _current_lm is None:
        setup_dspy()
    return _current_lm


def get_db_connection():
    """Returns a connection to the SQLite database."""
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"Database not found at {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


if __name__ == "__main__":
    print(f"Testing student model ({STUDENT_MODEL})...")
    try:
        lm = setup_dspy()
        response = lm("Say 'System Ready' if you can read this.")
        print(f"Student Response: {str(response)[:100]}...")
        print("✅ Student model ready.")
    except Exception as e:
        print(f"❌ Student model error: {e}")
    
    print(f"\nTesting teacher model ({TEACHER_MODEL})...")
    try:
        teacher = get_teacher_lm()
        response = teacher("Say 'Teacher Ready' if you can read this.")
        print(f"Teacher Response: {str(response)[:100]}...")
        print("✅ Teacher model ready.")
    except Exception as e:
        print(f"❌ Teacher model error: {e}")
    
    print("\nTesting database connection...")
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='view'")
        views = [row[0] for row in cursor.fetchall()]
        print(f"Database Views Found: {views[:5]}...")
        print("✅ Database ready.")
    except Exception as e:
        print(f"❌ Database error: {e}")
    
    print("\n✅ Environment is GO.")
