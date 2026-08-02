"""
=====================================================
Audio Search System Configuration
=====================================================
"""

# ============================================
# Audio
# ============================================

SAMPLE_RATE = 16000

# ============================================
# Database Chunking
# ============================================

# Each database chunk is 3 seconds
WINDOW_SECONDS = 3

# Sliding window stride (1 second)
STRIDE_SECONDS = 1

# ============================================
# Search
# ============================================

# Number of FAISS candidates
TOP_K = 10

# ============================================
# Query Limits
# ============================================

MIN_QUERY_DURATION = 3       # seconds
MAX_QUERY_DURATION = 60      # seconds

# ============================================
# Model
# ============================================

MODEL_NAME = "microsoft/wavlm-base-plus"

# ============================================
# Embedding
# ============================================

# L2 normalization
NORMALIZE_EMBEDDINGS = True

# ============================================
# Search Refinement
# ============================================

ENABLE_DTW = True

ENABLE_ALIGNMENT = True

# ============================================
# File Formats
# ============================================

SUPPORTED_EXTENSIONS = [
    ".wav",
    ".mp3",
    ".flac",
    ".m4a",
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
]
# --------------------------------------------
# Window Search
# --------------------------------------------

WINDOW_SEARCH_STRIDE = 0.25

EMBEDDING_BATCH_SIZE = 16