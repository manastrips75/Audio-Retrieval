from pathlib import Path
import faiss
import numpy as np


# -------------------------------------------------
# Build New Index
# -------------------------------------------------

def build_index(embeddings):
    """
    Creates a FAISS cosine similarity index.
    """

    embeddings = embeddings.astype(np.float32)

    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    return index


# -------------------------------------------------
# Add New Embeddings
# -------------------------------------------------

def add_to_index(index, embeddings):
    """
    Adds new embeddings to an existing index.
    """

    embeddings = embeddings.astype(np.float32)

    faiss.normalize_L2(embeddings)

    index.add(embeddings)

    return index


# -------------------------------------------------
# Save Index
# -------------------------------------------------

def save_index(index, path):

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    faiss.write_index(
        index,
        str(path),
    )


# -------------------------------------------------
# Load Index
# -------------------------------------------------

def load_index(path):

    return faiss.read_index(
        str(path)
    )


# -------------------------------------------------
# Check Index Exists
# -------------------------------------------------

def index_exists(path):

    return Path(path).exists()


# -------------------------------------------------
# Search
# -------------------------------------------------

def search(
    index,
    embedding,
    top_k=10,
):
    """
    Returns
    -------
    scores
    ids
    """

    embedding = embedding.astype(
        np.float32
    )

    embedding = embedding.reshape(
        1,
        -1,
    )

    faiss.normalize_L2(
        embedding
    )

    scores, ids = index.search(
        embedding,
        top_k,
    )

    return (
        scores[0],
        ids[0],
    )


# -------------------------------------------------
# Batch Search (Future Use)
# -------------------------------------------------

def batch_search(
    index,
    embeddings,
    top_k=10,
):
    """
    Search multiple embeddings at once.
    """

    embeddings = embeddings.astype(
        np.float32
    )

    faiss.normalize_L2(
        embeddings
    )

    return index.search(
        embeddings,
        top_k,
    )