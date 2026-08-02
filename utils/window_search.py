import numpy as np

from config import (
    WINDOW_SEARCH_STRIDE,
)

from utils.embedding import (
    get_embedding,
)


# -------------------------------------------------
# Create Sliding Windows
# -------------------------------------------------

def create_windows(
    audio,
    sample_rate,
    window_duration,
    stride=WINDOW_SEARCH_STRIDE,
):

    window_size = int(
        window_duration * sample_rate
    )

    stride_size = int(
        stride * sample_rate
    )

    windows = []

    offsets = []

    if len(audio) <= window_size:

        windows.append(audio)

        offsets.append(0.0)

        return windows, offsets

    for start in range(

        0,

        len(audio) - window_size + 1,

        stride_size,

    ):

        end = start + window_size

        windows.append(
            audio[start:end]
        )

        offsets.append(
            start / sample_rate
        )

    return windows, offsets


# -------------------------------------------------
# Cosine Similarity
# -------------------------------------------------

def cosine_similarity(

    query_embedding,

    embeddings,

):

    query_embedding = (

        query_embedding

        / np.linalg.norm(query_embedding)

    )

    embeddings = (

        embeddings

        / np.linalg.norm(

            embeddings,

            axis=1,

            keepdims=True,

        )

    )

    return embeddings @ query_embedding


# -------------------------------------------------
# Find Best Window
# -------------------------------------------------

def find_best_window(

    region_audio,

    sample_rate,

    query_audio,

    query_embedding,

):

    query_duration = (

        len(query_audio)

        / sample_rate

    )

    windows, offsets = create_windows(

        region_audio,

        sample_rate,

        query_duration,

        stride=WINDOW_SEARCH_STRIDE,

    )

    embeddings = []

    for window in windows:

        emb, _ = get_embedding(

            window,

            sample_rate,

        )

        embeddings.append(

            emb

        )

    embeddings = np.vstack(

        embeddings

    )

    similarities = cosine_similarity(

        query_embedding,

        embeddings,

    )

    best = int(

        np.argmax(similarities)

    )

    return {

        "audio": windows[best],

        "offset": offsets[best],

        "similarity": float(

            similarities[best]

        ),

    }