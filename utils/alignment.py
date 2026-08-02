import numpy as np
from scipy.signal import correlate


# -------------------------------------------------
# Cross-Correlation Alignment
# -------------------------------------------------

def refine_alignment(
    query_audio,
    candidate_audio,
    sample_rate,
):
    """
    Refines the alignment between query audio and
    candidate audio using cross-correlation.

    Parameters
    ----------
    query_audio : ndarray

    candidate_audio : ndarray

    sample_rate : int

    Returns
    -------
    offset_seconds : float
        Best alignment offset inside candidate audio.
    """

    # Remove DC offset
    query = query_audio - np.mean(query_audio)
    candidate = candidate_audio - np.mean(candidate_audio)

    # Normalize
    query = query / (np.linalg.norm(query) + 1e-8)
    candidate = candidate / (np.linalg.norm(candidate) + 1e-8)

    # Cross correlation
    correlation = correlate(
        candidate,
        query,
        mode="valid",
    )

    best_index = np.argmax(correlation)

    offset_seconds = best_index / sample_rate

    return offset_seconds


# -------------------------------------------------
# Optional Score
# -------------------------------------------------

def alignment_score(
    query_audio,
    candidate_audio,
):
    """
    Returns normalized correlation score.
    """

    query = query_audio - np.mean(query_audio)
    candidate = candidate_audio - np.mean(candidate_audio)

    query /= (np.linalg.norm(query) + 1e-8)
    candidate /= (np.linalg.norm(candidate) + 1e-8)

    corr = correlate(
        candidate,
        query,
        mode="valid",
    )

    return float(np.max(corr))