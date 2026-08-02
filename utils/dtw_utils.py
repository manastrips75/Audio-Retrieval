import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import cosine


# -------------------------------------------------
# DTW Alignment
# -------------------------------------------------

def dtw_distance(
    query_frames,
    candidate_frames,
):
    """
    Performs Dynamic Time Warping between query
    and candidate frame embeddings.

    Returns
    -------
    distance : float

    start_frame : int

    end_frame : int

    path : list
    """

    distance, path = fastdtw(
        query_frames,
        candidate_frames,
        dist=cosine,
    )

    candidate_indices = [
        p[1]
        for p in path
    ]

    start_frame = int(
        min(candidate_indices)
    )

    end_frame = int(
        max(candidate_indices)
    )

    return (
        float(distance),
        start_frame,
        end_frame,
        path,
    )


# -------------------------------------------------
# Re-rank Multiple Candidates
# -------------------------------------------------

def rerank(
    query_frames,
    candidate_list,
):
    """
    candidate_list:

    [
        (metadata, frame_features),
        ...
    ]

    Returns
    -------
    Sorted by DTW distance.
    """

    results = []

    for metadata, frames in candidate_list:

        (
            distance,
            start_frame,
            end_frame,
            path,
        ) = dtw_distance(
            query_frames,
            frames,
        )

        results.append({

            "distance": distance,

            "metadata": metadata,

            "start_frame": start_frame,

            "end_frame": end_frame,

            "path": path,

        })

    results.sort(
        key=lambda x: x["distance"]
    )

    return results


# -------------------------------------------------
# Frame → Seconds
# -------------------------------------------------

def frame_to_seconds(
    frame_index,
    total_frames,
    duration_seconds,
):
    """
    Converts frame index into seconds.
    """

    if total_frames == 0:
        return 0.0

    return (
        frame_index
        * duration_seconds
        / total_frames
    )