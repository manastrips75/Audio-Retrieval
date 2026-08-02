from pathlib import Path
import numpy as np

from config import (
    SAMPLE_RATE,
    WINDOW_SECONDS,
    STRIDE_SECONDS,
)


def create_chunks(
    audio,
    sample_rate=SAMPLE_RATE,
    chunk_seconds=WINDOW_SECONDS,
    stride_seconds=STRIDE_SECONDS,
    save_dir=None,
):
    """
    Creates overlapping sliding-window chunks.

    Returns
    -------
    chunks : list[np.ndarray]

    metadata : list[dict]
    """

    chunk_size = int(chunk_seconds * sample_rate)
    stride_size = int(stride_seconds * sample_rate)

    chunks = []
    metadata = []

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    chunk_id = 0

    # -------------------------------------------------
    # Short audio
    # -------------------------------------------------

    if len(audio) <= chunk_size:

        chunk = audio.astype(np.float32)

        chunk_path = None

        if save_dir is not None:

            chunk_path = save_dir / f"chunk_{chunk_id:06d}.npy"

            np.save(chunk_path, chunk)

        chunks.append(chunk)

        metadata.append({

            "chunk_id": chunk_id,

            "start_sample": 0,

            "end_sample": len(audio),

            "start_time": 0.0,

            "end_time": len(audio) / sample_rate,

            "duration": len(audio) / sample_rate,

            "chunk_path": str(chunk_path) if chunk_path else None,

        })

        return chunks, metadata

    # -------------------------------------------------
    # Sliding windows
    # -------------------------------------------------

    for start in range(
        0,
        len(audio) - chunk_size + 1,
        stride_size,
    ):

        end = start + chunk_size

        chunk = audio[start:end].astype(np.float32)

        chunk_path = None

        if save_dir is not None:

            chunk_path = save_dir / f"chunk_{chunk_id:06d}.npy"

            np.save(chunk_path, chunk)

        chunks.append(chunk)

        metadata.append({

            "chunk_id": chunk_id,

            "start_sample": start,

            "end_sample": end,

            "start_time": start / sample_rate,

            "end_time": end / sample_rate,

            "duration": chunk_seconds,

            "chunk_path": str(chunk_path) if chunk_path else None,

        })

        chunk_id += 1

    return chunks, metadata