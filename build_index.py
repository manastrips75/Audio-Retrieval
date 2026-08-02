from pathlib import Path
import time

import numpy as np
import pandas as pd

from config import SUPPORTED_EXTENSIONS

from utils.audio import extract_audio
from utils.chunking import create_chunks
from utils.embedding import generate_embeddings

from utils.faiss_utils import (
    build_index,
    add_to_index,
    save_index,
    load_index,
    index_exists,
)


# ==========================================================
# Directories
# ==========================================================

DATA_DIR = Path("data")

INDEX_DIR = Path("index")

CHUNK_DIR = INDEX_DIR / "chunks"

FRAME_DIR = INDEX_DIR / "frame_features"

INDEX_DIR.mkdir(exist_ok=True)

CHUNK_DIR.mkdir(exist_ok=True)

FRAME_DIR.mkdir(exist_ok=True)


metadata_path = INDEX_DIR / "metadata.csv"

embedding_path = INDEX_DIR / "embeddings.npy"

index_path = INDEX_DIR / "faiss.index"


# ==========================================================
# Progress Manager
# ==========================================================

class ProgressTracker:

    def __init__(self):

        self.total_chunks = 0

        self.current_chunk = 0

        self.start_time = None

    def start(self, total_chunks):

        self.total_chunks = total_chunks

        self.current_chunk = 0

        self.start_time = time.time()

    def update(self):

        self.current_chunk += 1

        elapsed = time.time() - self.start_time

        speed = (
            self.current_chunk / elapsed
            if elapsed > 0 else 0
        )

        remaining = (
            self.total_chunks
            - self.current_chunk
        )

        eta = (
            remaining / speed
            if speed > 0 else 0
        )

        progress = (
            self.current_chunk
            / self.total_chunks
        ) * 100

        mins = int(eta // 60)

        secs = int(eta % 60)

        print(
            f"\rEmbedding "
            f"{self.current_chunk}/{self.total_chunks} | "
            f"{progress:.1f}% | "
            f"{speed:.2f} chunks/s | "
            f"ETA {mins:02d}:{secs:02d}",
            end="",
            flush=True,
        )


tracker = ProgressTracker()


# ==========================================================
# Find Media Files
# ==========================================================

media_files = []

for ext in SUPPORTED_EXTENSIONS:

    media_files.extend(

        DATA_DIR.glob(f"*{ext}")

    )

print()

print(f"Found {len(media_files)} media files.")

if len(media_files) == 0:

    raise RuntimeError(

        "No media files found inside data/"

    )


# ==========================================================
# Load Existing Index
# ==========================================================

if metadata_path.exists():

    metadata_df = pd.read_csv(

        metadata_path

    )

    indexed_files = set(

        metadata_df["file"].unique()

    )

    all_metadata = metadata_df.to_dict(

        "records"

    )

    global_chunk_id = (

        int(metadata_df["chunk_id"].max()) + 1

    )

    old_embeddings = np.load(

        embedding_path

    )

else:

    indexed_files = set()

    all_metadata = []

    global_chunk_id = 0

    old_embeddings = None


new_embeddings = []


# ==========================================================
# Process Files
# ==========================================================

for media_file in media_files:

    if media_file.name in indexed_files:

        print(

            f"Skipping : {media_file.name}"

        )

        continue

    print()

    print("=" * 70)

    print(f"Processing : {media_file.name}")

    print("=" * 70)

    audio, sr = extract_audio(

        str(media_file)

    )

    chunks, metadata = create_chunks(

        audio,

        sample_rate=sr,

        save_dir=CHUNK_DIR,

    )

    tracker.start(

        len(chunks)

    )

    embeddings, frame_features = generate_embeddings(

        chunks,

        tracker,

    )

    print()

    new_embeddings.append(

        embeddings

    )
        # ==========================================================
    # Save Chunk Metadata
    # ==========================================================

    for i, row in enumerate(metadata):

        chunk_name = (
            f"chunk_{global_chunk_id:06d}.npy"
        )

        frame_name = (
            f"frame_{global_chunk_id:06d}.npy"
        )

        chunk_path = CHUNK_DIR / chunk_name

        frame_path = FRAME_DIR / frame_name

        old_chunk = Path(
            row["chunk_path"]
        )

        if old_chunk.exists():

            old_chunk.rename(
                chunk_path
            )

        np.save(
            frame_path,
            frame_features[i],
        )

        row["chunk_id"] = global_chunk_id

        row["chunk_path"] = str(
            chunk_path
        )

        row["frame_path"] = str(
            frame_path
        )

        row["file"] = media_file.name

        all_metadata.append(
            row
        )

        global_chunk_id += 1


# ==========================================================
# Nothing New
# ==========================================================

if len(new_embeddings) == 0:

    print()

    print("No new media found.")

    exit()


# ==========================================================
# Merge Embeddings
# ==========================================================

new_embeddings = np.vstack(
    new_embeddings
)

if old_embeddings is None:

    all_embeddings = new_embeddings

else:

    all_embeddings = np.vstack(

        [

            old_embeddings,

            new_embeddings,

        ]

    )


# ==========================================================
# Save Embeddings
# ==========================================================

np.save(

    embedding_path,

    all_embeddings,

)

print()

print("Embeddings Saved.")


# ==========================================================
# Save Metadata
# ==========================================================

metadata_df = pd.DataFrame(

    all_metadata

)

metadata_df.to_csv(

    metadata_path,

    index=False,

)

print("Metadata Saved.")


# ==========================================================
# Update FAISS
# ==========================================================

if index_exists(index_path):

    print()

    print("Updating Existing FAISS Index...")

    index = load_index(

        index_path

    )

    add_to_index(

        index,

        new_embeddings,

    )

else:

    print()

    print("Creating New FAISS Index...")

    index = build_index(

        new_embeddings

    )

save_index(

    index,

    index_path,

)

print("FAISS Index Saved.")

print()

print("=" * 70)

print("Indexing Completed Successfully")

print(f"Total Chunks : {index.ntotal}")

print("=" * 70)