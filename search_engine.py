from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    TOP_K,
    SAMPLE_RATE,
    MIN_QUERY_DURATION,
    MAX_QUERY_DURATION,
)

from utils.audio import extract_audio
from utils.embedding import get_embedding

from utils.faiss_utils import (
    load_index,
    search,
)

from utils.window_utils import (
    merge_candidates,
)

from utils.window_search import (
    find_best_window,
)

from utils.dtw_utils import (
    dtw_distance,
)

from utils.alignment import (
    refine_alignment,
)

INDEX_DIR = Path("index")


def seconds_to_time(seconds):

    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60

    return f"{h:02d}:{m:02d}:{s:05.2f}"


class AudioSearchEngine:

    def __init__(self):

        print("Loading Index...")

        self.index = load_index(
            INDEX_DIR / "faiss.index"
        )

        self.metadata = pd.read_csv(
            INDEX_DIR / "metadata.csv"
        )

        print("Search Engine Ready!")

    def _load_region_audio(
        self,
        rows,
    ):

        audios = []

        for row in rows:

            audios.append(
                np.load(
                    row["chunk_path"]
                )
            )

        if len(audios) == 1:
            return audios[0]

        region = audios[0]

        overlap = int(
            (3 - 1) * SAMPLE_RATE
        )

        for audio in audios[1:]:

            region = np.concatenate(
                (
                    region,
                    audio[overlap:]
                )
            )

        return region

    def _load_region_frames(
        self,
        rows,
    ):
        """
        Load cached frame features.
        """

        frames = []

        for row in rows:

            frames.append(
                np.load(
                    row["frame_path"]
                )
            )

        return np.vstack(frames)

    def search(
        self,
        query_path,
        top_k=TOP_K,
    ):

        query_audio, sr = extract_audio(
            query_path
        )

        query_duration = (
            len(query_audio)
            / sr
        )

        if query_duration < MIN_QUERY_DURATION:

            raise ValueError(
                f"Query must be at least {MIN_QUERY_DURATION} sec"
            )

        if query_duration > MAX_QUERY_DURATION:

            raise ValueError(
                f"Query must be below {MAX_QUERY_DURATION} sec"
            )

        query_embedding, query_frames = get_embedding(
            query_audio,
            sr,
        )

        scores, ids = search(
            self.index,
            query_embedding,
            top_k,
        )

        candidate_rows = []

        for idx in ids:

            candidate_rows.append(
                self.metadata.iloc[idx]
            )

        merged_regions = merge_candidates(
            candidate_rows
        )

        best_result = None

        for region in merged_regions:

            region_audio = self._load_region_audio(
                region["rows"]
            )

            region_frames = self._load_region_frames(
                region["rows"]
            )

            best_window = find_best_window(
                region_audio,
                sr,
                query_audio,
                query_embedding,
            )

            candidate_audio = best_window["audio"]

            candidate_offset = best_window["offset"]

            _, candidate_frames = get_embedding(
                candidate_audio,
                sr,
            )
            (
                distance,
                start_frame,
                end_frame,
                _,
            ) = dtw_distance(
                query_frames,
                candidate_frames,
            )

            seconds_per_frame = (
                len(candidate_audio)
                / sr
                / len(candidate_frames)
            )

            dtw_offset = (
                start_frame
                * seconds_per_frame
            )

            refined_offset = refine_alignment(
                query_audio,
                candidate_audio,
                sr,
            )

            estimated_start = (
                region["start_time"]
                + candidate_offset
                + dtw_offset
                + refined_offset
            )

            estimated_end = (
                estimated_start
                + query_duration
            )

            result = {

                "file": region["file"],

                "start": estimated_start,

                "end": estimated_end,

                "start_text": seconds_to_time(
                    estimated_start
                ),

                "end_text": seconds_to_time(
                    estimated_end
                ),

                "similarity": max(
                    float(scores[0]),
                    float(best_window["similarity"]),
                ),

                "dtw_score": float(distance),

            }

            if (
                best_result is None
                or distance
                < best_result["dtw_score"]
            ):

                best_result = result

        return best_result