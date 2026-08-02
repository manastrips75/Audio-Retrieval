import time

import numpy as np
import torch

from transformers import (
    AutoFeatureExtractor,
    WavLMModel,
)

MODEL_NAME = "microsoft/wavlm-base-plus"

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {device}")

feature_extractor = AutoFeatureExtractor.from_pretrained(
    MODEL_NAME
)

model = WavLMModel.from_pretrained(
    MODEL_NAME
)

model.to(device)

model.eval()


# ==========================================================
# Single Embedding
# ==========================================================

@torch.no_grad()
def get_embedding(
    audio,
    sample_rate=16000,
):

    inputs = feature_extractor(

        audio,

        sampling_rate=sample_rate,

        return_tensors="pt",

    )

    inputs = {

        k: v.to(device)

        for k, v in inputs.items()

    }

    outputs = model(**inputs)

    frame_features = outputs.last_hidden_state.squeeze(0)

    pooled_embedding = frame_features.mean(dim=0)

    pooled_embedding = pooled_embedding.cpu().numpy().astype(np.float32)

    frame_features = frame_features.cpu().numpy().astype(np.float32)

    return pooled_embedding, frame_features


# ==========================================================
# Generate Embeddings
# ==========================================================

def generate_embeddings(
    chunks,
    tracker=None,
):

    embeddings = []

    frame_feature_list = []

    total = len(chunks)

    start_time = time.time()

    print()

    print("=" * 70)

    print("Generating Embeddings")

    print("=" * 70)

    for i, chunk in enumerate(chunks):

        emb, frames = get_embedding(chunk)

        embeddings.append(
            emb
        )

        frame_feature_list.append(
            frames
        )

        if tracker is not None:

            tracker.update()

        else:

            elapsed = time.time() - start_time

            speed = (
                (i + 1) / elapsed
                if elapsed > 0
                else 0
            )

            remaining = total - (i + 1)

            eta = (
                remaining / speed
                if speed > 0
                else 0
            )

            mins = int(eta // 60)

            secs = int(eta % 60)

            progress = (
                ((i + 1) / total)
                * 100
            )

            print(

                f"\r"

                f"{i+1}/{total} "

                f"| {progress:.1f}% "

                f"| {speed:.2f} chunks/sec "

                f"| ETA {mins:02d}:{secs:02d}",

                end="",

                flush=True,

            )

    print()

    print("=" * 70)

    print("Embedding Generation Finished")

    print("=" * 70)

    embeddings = np.vstack(
        embeddings
    )

    return embeddings, frame_feature_list