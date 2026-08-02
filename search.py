# search.py

import json
import time

import numpy as np
import torch
import faiss
import librosa

from transformers import AutoFeatureExtractor, AutoModel



# =====================================
# CONFIG
# =====================================

MODEL_NAME = "microsoft/wavlm-base-plus"

SAMPLE_RATE = 16000


INDEX_FILE = "audio_index.faiss"

METADATA_FILE = "metadata.json"



device = (

    "cuda"

    if torch.cuda.is_available()

    else "cpu"

)





processor = None

model = None

index = None

metadata = None






# =====================================
# Load Model
# =====================================

def load_model():


    global processor
    global model



    if model is None:


        print(
            "Loading WavLM model..."
        )


        processor = AutoFeatureExtractor.from_pretrained(

            MODEL_NAME

        )


        model = AutoModel.from_pretrained(

            MODEL_NAME

        )


        model.to(device)

        model.eval()



        print(

            "Model ready on",

            device

        )








# =====================================
# Reload FAISS Index
# =====================================

def reload_index():


    global index
    global metadata



    print(
        "Reloading FAISS index..."
    )



    index = faiss.read_index(

        INDEX_FILE

    )



    with open(

        METADATA_FILE,

        "r"

    ) as f:


        metadata = json.load(f)





    print(

        "FAISS vectors:",

        index.ntotal

    )


    print(

        "Metadata:",

        len(metadata)

    )



    if index.ntotal != len(metadata):


        raise Exception(

            "Index and metadata mismatch"

        )






# =====================================
# Load Everything
# =====================================

def load_resources():


    load_model()



    if index is None:


        reload_index()







# =====================================
# Create Embedding
# =====================================

def generate_embedding(audio):


    load_resources()



    inputs = processor(

        audio,

        sampling_rate=SAMPLE_RATE,

        return_tensors="pt"

    )



    inputs = {


        k:v.to(device)

        for k,v in inputs.items()

    }





    with torch.no_grad():


        output = model(

            **inputs

        )





    embedding = (

        output

        .last_hidden_state

        .mean(dim=1)

    )



    embedding = (

        embedding

        .cpu()

        .numpy()

    )



    embedding = embedding.astype(

        "float32"

    )



    # cosine similarity

    faiss.normalize_L2(

        embedding

    )



    return embedding






# =====================================
# Seconds to MM:SS
# =====================================

def seconds_to_time(seconds):


    minutes = int(

        seconds // 60

    )


    sec = int(

        seconds % 60

    )


    return f"{minutes:02d}:{sec:02d}"








# =====================================
# Search
# =====================================

def search_audio(

        query_path,

        top_k=5

):


    start_time = time.time()



    load_resources()



    print(

        "\nSearching:",

        query_path

    )





    audio, sr = librosa.load(

        query_path,

        sr=SAMPLE_RATE

    )





    duration = len(audio)/SAMPLE_RATE



    print(

        "Query duration:",

        round(duration,2),

        "sec"

    )






    # IMPORTANT:
    # Do not split query.
    # Query is already short.

    embedding_start = time.time()



    query_embedding = generate_embedding(

        audio

    )



    embedding_time = round(

        time.time()-embedding_start,

        4

    )






    # Search more internally

    SEARCH_K = 20



    scores, ids = index.search(

        query_embedding,

        SEARCH_K

    )




    results = []




    for score, idx in zip(

        scores[0],

        ids[0]

    ):



        if idx == -1:

            continue




        item = metadata[idx]



        results.append(

            {

                "file":

                item["file"],


                "start_seconds":

                item["start_time"],


                "end_seconds":

                item["end_time"],



                "timestamp":

                seconds_to_time(

                    item["start_time"]

                )

                +

                " - "

                +

                seconds_to_time(

                    item["end_time"]

                ),



                "similarity":

                round(

                    float(score),

                    4

                ),



                "embedding_time":

                embedding_time

            }

        )





    # Highest similarity first

    results.sort(

        key=lambda x:

        x["similarity"],

        reverse=True

    )





    total_time = round(

        time.time()-start_time,

        3

    )



    print(

        "Search time:",

        total_time,

        "sec"

    )



    if results:


        print(

            "\nBest Match:"

        )

        print(

            results[0]

        )



    return results[:top_k]








# =====================================
# Terminal Test
# =====================================

if __name__ == "__main__":


    import sys



    if len(sys.argv)<2:


        print(

            "Usage: python search.py query.wav"

        )


        exit()



    results = search_audio(

        sys.argv[1]

    )



    print(

        "\nRESULTS"

    )



    for r in results:


        print(r)