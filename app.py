from pathlib import Path
import threading
import subprocess
import time
import json
import os

from flask import (
    Flask,
    render_template,
    request,
    jsonify,
)

# ==========================================================
# Flask
# ==========================================================

app = Flask(__name__)

# ==========================================================
# Folders
# ==========================================================

DATA_FOLDER = Path("data")

QUERY_FOLDER = Path("query")

INDEX_FOLDER = Path("index")

STATIC_FOLDER = Path("static")

DATA_FOLDER.mkdir(exist_ok=True)

QUERY_FOLDER.mkdir(exist_ok=True)

INDEX_FOLDER.mkdir(exist_ok=True)

STATIC_FOLDER.mkdir(exist_ok=True)

# ==========================================================
# Progress Object
# ==========================================================

progress = {

    "running": False,

    "progress": 0,

    "status": "Ready",

    "file": "-",

    "chunk": 0,

    "total_chunks": 0,

    "eta": "--:--",

    "speed": "0",

}

progress_lock = threading.Lock()

# ==========================================================
# Home
# ==========================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )

# ==========================================================
# Upload Media
# ==========================================================

@app.route(
    "/upload-media",
    methods=["POST"],
)
def upload_media():

    if "media" not in request.files:

        return jsonify({

            "success": False,

            "message": "No media selected.",

        })

    file = request.files["media"]

    if file.filename == "":

        return jsonify({

            "success": False,

            "message": "Invalid filename.",

        })

    save_path = DATA_FOLDER / file.filename

    file.save(save_path)

    return jsonify({

        "success": True,

        "message": f"{file.filename} uploaded successfully."

    })

# ==========================================================
# Upload Query
# ==========================================================

@app.route(
    "/upload-query",
    methods=["POST"],
)
def upload_query():

    if "query" not in request.files:

        return jsonify({

            "success": False,

            "message": "No query selected.",

        })

    file = request.files["query"]

    if file.filename == "":

        return jsonify({

            "success": False,

            "message": "Invalid filename.",

        })

    save_path = QUERY_FOLDER / "query.wav"

    file.save(save_path)

    return jsonify({

        "success": True,

        "message": "Query uploaded."

    })

# ==========================================================
# Update Progress
# ==========================================================

def update_progress(

    progress_value=None,

    status=None,

    current_file=None,

    chunk=None,

    total_chunks=None,

    eta=None,

    speed=None,

):

    with progress_lock:

        if progress_value is not None:

            progress["progress"] = progress_value

        if status is not None:

            progress["status"] = status

        if current_file is not None:

            progress["file"] = current_file

        if chunk is not None:

            progress["chunk"] = chunk

        if total_chunks is not None:

            progress["total_chunks"] = total_chunks

        if eta is not None:

            progress["eta"] = eta

        if speed is not None:

            progress["speed"] = speed
# ==========================================================
# Build Thread
# ==========================================================

def index_worker():

    with progress_lock:

        progress["running"] = True

        progress["progress"] = 0

        progress["status"] = "Starting..."

        progress["file"] = "-"

        progress["chunk"] = 0

        progress["total_chunks"] = 0

        progress["eta"] = "--:--"

        progress["speed"] = "0"

    try:

        process = subprocess.Popen(

            [

                sys.executable,

                "build_index.py",

            ],

            stdout=subprocess.PIPE,

            stderr=subprocess.STDOUT,

            universal_newlines=True,

            bufsize=1,

        )

        while True:

            line = process.stdout.readline()

            if line == "" and process.poll() is not None:

                break

            line = line.strip()

            if line == "":

                continue

            print(line)

            # -----------------------------------------
            # Current File
            # -----------------------------------------

            if line.startswith("Processing :"):

                filename = line.replace(

                    "Processing :",

                    "",

                ).strip()

                update_progress(

                    status="Processing",

                    current_file=filename,

                )

            # -----------------------------------------
            # Progress Line
            #
            # Example:
            #
            # Embedding 145/2000 | 7.2% |
            # 4.30 chunks/s | ETA 08:10
            # -----------------------------------------

            if line.startswith("Embedding"):

                try:

                    left = line.split("|")

                    chunk_text = left[0]

                    percent_text = left[1]

                    speed_text = left[2]

                    eta_text = left[3]

                    numbers = (

                        chunk_text

                        .replace(

                            "Embedding",

                            "",

                        )

                        .strip()

                    )

                    current_chunk = int(

                        numbers.split("/")[0]

                    )

                    total_chunks = int(

                        numbers.split("/")[1]

                    )

                    percent = float(

                        percent_text

                        .replace("%", "")

                        .strip()

                    )

                    speed = (

                        speed_text

                        .replace(

                            "chunks/s",

                            "",

                        )

                        .strip()

                    )

                    eta = (

                        eta_text

                        .replace(

                            "ETA",

                            "",

                        )

                        .strip()

                    )

                    update_progress(

                        progress_value=percent,

                        status="Generating Embeddings",

                        chunk=current_chunk,

                        total_chunks=total_chunks,

                        eta=eta,

                        speed=speed,

                    )

                except Exception:

                    pass

        process.wait()

        update_progress(

            progress_value=100,

            status="Finished",

            eta="00:00",

        )

    except Exception as e:

        update_progress(

            status=f"Error : {str(e)}"

        )

    with progress_lock:

        progress["running"] = False
# ==========================================================
# Build Index Route
# ==========================================================

@app.route(
    "/build-index",
    methods=["POST"],
)
def build_index():

    with progress_lock:

        if progress["running"]:

            return jsonify({

                "success": False,

                "message": "Indexing already running.",

            })

    thread = threading.Thread(

        target=index_worker,

        daemon=True,

    )

    thread.start()

    return jsonify({

        "success": True,

        "message": "Indexing started.",

    })


# ==========================================================
# Progress Route
# ==========================================================

@app.route("/progress")
def get_progress():

    with progress_lock:

        return jsonify(progress)


# ==========================================================
# Search Route
# ==========================================================

@app.route(
    "/search",
    methods=["POST"],
)
def search():

    try:

        if "query_audio" not in request.files:

            return jsonify({

                "success": False,

                "message": "No query uploaded.",

            })

        file = request.files["query_audio"]

        if file.filename == "":

            return jsonify({

                "success": False,

                "message": "Please choose a query file.",

            })

        query_path = QUERY_FOLDER / "query.wav"

        file.save(query_path)

        from search_engine import AudioSearchEngine

        engine = AudioSearchEngine()

        result = engine.search(

            str(query_path)

        )

        try:

            os.remove(query_path)

        except:

            pass

        return jsonify({

            "success": True,

            "file": result["file"],

            "start": result["start_text"],

            "end": result["end_text"],

            "similarity": f"{result['similarity'] * 100:.2f}%",

        })

    except Exception as e:

        return jsonify({

            "success": False,

            "message": str(e),

        })


# ==========================================================
# Run Flask
# ==========================================================

if __name__ == "__main__":

    app.run(

        host="0.0.0.0",

        port=5000,

        debug=True,

        threaded=True,

    )