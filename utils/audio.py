from pathlib import Path
import platform
import shutil
import subprocess
import tempfile

import librosa
import soundfile as sf


# ==========================================================
# FFmpeg Detection
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if platform.system() == "Windows":

    LOCAL_FFMPEG = PROJECT_ROOT / "ffmpeg" / "ffmpeg.exe"

    if LOCAL_FFMPEG.exists():
        FFMPEG = str(LOCAL_FFMPEG)
    else:
        FFMPEG = shutil.which("ffmpeg")

else:
    FFMPEG = shutil.which("ffmpeg")


if FFMPEG is None:
    raise RuntimeError(
        "\nFFmpeg not found.\n"
        "Install FFmpeg and ensure it is available in your PATH."
    )


# ==========================================================
# Audio Extraction
# ==========================================================

def extract_audio(
    input_file,
    sample_rate=16000,
):
    """
    Extract audio from audio/video file.

    Parameters
    ----------
    input_file : str

    sample_rate : int

    Returns
    -------
    audio : np.ndarray

    sample_rate : int
    """

    input_file = Path(input_file)

    extension = input_file.suffix.lower()

    # ------------------------------------------------------
    # Direct WAV Loading
    # ------------------------------------------------------

    if extension == ".wav":

        audio, sr = librosa.load(
            input_file,
            sr=sample_rate,
            mono=True,
        )

        return audio.astype("float32"), sr

    # ------------------------------------------------------
    # Convert Everything Else using FFmpeg
    # ------------------------------------------------------

    with tempfile.NamedTemporaryFile(
        suffix=".wav",
        delete=False,
    ) as temp_file:

        temp_wav = temp_file.name

    command = [

        FFMPEG,

        "-y",

        "-i",
        str(input_file),

        "-vn",

        "-ac",
        "1",

        "-ar",
        str(sample_rate),

        temp_wav,

    ]

    subprocess.run(

        command,

        stdout=subprocess.DEVNULL,

        stderr=subprocess.DEVNULL,

        check=True,

    )

    audio, sr = sf.read(temp_wav)

    Path(temp_wav).unlink(missing_ok=True)

    return audio.astype("float32"), sr