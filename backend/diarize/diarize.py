import sys
import json
import tempfile
import os
from fastapi import FastAPI, UploadFile, File
from pyannote.audio import Pipeline
import uvicorn

# Initialize FastAPI
app = FastAPI()

# Load the diarization pipeline once
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")

# --- FastAPI endpoint ---
@app.post("/diarize")
async def diarize(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    diarization = pipeline(tmp_path)
    results = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        results.append({
            "speaker": speaker,
            "start": float(turn.start),
            "end": float(turn.end)
        })
    return {"segments": results}


# --- CLI mode ---
def diarize_file(audio_path: str):
    try:
        diarization = pipeline(audio_path)
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                "start": float(turn.start),
                "end": float(turn.end),
                "speaker": speaker
            })
        print(json.dumps(segments))
    except Exception as e:
        print(json.dumps({"error": f"diarization failed: {str(e)}"}))
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Run in CLI mode
        diarize_file(sys.argv[1])
    else:
        # Run as FastAPI service
        port = int(os.environ.get("PORT", 8000))  # 👈 use Render's assigned port
        uvicorn.run(app, host="0.0.0.0", port=port)

