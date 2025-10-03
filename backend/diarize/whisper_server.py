from flask import Flask, request, jsonify
from faster_whisper import WhisperModel # <-- UPDATED: Import from faster_whisper
import os
import tempfile

app = Flask(__name__)

model_size = "base.en"

print(f"Loading faster-whisper model '{model_size}' (this may take a moment)...")
# --- UPDATED: Use WhisperModel from faster_whisper ---
# This is much more memory efficient.
model = WhisperModel(model_size, device="cpu", compute_type="int8")
print("✅ Whisper model loaded and ready.")

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_filepath = os.path.join(temp_dir, file.filename)
        file.save(temp_filepath)

        try:
            # --- UPDATED: The transcribe function is slightly different ---
            # It returns segments, which we join together to form the full text.
            segments, info = model.transcribe(temp_filepath, beam_size=5)
            
            print(f"Detected language '{info.language}' with probability {info.language_probability}")
            
            full_transcript = "".join(segment.text for segment in segments)
            
            print(f"Successfully transcribed file: {file.filename}")
            return jsonify({"transcript": full_transcript})
        except Exception as e:
            print(f"Error during transcription: {e}")
            return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)