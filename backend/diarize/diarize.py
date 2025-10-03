
import sys
import json
try:
    from pyannote.audio import Pipeline
except Exception as e:
    print(json.dumps({'error': 'pyannote not installed or failed to import: ' + str(e)}))
    sys.exit(1)
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(json.dumps({'error': 'usage: diarize.py /path/to/audio.wav'}))
        sys.exit(1)
    audio_path = sys.argv[1]
    try:
        pipeline = Pipeline.from_pretrained('pyannote/speaker-diarization')
        diarization = pipeline(audio_path)
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({'start': float(turn.start), 'end': float(turn.end), 'speaker': speaker})
        print(json.dumps(segments))
    except Exception as e:
        print(json.dumps({'error': 'diarization failed: ' + str(e)}))
        sys.exit(1)
