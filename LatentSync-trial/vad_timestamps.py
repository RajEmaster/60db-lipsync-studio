"""Small worker interface so other apps can use the project VAD worker."""
import argparse
import json
from pathlib import Path
from vad import detect_speech

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audio', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = detect_speech(args.audio)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(f"RealtimeVAD: {len(result['speech_intervals'])} speech segments, "
          f"{len(result['pause_intervals'])} confirmed gaps.", flush=True)
