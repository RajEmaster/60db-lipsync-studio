"""Project-owned speech timing policy using the installed RealtimeVAD package."""
import numpy as np
import soundfile as sf
from RealtimeVAD import VAD

def detect_speech(path, minimum_pause=0.3):
    """Feed 32 ms blocks to RealtimeVAD; use confirmed ends, never speculative pauses."""
    detector = VAD(end_s=minimum_pause, quiet_end_s=minimum_pause, preroll_s=0.15)
    speech, events = [], []
    with sf.SoundFile(path) as source:
        if source.samplerate != 16000 or source.channels != 1:
            raise ValueError('RealtimeVAD input must be 16 kHz mono.')
        duration = len(source) / 16000
        for block in source.blocks(blocksize=512, dtype='float32'):
            for event in detector.feed(block):
                events.append({'kind': event.kind, 'time': event.time})
                if event.kind == 'end':
                    speech.append([max(0, event.segment.start), min(duration, event.segment.speech_end)])
        # Flush a final unfinished utterance. These artificial samples are never exported.
        for event in detector.feed(np.zeros(16000, dtype=np.float32)):
            if event.kind == 'end':
                speech.append([max(0, event.segment.start), min(duration, event.segment.speech_end)])
    merged = []
    for start, end in speech:
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    quiet, cursor = [], 0.0
    for start, end in merged:
        if start - cursor >= minimum_pause:
            quiet.append([cursor, start])
        cursor = end
    if duration - cursor >= minimum_pause:
        quiet.append([cursor, duration])
    return {'engine': 'RealtimeVAD', 'minimum_pause_seconds': minimum_pause,
            'duration_seconds': duration, 'speech_intervals': merged,
            'pause_intervals': quiet, 'events': events}

