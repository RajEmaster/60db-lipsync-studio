"""Link section-local diarization labels using CPU WeSpeaker embeddings."""
from collections import defaultdict
import hashlib
from pathlib import Path

import numpy as np
import soundfile as sf

from download_trial import download


ROOT = Path(__file__).resolve().parent
MODEL_FILE = 'voxceleb_resnet34_LM.onnx'
MODEL_PATH = Path('checkpoints') / 'speaker_matching' / MODEL_FILE
MODEL_REVISION = 'f0c48c298fd835726c27956a5d617bad7115627e'
MODEL_SHA256 = '7bb2f06e9df17cdf1ef14ee8a15ab08ed28e8d0ef5054ee135741560df2ec068'
MATCH_SCORE = 0.65
MATCH_MARGIN = 0.08


class SpeakerEncoder:
    def __init__(self):
        import kaldi_native_fbank as kaldi
        import onnxruntime as ort

        path = ROOT / MODEL_PATH
        if not path.is_file():
            print('Downloading the local speaker matching model…', flush=True)
            download('Wespeaker/wespeaker-voxceleb-resnet34-LM', MODEL_FILE, MODEL_PATH, MODEL_REVISION)
        if hashlib.sha256(path.read_bytes()).hexdigest() != MODEL_SHA256:
            raise RuntimeError(f'Speaker matching model checksum failed at {path}.')
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])
        self.kaldi = kaldi

    def embed(self, sample):
        audio, rate = sf.read(sample, dtype='int16')
        if rate != 16000 or audio.ndim != 1:
            raise ValueError('Speaker samples must be 16 kHz mono.')
        if len(audio) < rate * 0.8:
            return None
        opts = self.kaldi.FbankOptions()
        opts.frame_opts.samp_freq = rate
        opts.frame_opts.dither = 0
        opts.frame_opts.window_type = 'hamming'
        opts.mel_opts.num_bins = 80
        bank = self.kaldi.OnlineFbank(opts)
        bank.accept_waveform(rate, audio.tolist())
        bank.input_finished()
        features = np.stack([bank.get_frame(index) for index in range(bank.num_frames_ready)])
        features -= features.mean(axis=0)
        embedding = self.session.run(['embs'], {'feats': features[None]})[0].reshape(-1)
        norm = np.linalg.norm(embedding)
        return embedding / norm if np.isfinite(norm) and norm > 0 else None


def match_speakers(job, observations):
    sections = defaultdict(list)
    for item in observations:
        sections[item['turn_label'].split('/', 1)[0]].append(item)
    if len(sections) <= 1:
        return {item['turn_label']: f'Speaker {index}' for index, item in enumerate(observations, 1)}

    encoder = SpeakerEncoder()
    mapping, profiles = {}, {}
    next_speaker = 1
    for section, items in sections.items():
        print(f'Matching voices in {section}…', flush=True)
        embeddings = [encoder.embed(job / item['sample']) for item in items]
        candidates = []
        for index, embedding in enumerate(embeddings):
            if embedding is None:
                continue
            scores = sorted(((float(embedding @ profile), speaker) for speaker, profile in profiles.items()),
                            reverse=True)
            if scores and scores[0][0] >= MATCH_SCORE and (
                len(scores) == 1 or scores[0][0] - scores[1][0] >= MATCH_MARGIN
            ):
                candidates.append((scores[0][0], index, scores[0][1]))

        matches, used = {}, set()
        for _, index, speaker in sorted(candidates, reverse=True):
            if speaker not in used:
                matches[index] = speaker
                used.add(speaker)
        for index, (item, embedding) in enumerate(zip(items, embeddings)):
            speaker = matches.get(index)
            if speaker is None:
                speaker = f'Speaker {next_speaker}'
                next_speaker += 1
            mapping[item['turn_label']] = speaker
            if embedding is not None:
                profile = profiles.get(speaker, embedding) + embedding
                profiles[speaker] = profile / np.linalg.norm(profile)
    return mapping
