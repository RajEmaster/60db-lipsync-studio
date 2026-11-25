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
# Within a section, only move a turn to a different speaker when its voice clearly matches
# that speaker better than its diarized one — conservative, so we fix obvious mislabels
# without disturbing uncertain turns (which could make diarization worse).
REFINE_SCORE = 0.50     # the other speaker must look like a real match
REFINE_MARGIN = 0.10    # and beat the diarized speaker by this cosine margin


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
        return self.embed_audio(audio, rate)

    def embed_audio(self, audio, rate):
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


def refine_turns_by_voice(original, turns, rate=16000):
    """Correct 60db's per-turn speaker labels using voice embeddings.

    60db diarization usually gets the speaker COUNT and rough grouping right but mislabels
    individual segments, which shows up as both wrong turn boundaries (a false speaker change)
    and the wrong cloned voice on that turn. Here we build a voice profile per diarized
    speaker (the mean embedding of their turns) and move a turn to another speaker only when
    its voice clearly matches that speaker better. `turns` is a list of (first, last, speaker,
    text) in samples; returns the same list with corrected speaker labels. Any failure (no
    model, bad audio, too few turns) returns the turns unchanged so a job never breaks.
    """
    labels = [turn[2] for turn in turns]
    if len(set(labels)) < 2 or len(turns) < 3:
        return turns   # single speaker or too little to disambiguate
    try:
        encoder = SpeakerEncoder()
        audio, file_rate = sf.read(original, dtype='int16')
    except Exception as exc:
        print(f'Voice-based turn refinement skipped (unavailable): {exc}', flush=True)
        return turns
    if file_rate != rate or audio.ndim != 1:
        return turns

    embeddings = []
    for first, last, _speaker, _text in turns:
        segment = audio[int(first):int(last)]
        try:
            embeddings.append(encoder.embed_audio(segment, rate))
        except Exception:
            embeddings.append(None)

    def profile_for(label):
        vectors = [embeddings[i] for i, turn in enumerate(turns)
                   if turn[2] == label and embeddings[i] is not None]
        if not vectors:
            return None
        summed = np.sum(vectors, axis=0)
        norm = np.linalg.norm(summed)
        return summed / norm if norm > 0 else None

    profiles = {label: profile_for(label) for label in set(labels)}
    refined, moved = [], 0
    for index, (first, last, speaker, text) in enumerate(turns):
        new_label = speaker
        embedding = embeddings[index]
        own_profile = profiles.get(speaker)
        if embedding is not None and own_profile is not None:
            own_score = float(embedding @ own_profile)
            others = [(float(embedding @ profile), label)
                      for label, profile in profiles.items()
                      if label != speaker and profile is not None]
            if others:
                best_score, best_label = max(others)
                if best_score >= REFINE_SCORE and best_score - own_score >= REFINE_MARGIN:
                    new_label = best_label
                    moved += 1
        refined.append((first, last, new_label, text))
    print(f'Voice-based turn refinement: reassigned {moved}/{len(turns)} turns by voice.', flush=True)
    return refined
