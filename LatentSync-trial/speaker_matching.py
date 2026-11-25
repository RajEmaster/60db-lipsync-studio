"""Link section-local diarization labels using CPU WeSpeaker embeddings."""
from collections import Counter, defaultdict
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
# Voice-based per-turn correction, designed so it can NEVER merge two speakers into one.
# A turn is proposed for a move only when another speaker is a strong match AND clearly beats
# the turn's own speaker (leave-one-out), which can spot a mislabel even between similar
# voices. The merge-proofing is the CAP below: at most MAX_MOVE_FRACTION of any speaker's
# turns may move and it always keeps at least one, so a whole speaker is never collapsed.
REFINE_TARGET = 0.58      # the other speaker must be a strong match
REFINE_MARGIN = 0.12      # and clearly beat the turn's own speaker by this cosine margin
MAX_MOVE_FRACTION = 0.34  # never move more than this share of a speaker's turns (anti-merge)


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


def _mean_embedding(vectors):
    vectors = [v for v in vectors if v is not None]
    if not vectors:
        return None
    summed = np.sum(vectors, axis=0)
    norm = np.linalg.norm(summed)
    return summed / norm if norm > 0 else None


def refine_turns_by_voice(original, turns, rate=16000, margin=REFINE_MARGIN):
    """Fix turns 60db clearly mislabeled, using voice embeddings, WITHOUT merging speakers.

    60db usually separates speakers correctly but mislabels the odd segment, which gives a
    wrong turn boundary and the wrong cloned voice. A turn is moved only when it is a poor
    match to the REST of its own speaker (a genuine outlier, measured leave-one-out) AND
    strongly matches another speaker — and no more than a small fraction of any speaker's
    turns may move, always leaving it at least one. So two distinct speakers (even similar
    ones) each keep matching their own group and are never collapsed into one. `turns` is a
    list of (first, last, speaker, text) in samples. Any failure returns the diarized turns
    unchanged so a job never breaks.
    """
    labels = [turn[2] for turn in turns]
    counts = Counter(labels)
    if len(counts) < 2 or len(turns) < 4:
        return turns   # single speaker or too little to disambiguate safely
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

    by_label = defaultdict(list)
    for index, turn in enumerate(turns):
        if embeddings[index] is not None:
            by_label[turn[2]].append(index)
    profiles = {label: _mean_embedding([embeddings[i] for i in idxs])
                for label, idxs in by_label.items()}

    # Propose moving only clear outliers that strongly fit another speaker.
    proposals = {}   # index -> (new_label, confidence)
    for index, (first, last, speaker, text) in enumerate(turns):
        embedding = embeddings[index]
        if embedding is None:
            continue
        own_profile = _mean_embedding([embeddings[i] for i in by_label.get(speaker, []) if i != index])
        if own_profile is None:
            continue   # the only turn of its speaker -> never strand/move it
        own_score = float(embedding @ own_profile)   # leave-one-out: fit to the REST of its speaker
        others = [(float(embedding @ profile), label)
                  for label, profile in profiles.items()
                  if label != speaker and profile is not None]
        if not others:
            continue
        best_score, best_label = max(others)
        # Another speaker is a strong match and clearly beats this turn's own speaker. The
        # per-speaker cap below is what guarantees this can never collapse a whole speaker.
        if best_score >= REFINE_TARGET and best_score - own_score >= margin:
            proposals[index] = (best_label, best_score - own_score)

    # Anti-merge cap: move at most MAX_MOVE_FRACTION of a speaker's turns, always leaving it
    # at least one. Apply the most confident moves first. This cannot reduce the speaker count.
    refined_labels = list(labels)
    moved = 0
    by_source = defaultdict(list)
    for index, (new_label, confidence) in proposals.items():
        by_source[labels[index]].append((confidence, index, new_label))
    for source, items in by_source.items():
        allowed = min(int(counts[source] * MAX_MOVE_FRACTION), counts[source] - 1)
        for _confidence, index, new_label in sorted(items, reverse=True)[:max(0, allowed)]:
            refined_labels[index] = new_label
            moved += 1

    refined = [(first, last, refined_labels[index], text)
               for index, (first, last, _speaker, text) in enumerate(turns)]
    print(f'Voice-based turn refinement: reassigned {moved}/{len(turns)} turns '
          f'(speakers kept: {len(set(refined_labels))}/{len(counts)}).', flush=True)
    return refined
