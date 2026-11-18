"""CPU source-separation worker using the existing, read-only PyTorch runtime."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'separator_packages'))


import numpy as np
import soundfile as sf
import torch
from demucs.pretrained import get_model
from demucs.apply import apply_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.hub.set_dir(str(ROOT / 'models' / 'torch'))
    data, rate = sf.read(args.input, dtype='float32', always_2d=True)
    if rate != 44100 or data.shape[1] != 2:
        raise ValueError('Worker input must be 44.1 kHz stereo.')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    print('Loading Demucs htdemucs on CPU (first run downloads model weights)...', flush=True)
    model = get_model('htdemucs').cpu().eval()
    mix = torch.from_numpy(data.T.copy())
    reference = mix.mean(0)
    mean, scale = reference.mean(), reference.std().clamp_min(1e-6)
    print('Separating vocal/speech stem from background in 4-second windows...', flush=True)
    with torch.inference_mode():
        sources = apply_model(model, ((mix - mean) / scale)[None], device='cpu',
                              shifts=0, split=True, overlap=0.25, segment=4,
                              progress=True, num_workers=0)[0]
        voice = (sources[model.sources.index('vocals')] * scale + mean).T.numpy()
    # Residual preserves every sample of the original mix when the stems are summed.
    background = data - voice
    sf.write(output / 'speaker_voice.wav', voice, rate, subtype='FLOAT')
    sf.write(output / 'background.wav', background, rate, subtype='FLOAT')
    print(f'STEM_SUCCESS samples={len(data)} reconstruction_error={np.max(np.abs(data - voice - background)):.3g}', flush=True)


if __name__ == '__main__':
    main()
