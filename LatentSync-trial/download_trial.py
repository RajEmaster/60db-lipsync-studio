"""Download official trial weights directly, with resumable partial files."""
from pathlib import Path
import time
import requests

ROOT = Path(__file__).resolve().parent

def download(repo, remote, target, revision='main'):
    target = ROOT / target
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        print('Present:', target.name, flush=True)
        return
    partial = target.with_suffix(target.suffix + '.partial')
    for attempt in range(5):
        offset = partial.stat().st_size if partial.exists() else 0
        try:
            with requests.get(
                f'https://huggingface.co/{repo}/resolve/{revision}/{remote}?download=true',
                headers={'Range': f'bytes={offset}-'} if offset else {},
                stream=True, timeout=(20, 60),
            ) as response:
                response.raise_for_status()
                append = offset > 0 and response.status_code == 206
                done = offset if append else 0
                total = done + int(response.headers.get('Content-Length', 0))
                report = time.monotonic()
                with partial.open('ab' if append else 'wb') as output:
                    for chunk in response.iter_content(4 * 1024 * 1024):
                        output.write(chunk)
                        done += len(chunk)
                        if time.monotonic() - report > 20:
                            print(f'{remote}: {done / 2**20:.0f}/{total / 2**20:.0f} MiB', flush=True)
                            report = time.monotonic()
                if total and done != total:
                    raise IOError(f'Incomplete download: {done}/{total}')
                partial.replace(target)
                print('Downloaded:', target.name, flush=True)
                return
        except (requests.RequestException, IOError) as exc:
            print(f'Retry {attempt + 1}: {exc}', flush=True)
    raise RuntimeError(f'Unable to download {remote}')

if __name__ == '__main__':
    download('ByteDance/LatentSync-1.5', 'latentsync_unet.pt', 'checkpoints/latentsync_unet.pt')
    download('ByteDance/LatentSync-1.5', 'whisper/tiny.pt', 'checkpoints/whisper/tiny.pt')
    download('stabilityai/sd-vae-ft-mse', 'config.json', 'checkpoints/vae/config.json')
    download('stabilityai/sd-vae-ft-mse', 'diffusion_pytorch_model.safetensors', 'checkpoints/vae/diffusion_pytorch_model.safetensors')
