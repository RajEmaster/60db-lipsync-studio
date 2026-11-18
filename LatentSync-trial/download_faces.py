"""Fetch the official InsightFace bundle; retain only detector and 2D landmarks."""
from pathlib import Path
import time
import zipfile
import requests

root = Path(__file__).resolve().parent / 'checkpoints' / 'auxiliary'
root.mkdir(parents=True, exist_ok=True)
archive = root / 'buffalo_l.zip'
if not archive.exists():
    url = 'https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip'
    with requests.get(url, stream=True, timeout=(20, 60)) as response:
        response.raise_for_status()
        done = 0
        last = time.monotonic()
        with archive.open('wb') as output:
            for block in response.iter_content(4 * 1024**2):
                output.write(block)
                done += len(block)
                if time.monotonic() - last > 20:
                    print(f'Face bundle: {done / 2**20:.0f} MiB', flush=True)
                    last = time.monotonic()
target = root / 'models' / 'buffalo_l'
target.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(archive) as bundle:
    for name in bundle.namelist():
        if Path(name).name in {'det_10g.onnx', '2d106det.onnx'}:
            (target / Path(name).name).write_bytes(bundle.read(name))
print('Face detection models ready', flush=True)
