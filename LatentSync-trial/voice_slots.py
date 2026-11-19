"""Free one slot for a new clone without deleting a user's own voices."""
from collections import defaultdict
import json
from pathlib import Path
import time


JOBS = Path(__file__).resolve().parent / 'local_jobs'
RECENT_USE_SECONDS = 120


def recycle_voice_slot(client, workspace):
    workspaces = client.get_workspaces()['data']
    if len(workspaces) != 1 or str(workspaces[0]['id']) != str(workspace):
        raise RuntimeError('Automatic voice cleanup requires an unambiguous 60db workspace.')

    protected = set()
    for job in (JOBS / 'long').iterdir() if (JOBS / 'long').exists() else ():
        if not job.is_dir() or ((job / 'completed.json').is_file() and (job / 'output.mp4').is_file()):
            continue
        voices = job / 'voices.json'
        if voices.is_file():
            protected.update(json.loads(voices.read_text(encoding='utf-8')).values())
        config = job / 'job.json'
        if config.is_file():
            voice = json.loads(config.read_text(encoding='utf-8')).get('voice_id')
            if voice:
                protected.add(voice)

    records = defaultdict(list)
    for path in (JOBS / 'voice_clones').glob('*.json'):
        result = json.loads(path.read_text(encoding='utf-8'))
        identity = result.get('id') or result.get('voice_id')
        if identity:
            records[str(identity)].append(path)

    candidates = []
    for voice in client.get_my_voices()['data']:
        identity = voice.get('voice_id')
        name = voice.get('name') or ''
        if (identity not in records or not name.startswith('Lipsyncstudio ')
                or identity in protected or voice.get('voice_provider_id') in protected
                or voice.get('is_public') or voice.get('status') in ('processing', 'pending', 'training')):
            continue
        last_used = max(path.stat().st_mtime for path in records[identity])
        if time.time() - last_used >= RECENT_USE_SECONDS:
            candidates.append((last_used, identity, name))
    if not candidates:
        raise RuntimeError('The voice limit is full and no inactive Lipsyncstudio clone can be recycled.')

    _, identity, name = min(candidates)
    client.delete_voice(identity)
    if any(voice.get('voice_id') == identity for voice in client.get_my_voices()['data']):
        raise RuntimeError('60db has not confirmed removal of the old clone. No other voices were deleted.')
    for path in records[identity]:
        path.unlink(missing_ok=True)
    print(f'Freed one 60db slot by removing an inactive app clone ({name}).', flush=True)
