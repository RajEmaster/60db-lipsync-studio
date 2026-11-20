"""Reclaim slots used by inactive app-generated voice clones."""
from collections import defaultdict
import json
from pathlib import Path
import time

from locking_compat import try_lock, unlock

JOBS = Path(__file__).resolve().parent / 'local_jobs'
RECENT_USE_SECONDS = 120


def job_is_running(job):
    lock = job / 'job.lock'
    if not lock.is_file():
        return False
    with lock.open('r+b') as handle:
        if not try_lock(handle):
            return True
        unlock(handle)
    return False


def recycle_voice_slots(client, workspace, *, limit=None):
    workspaces = client.get_workspaces()['data']
    if len(workspaces) != 1 or str(workspaces[0]['id']) != str(workspace):
        raise RuntimeError('Automatic voice cleanup requires an unambiguous 60db workspace.')

    protected, released = set(), set()
    for job in (JOBS / 'long').iterdir() if (JOBS / 'long').exists() else ():
        if not job.is_dir():
            continue
        path = job / 'voices.json'
        voices = json.loads(path.read_text(encoding='utf-8')) if path.is_file() else {}
        live = set()
        if job_is_running(job):
            scope = job / 'active_voice.json'
            if scope.is_file():
                active = json.loads(scope.read_text(encoding='utf-8'))
                section = job / active['section']
                if not ((section / 'dub_ready.json').is_file()
                        and (section / 'speech/replacement_speech.wav').is_file()):
                    identity = voices.get(active['speaker'])
                    if identity:
                        live.add(identity)
            else:
                live.update(voices.values())
        protected.update(live)
        released.update(set(voices.values()) - live)
        config = job / 'job.json'
        if config.is_file():
            voice = json.loads(config.read_text(encoding='utf-8')).get('voice_id')
            if voice:
                protected.add(voice)

    records = defaultdict(list)
    for path in (JOBS / 'voice_clones').glob('*.json'):
        result = json.loads(path.read_text(encoding='utf-8'))
        for identity in {result.get('id'), result.get('voice_id'), result.get('voice_provider_id')} - {None, ''}:
            records[str(identity)].append(path)

    candidates = []
    for voice in client.get_my_voices()['data']:
        identity = voice.get('voice_id') or voice.get('id')
        if not identity:
            continue
        identity = str(identity)
        aliases = {identity, str(voice.get('voice_provider_id') or identity)}
        paths = {path for alias in aliases for path in records[alias]}
        name = voice.get('name') or ''
<<<<<<< HEAD
        # Recycle any app-created clone ("Lipsyncstudio ...") that is not currently
        # protected. A missing local record (e.g. on ephemeral/cloud storage where
        # local_jobs is wiped between sessions) means we simply have no recent-use
        # timestamp, so the clone is treated as old and eligible -- never a user voice.
        if (not name.startswith('Lipsyncstudio ')
                or identity in protected or voice.get('voice_provider_id') in protected
                or voice.get('is_public') or voice.get('status') in ('processing', 'pending', 'training')):
            continue
        paths = records.get(identity, [])
        last_used = max((path.stat().st_mtime for path in paths), default=0.0)
        if (voice.get('voice_provider_id') in released or not paths
                or time.time() - last_used >= RECENT_USE_SECONDS):
            candidates.append((last_used, identity, name))
    if not candidates:
        raise RuntimeError('The voice limit is full and no inactive Lipsyncstudio clone can be recycled.')

    _, identity, name = min(candidates)
    client.delete_voice(identity)
    if any(voice.get('voice_id') == identity for voice in client.get_my_voices()['data']):
        raise RuntimeError('60db has not confirmed removal of the old clone. No other voices were deleted.')
    for path in records.get(identity, []):
        path.unlink(missing_ok=True)
    print(f'Freed one 60db slot by removing an inactive app clone ({name}).', flush=True)
=======
        if (not paths or not name.startswith('Lipsyncstudio ') or aliases & protected
                or voice.get('is_public') or voice.get('status') in ('processing', 'pending', 'training')):
            continue
        last_used = max(path.stat().st_mtime for path in paths)
        if aliases & released or time.time() - last_used >= RECENT_USE_SECONDS:
            candidates.append((last_used, identity, name, paths))

    removed = 0
    for _, identity, name, paths in sorted(candidates, key=lambda candidate: candidate[:2])[:limit]:
        client.delete_voice(identity)
        if any(str(voice.get('voice_id') or voice.get('id')) == identity
               for voice in client.get_my_voices()['data']):
            raise RuntimeError(f'60db has not confirmed removal of {name}.')
        for path in paths:
            path.unlink(missing_ok=True)
        removed += 1
        print(f'Freed one 60db slot by removing an inactive app clone ({name}).', flush=True)
    return removed
>>>>>>> b6adc3dd822719bca19368dac0f75cd970ec6494
