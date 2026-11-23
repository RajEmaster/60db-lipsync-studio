"""Client side of the resident render worker: start it, feed clips, collect results.

render_chunks.py uses RenderSession to render every clip of a section through a single
long-lived worker process (model loaded once) instead of a cold subprocess per clip.
The queue lives inside the job folder, so an interrupted section resumes without a reload
penalty beyond the one unavoidable startup load.
"""
import json
import os
from pathlib import Path
import subprocess
import time

from settings import WORKER_PYTHON

ROOT = Path(__file__).resolve().parent


class WorkerError(RuntimeError):
    pass


def detect_gpu_count():
    """How many CUDA devices the worker interpreter can see (respects CUDA_VISIBLE_DEVICES)."""
    try:
        out = subprocess.run(
            [str(WORKER_PYTHON), '-c', 'import torch;print(torch.cuda.device_count())'],
            cwd=ROOT, capture_output=True, text=True, timeout=180)
        return max(0, int((out.stdout or '0').strip() or '0'))
    except (ValueError, OSError, subprocess.SubprocessError):
        return 0


class RenderSession:
    def __init__(self, queue_dir, *, version='1.6', offload='auto', cpu_threads=4,
                 seed=1247, crop_smoothing=0.2,
                 parent_pid=None, log_path=None, startup_timeout=900):
        self.queue = Path(queue_dir)
        self.requests = self.queue / 'requests'
        self.results = self.queue / 'results'
        self.version = version
        self.offload = offload
        self.cpu_threads = cpu_threads
        self.seed = seed
        self.crop_smoothing = crop_smoothing
        self.parent_pid = parent_pid
        self.log_path = Path(log_path) if log_path else self.queue / 'worker.log'
        self.startup_timeout = startup_timeout
        self.process = None
        self._log = None
        self.ready_info = {}

    # ------------------------------------------------------------- lifecycle
    def __enter__(self):
        self.requests.mkdir(parents=True, exist_ok=True)
        self.results.mkdir(parents=True, exist_ok=True)
        # A stale STOP from a previous crashed run would make the new worker exit at once.
        (self.queue / 'STOP').unlink(missing_ok=True)
        ready_file = self.queue / 'ready.json'
        ready_file.unlink(missing_ok=True)
        command = [str(WORKER_PYTHON), '-u', str(ROOT / 'inference_worker.py'),
                   '--queue', str(self.queue), '--version', str(self.version),
                   '--offload', str(self.offload), '--cpu-threads', str(self.cpu_threads),
                   '--seed', str(self.seed), '--crop-smoothing', str(self.crop_smoothing),
                   '--ready-file', str(ready_file)]
        if self.parent_pid:
            command += ['--parent-pid', str(self.parent_pid)]
        self._log = self.log_path.open('w', encoding='utf-8')
        self.process = subprocess.Popen(
            command, cwd=ROOT, stdout=self._log, stderr=subprocess.STDOUT,
            env={**os.environ, 'PYTHONIOENCODING': 'utf-8'},
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        self._await_ready(ready_file)
        return self

    def _await_ready(self, ready_file):
        deadline = time.monotonic() + self.startup_timeout
        while time.monotonic() < deadline:
            if ready_file.is_file():
                try:
                    self.ready_info = json.loads(ready_file.read_text(encoding='utf-8'))
                except (OSError, ValueError):
                    self.ready_info = {}
                return
            if self.process.poll() is not None:
                raise WorkerError(f'Render worker exited during startup (code '
                                  f'{self.process.returncode}).\n{self._log_tail()}')
            time.sleep(0.25)
        raise WorkerError('Render worker did not finish loading the model in time.\n'
                          + self._log_tail())

    def __exit__(self, *exc):
        try:
            (self.queue / 'STOP').write_text('stop', encoding='ascii')
            if self.process is not None:
                try:
                    self.process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    self.process.terminate()
                    try:
                        self.process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        self.process.kill()
        finally:
            if self._log is not None:
                self._log.close()
        return False

    # --------------------------------------------------------------- render
    def render(self, clip_id, video, audio, output, *, frames=16, steps=20,
               guidance_scale=None, poll_timeout=3600):
        result_path = self.results / f'res_{clip_id}.json'
        result_path.unlink(missing_ok=True)
        spec = {'id': clip_id, 'video': str(Path(video).resolve()),
                'audio': str(Path(audio).resolve()), 'output': str(Path(output).resolve()),
                'frames': int(frames), 'steps': int(steps)}
        if guidance_scale is not None:
            spec['guidance_scale'] = float(guidance_scale)
        request = self.requests / f'req_{clip_id}.json'
        tmp = request.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(spec, ensure_ascii=False), encoding='utf-8')
        tmp.replace(request)

        deadline = time.monotonic() + poll_timeout
        while time.monotonic() < deadline:
            if result_path.is_file():
                result = json.loads(result_path.read_text(encoding='utf-8'))
                result_path.unlink(missing_ok=True)
                if result.get('status') == 'ok':
                    return result['stats']
                raise WorkerError(f'Clip {clip_id} failed in the render worker: '
                                  f'{result.get("error")}\n{self._log_tail()}')
            if self.process.poll() is not None:
                raise WorkerError(f'Render worker died while rendering clip {clip_id} '
                                  f'(code {self.process.returncode}).\n{self._log_tail()}')
            time.sleep(0.2)
        raise WorkerError(f'Clip {clip_id} timed out after {poll_timeout}s.\n{self._log_tail()}')

    def _log_tail(self, limit=3500):
        try:
            return self.log_path.read_text(encoding='utf-8', errors='replace')[-limit:]
        except OSError:
            return ''


class RenderPool:
    """One resident worker per GPU, all draining a single shared clip queue.

    Independent clips render in parallel across every assigned GPU (workers self-balance
    by atomically claiming requests). If a worker dies, the clips it had claimed are
    requeued for the survivors, so a single GPU fault degrades speed instead of failing
    the job. With one GPU this behaves exactly like a single resident worker.
    """

    def __init__(self, queue_dir, gpus, *, version='1.6', offload='auto', cpu_threads=4,
                 seed=1247, crop_smoothing=0.2,
                 parent_pid=None, log_dir=None, startup_timeout=1200):
        self.queue = Path(queue_dir)
        self.requests = self.queue / 'requests'
        self.results = self.queue / 'results'
        self.gpus = list(gpus)
        if not self.gpus:
            raise WorkerError('RenderPool needs at least one GPU index.')
        self.version = version
        self.offload = offload
        self.cpu_threads = cpu_threads
        self.seed = seed
        self.crop_smoothing = crop_smoothing
        self.parent_pid = parent_pid
        self.log_dir = Path(log_dir) if log_dir else self.queue
        self.startup_timeout = startup_timeout
        self.workers = []   # each: {proc, tag, device, ready_file, log, ready, requeued}

    # ------------------------------------------------------------- lifecycle
    def __enter__(self):
        self.requests.mkdir(parents=True, exist_ok=True)
        self.results.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        (self.queue / 'STOP').unlink(missing_ok=True)
        for device in self.gpus:
            tag = f'g{device}'
            ready_file = self.queue / f'ready_{tag}.json'
            ready_file.unlink(missing_ok=True)
            log_path = self.log_dir / f'worker_{tag}.log'
            log = log_path.open('w', encoding='utf-8')
            command = [str(WORKER_PYTHON), '-u', str(ROOT / 'inference_worker.py'),
                       '--queue', str(self.queue), '--version', str(self.version),
                       '--offload', str(self.offload), '--cpu-threads', str(self.cpu_threads),
                       '--seed', str(self.seed), '--crop-smoothing', str(self.crop_smoothing),
                       '--ready-file', str(ready_file), '--worker-tag', tag,
                       '--gpu-lock-name', f'gpu_{device}.lock']
            if self.parent_pid:
                command += ['--parent-pid', str(self.parent_pid)]
            env = {**os.environ, 'PYTHONIOENCODING': 'utf-8',
                   'CUDA_VISIBLE_DEVICES': str(device)}
            proc = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                    env=env, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            self.workers.append({'proc': proc, 'tag': tag, 'device': device,
                                 'ready_file': ready_file, 'log': log, 'log_path': log_path,
                                 'ready': False, 'requeued': False})
        self._await_ready()
        return self

    def _await_ready(self):
        deadline = time.monotonic() + self.startup_timeout
        while time.monotonic() < deadline:
            for w in self.workers:
                if not w['ready'] and w['ready_file'].is_file():
                    w['ready'] = True
            if all(w['ready'] for w in self.workers):
                break
            # A worker may die on startup (e.g. a second card that can't fit the model);
            # continue as long as at least one is coming up.
            if all(w['ready'] or w['proc'].poll() is not None for w in self.workers):
                break
            time.sleep(0.25)
        self.ready_workers = [w for w in self.workers if w['ready']]
        if not self.ready_workers:
            raise WorkerError('No render worker finished loading the model.\n' + self._logs_tail())
        return self.ready_workers

    @property
    def active_gpu_count(self):
        return sum(1 for w in self.workers if w['proc'].poll() is None)

    def __exit__(self, *exc):
        try:
            (self.queue / 'STOP').write_text('stop', encoding='ascii')
            for w in self.workers:
                proc = w['proc']
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.terminate()
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        proc.kill()
        finally:
            for w in self.workers:
                try:
                    w['log'].close()
                except OSError:
                    pass
        return False

    # --------------------------------------------------------------- render
    def _submit(self, clip):
        clip_id = str(clip['id'])
        (self.results / f'res_{clip_id}.json').unlink(missing_ok=True)
        spec = {'id': clip_id, 'video': str(Path(clip['video']).resolve()),
                'audio': str(Path(clip['audio']).resolve()),
                'output': str(Path(clip['output']).resolve()),
                'frames': int(clip.get('frames', 16)), 'steps': int(clip.get('steps', 20))}
        if clip.get('guidance_scale') is not None:
            spec['guidance_scale'] = float(clip['guidance_scale'])
        request = self.requests / f'req_{clip_id}.json'
        tmp = request.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(spec, ensure_ascii=False), encoding='utf-8')
        tmp.replace(request)

    def _requeue_dead_claims(self):
        """Return clips claimed by a now-dead worker to the queue for the survivors."""
        for w in self.workers:
            if w['requeued'] or w['proc'].poll() is None:
                continue
            for claimed in self.requests.glob(f"req_*.json.{w['tag']}.claimed"):
                original = claimed.with_name(claimed.name[:-len(f".{w['tag']}.claimed")])
                try:
                    claimed.replace(original)
                except OSError:
                    pass
            w['requeued'] = True

    def render_all(self, clips, on_done=None, poll_timeout=36000):
        """Render every clip across the pool. Returns {id: stats}. on_done(id, stats)
        fires as each clip finishes (order not guaranteed)."""
        if not clips:
            return {}
        for clip in clips:
            self._submit(clip)
        remaining = {str(clip['id']) for clip in clips}
        collected = {}
        deadline = time.monotonic() + poll_timeout
        while remaining:
            for clip_id in list(remaining):
                result_path = self.results / f'res_{clip_id}.json'
                if not result_path.is_file():
                    continue
                result = json.loads(result_path.read_text(encoding='utf-8'))
                result_path.unlink(missing_ok=True)
                if result.get('status') != 'ok':
                    raise WorkerError(f'Clip {clip_id} failed in a render worker: '
                                      f'{result.get("error")}\n{self._logs_tail()}')
                collected[clip_id] = result['stats']
                remaining.discard(clip_id)
                if on_done:
                    on_done(clip_id, result['stats'])
            if not remaining:
                break
            if self.active_gpu_count == 0:
                raise WorkerError('All render workers died before the job finished.\n'
                                  + self._logs_tail())
            self._requeue_dead_claims()
            if time.monotonic() > deadline:
                raise WorkerError(f'Render pool timed out with {len(remaining)} clips left.\n'
                                  + self._logs_tail())
            time.sleep(0.2)
        return collected

    def _logs_tail(self, limit=2000):
        parts = []
        for w in self.workers:
            try:
                tail = w['log_path'].read_text(encoding='utf-8', errors='replace')[-limit:]
            except OSError:
                tail = ''
            parts.append(f"--- worker {w['tag']} (GPU {w['device']}) ---\n{tail}")
        return '\n'.join(parts)
