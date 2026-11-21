"""Long-lived GPU render worker: load LatentSync once, drain a clip queue.

Runs under WORKER_PYTHON (the .venv that has torch + latentsync). A caller (render_chunks
or, later, the pool manager) drops clip requests into <queue>/requests and reads results
from <queue>/results. The model is loaded a single time for the worker's whole lifetime,
so every clip after the first renders with no reload — this is the core latency win over
the previous cold-subprocess-per-clip design.

Queue protocol (plain files, atomic via rename; durable and resumable):
  <queue>/requests/req_<id>.json   {id, video, audio, output, frames, steps, guidance_scale}
  <queue>/results/res_<id>.json    {id, status: 'ok'|'error', stats?, error?}
  <queue>/STOP                     presence tells the worker to finish and exit
The worker claims a request by renaming it to req_<id>.json.claimed before rendering.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def _write_json_atomic(path: Path, data: dict):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')
    tmp.replace(path)


def _claim(request: Path, tag: str):
    # The tag records which worker holds the clip, so the pool can requeue it if that
    # worker dies. Several workers racing for the same file: exactly one rename wins.
    claimed = request.with_name(f'{request.name}.{tag}.claimed')
    try:
        request.replace(claimed)   # atomic; loser of a race gets FileNotFoundError
        return claimed
    except (FileNotFoundError, OSError):
        return None


def _pending_requests(requests_dir: Path):
    return sorted(requests_dir.glob('req_*.json'), key=lambda p: p.name)


def run(args):
    from engine import EngineConfig, LatentSyncEngine
    from worker_lifecycle import exclusive_job_resource

    queue = Path(args.queue)
    requests_dir = queue / 'requests'
    results_dir = queue / 'results'
    requests_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    stop_file = queue / 'STOP'
    tag = args.worker_tag or str(os.getpid())

    # One worker per physical GPU: each takes a device-specific lock so workers on
    # different cards never block each other (the engine sees its card as cuda:0 via
    # CUDA_VISIBLE_DEVICES, set by the pool manager).
    with exclusive_job_resource(args.gpu_lock_name, 'Another render owns this GPU; waiting.'):
        engine = LatentSyncEngine(EngineConfig(
            version=args.version, offload=args.offload, cpu_threads=args.cpu_threads))
        visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        if args.ready_file:
            _write_json_atomic(Path(args.ready_file),
                               {'ready': True, 'version': engine.config.version,
                                'offload': engine.offload, 'resolution': engine.resolution,
                                'tag': tag, 'cuda_visible_devices': visible})
        print(f'WORKER_READY tag={tag} CUDA_VISIBLE_DEVICES={visible!r}', flush=True)

        idle_since = time.monotonic()
        while True:
            if stop_file.exists():
                print('WORKER_STOP file seen; exiting.', flush=True)
                break
            pending = _pending_requests(requests_dir)
            if not pending:
                if args.idle_seconds and time.monotonic() - idle_since > args.idle_seconds:
                    print('WORKER_IDLE timeout; exiting.', flush=True)
                    break
                time.sleep(0.2)
                continue
            for request in pending:
                claimed = _claim(request, tag)
                if claimed is None:
                    continue
                try:
                    spec = json.loads(claimed.read_text(encoding='utf-8'))
                except (OSError, ValueError) as exc:
                    claimed.unlink(missing_ok=True)
                    print(f'Skipped unreadable request {request.name}: {exc}', flush=True)
                    continue
                clip_id = spec['id']
                result_path = results_dir / f'res_{clip_id}.json'
                try:
                    stats = engine.render_clip(
                        spec['video'], spec['audio'], spec['output'],
                        frames=spec.get('frames', 16), steps=spec.get('steps', 20),
                        guidance_scale=spec.get('guidance_scale'))
                    _write_json_atomic(result_path, {'id': clip_id, 'status': 'ok', 'stats': stats})
                    print(f'CLIP_OK {clip_id} {stats["seconds"]:.1f}s '
                          f'(peak {stats["peak_cuda_reserved_mib"]} MiB)', flush=True)
                except Exception as exc:  # one bad clip must not kill the resident worker
                    _write_json_atomic(result_path, {'id': clip_id, 'status': 'error',
                                                     'error': repr(exc)})
                    print(f'CLIP_ERROR {clip_id}: {exc!r}', flush=True)
                finally:
                    claimed.unlink(missing_ok=True)
                idle_since = time.monotonic()
        engine.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue', required=True, help='Queue directory with requests/ and results/')
    parser.add_argument('--version', default='1.6', help='LatentSync engine version (1.5 or 1.6)')
    parser.add_argument('--offload', default='auto', choices=['auto', 'resident', 'swap'])
    parser.add_argument('--cpu-threads', type=int, default=4)
    parser.add_argument('--parent-pid', type=int, help='Owning process; worker stops if it exits')
    parser.add_argument('--idle-seconds', type=float, default=0,
                        help='Exit after this many seconds with no work (0 = never)')
    parser.add_argument('--ready-file', help='Write a JSON marker here once the model is loaded')
    parser.add_argument('--gpu-lock-name', default='gpu.lock',
                        help='Per-device lock file so one worker owns each GPU')
    parser.add_argument('--worker-tag', help='Unique tag stamped on claimed clips (default: pid)')
    args = parser.parse_args()
    if args.parent_pid:
        from worker_lifecycle import watch_parent
        watch_parent(args.parent_pid)
    run(args)


if __name__ == '__main__':
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='replace')
    main()
