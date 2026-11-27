"""servicectl — CLI to exercise the Phase-0 service pipeline (analyze / synthesize).

This is the thin worker-side entrypoint. It adds the LatentSync-trial pipeline to
the path and calls service_pipeline. The Rust API (../api-rust) will later drive the
same functions through a queue; this CLI lets us test the two operations end to end
before any Rust exists.

Examples:
  python servicectl.py analyze  ../../LatentSync-trial/trial_inputs/sample_ref.mp4
  python servicectl.py synthesize <selection_id> 1 hi --out ./out.mp4
  python servicectl.py release   <selection_id>
  python servicectl.py purge --ttl 1800
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]            # repo root
PIPELINE = ROOT / 'LatentSync-trial'
sys.path.insert(0, str(PIPELINE))

import service_pipeline as svc   # noqa: E402  (after sys.path setup)

DEFAULT_WORK_ROOT = Path(
    os.environ.get('SERVICE_WORK_ROOT', PIPELINE / 'local_jobs' / 'selections'))


def _parse_voices(text):
    voices = {}
    for part in (text or '').split(','):
        part = part.strip()
        if not part:
            continue
        if '=' not in part:
            raise SystemExit(f'--voices entry must be "Speaker N=voice_id": {part!r}')
        label, voice = part.split('=', 1)
        voices[label.strip()] = voice.strip()
    return voices


def _settings_from_args(args):
    settings = {}
    for key in ('steps', 'frames', 'seed'):
        value = getattr(args, key, None)
        if value is not None:
            settings[key] = value
    for key in ('guidance_scale', 'speed'):
        value = getattr(args, key, None)
        if value is not None:
            settings[key] = value
    if getattr(args, 'section_seconds', None) is not None:
        settings['section_seconds'] = args.section_seconds
    return settings


def cmd_analyze(args):
    manifest = svc.analyze(
        args.media, args.work_root,
        source_language=args.source_language,
        settings=_settings_from_args(args))
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


def cmd_synthesize(args):
    result = svc.synthesize(
        args.selection_id, args.section_id, args.target_language, args.work_root,
        speaker_voices=_parse_voices(args.voices),
        settings=_settings_from_args(args))
    video = Path(result['video_path'])
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(video), str(out))   # deliver to the caller, keep nothing
        result['video_path'] = str(out)
    print(json.dumps(result, indent=2, ensure_ascii=False))


def cmd_release(args):
    existed = svc.release(args.selection_id, args.work_root)
    print(json.dumps({'selection_id': args.selection_id, 'released': existed}))


def cmd_purge(args):
    purged = svc.purge_expired(args.work_root, args.ttl)
    print(json.dumps({'purged': purged, 'count': len(purged)}))


def build_parser():
    parser = argparse.ArgumentParser(description='Service pipeline CLI (Phase 0).')
    parser.add_argument('--work-root', type=Path, default=DEFAULT_WORK_ROOT,
                        help=f'Selection cache root (default: {DEFAULT_WORK_ROOT}).')
    sub = parser.add_subparsers(dest='command', required=True)

    a = sub.add_parser('analyze', help='Prepare a selection (separate/STT/diarize).')
    a.add_argument('media')
    a.add_argument('--source-language', default=None)
    a.add_argument('--section-seconds', type=float, default=None)
    a.set_defaults(func=cmd_analyze)

    s = sub.add_parser('synthesize', help='Render one section in a target language.')
    s.add_argument('selection_id')
    s.add_argument('section_id', type=int)
    s.add_argument('target_language')
    s.add_argument('--out', default=None, help='Write the delivered clip here.')
    s.add_argument('--voices', default='', help='"Speaker 1=voice_id,Speaker 2=voice_id".')
    s.add_argument('--steps', type=int, default=None)
    s.add_argument('--frames', type=int, default=None)
    s.add_argument('--seed', type=int, default=None)
    s.add_argument('--guidance-scale', dest='guidance_scale', type=float, default=None)
    s.add_argument('--speed', type=float, default=None)
    s.set_defaults(func=cmd_synthesize)

    r = sub.add_parser('release', help='Drop a prepared selection.')
    r.add_argument('selection_id')
    r.set_defaults(func=cmd_release)

    p = sub.add_parser('purge', help='Remove selections older than the TTL.')
    p.add_argument('--ttl', type=int, default=svc.DEFAULT_TTL_SECONDS)
    p.set_defaults(func=cmd_purge)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except svc.ServiceError as exc:
        print(json.dumps({'error': type(exc).__name__, 'status': exc.http_status,
                          'message': str(exc)}), file=sys.stderr)
        raise SystemExit(2)


if __name__ == '__main__':
    main()
