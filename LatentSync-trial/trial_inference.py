"""Isolated LatentSync 1.5 trial for a 4 GB GPU; experimental memory settings."""
import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / 'vendor' / 'insightface-0.7.3'))
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, 'reconfigure'):
        stream.reconfigure(encoding='utf-8', errors='replace')

import torch
from accelerate import init_empty_weights
from diffusers import AutoencoderKL, DDIMScheduler
from omegaconf import OmegaConf
from latentsync.models.unet import UNet3DConditionModel
from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
from latentsync.whisper.audio2feature import Audio2Feature


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', default='trial_inputs/video.mp4')
    parser.add_argument('--audio', default='trial_inputs/audio.wav')
    parser.add_argument('--output', default='trial_outputs/smoke.mp4')
    parser.add_argument('--frames', type=int, default=4)
    parser.add_argument('--steps', type=int, default=2)
    parser.add_argument('--sequential-offload', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(1247)
    started = time.monotonic()
    print('GPU:', torch.cuda.get_device_name(0), flush=True)
    config = OmegaConf.load('configs/unet/stage2.yaml')
    print('Loading 1.5 checkpoint with memory mapping...', flush=True)
    checkpoint = torch.load('checkpoints/latentsync_unet.pt', map_location='cpu', mmap=True, weights_only=True)
    state = checkpoint['state_dict']
    with init_empty_weights():
        unet = UNet3DConditionModel.from_config(OmegaConf.to_container(config.model))
    state = {key: value.to(dtype=torch.float16) if value.is_floating_point() else value
             for key, value in state.items()}
    result = torch.nn.Module.load_state_dict(unet, state, strict=True, assign=True)
    print('Checkpoint:', result, flush=True)
    del checkpoint, state
    gc.collect()
    unet.eval().requires_grad_(False)
    vae = AutoencoderKL.from_pretrained('checkpoints/vae', torch_dtype=torch.float16, local_files_only=True)
    vae.config.scaling_factor = 0.18215
    vae.config.shift_factor = 0
    vae.enable_slicing()
    vae.enable_tiling()
    audio = Audio2Feature(model_path='checkpoints/whisper/tiny.pt', device='cpu', num_frames=args.frames)
    pipeline = LipsyncPipeline(vae=vae, audio_encoder=audio, unet=unet,
                              scheduler=DDIMScheduler.from_pretrained('configs'))
    # Keep CPU Whisper out of DiffusionPipeline's device discovery; its features
    # are copied to CUDA by __call__. The other components are staged explicitly.
    pipeline.__class__._execution_device = property(lambda self: torch.device('cuda'))
    original_mask = pipeline.prepare_mask_latents
    original_images = pipeline.prepare_image_latents
    original_decode = pipeline.decode_latents

    def vae_stage():
        if not args.sequential_offload:
            unet.to('cpu')
        torch.cuda.empty_cache()
        vae.to('cuda')

    def mask(*a, **kw):
        vae_stage()
        return original_mask(*a, **kw)

    def images(*a, **kw):
        return original_images(*a, **kw)

    def decode(*a, **kw):
        vae_stage()
        return original_decode(*a, **kw)

    pipeline.prepare_mask_latents = mask
    pipeline.prepare_image_latents = images
    pipeline.decode_latents = decode
    if args.sequential_offload:
        from accelerate import cpu_offload
        cpu_offload(unet, execution_device=torch.device('cuda'))
    original_forward = unet.forward

    def forward(*a, **kw):
        vae.to('cpu')
        torch.cuda.empty_cache()
        if not args.sequential_offload:
            unet.to('cuda')
        return original_forward(*a, **kw)

    unet.forward = forward
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    print(f'Rendering: {args.frames} frames per batch, {args.steps} steps', flush=True)
    pipeline(video_path=args.video, audio_path=args.audio, video_out_path=args.output,
             num_frames=args.frames, num_inference_steps=args.steps, guidance_scale=1.0,
             weight_dtype=torch.float16, width=256, height=256,
             mask_image_path=config.data.mask_image_path, temp_dir='trial_temp')
    if not Path(args.output).is_file():
        raise RuntimeError('No output video was produced')
    stats = {'output': args.output, 'seconds': round(time.monotonic() - started, 2),
             'peak_cuda_allocated_mib': round(torch.cuda.max_memory_allocated() / 2**20),
             'peak_cuda_reserved_mib': round(torch.cuda.max_memory_reserved() / 2**20),
             'frames_per_batch': args.frames, 'steps': args.steps,
             'sequential_offload': args.sequential_offload}
    Path(args.output + '.json').write_text(json.dumps(stats, indent=2))
    print('TRIAL_SUCCESS', json.dumps(stats), flush=True)


if __name__ == '__main__':
    from worker_lifecycle import exclusive_gpu
    with exclusive_gpu():
        main()
