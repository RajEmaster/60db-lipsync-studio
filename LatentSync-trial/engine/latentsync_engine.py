"""Resident LatentSync engine: load the model once, render many clips without reloading.

This replaces the per-clip cold-start in trial_inference.py. The model stack (UNet +
VAE + Whisper) stays loaded for the whole lifetime of the process, so a long video that
is split into dozens of clips pays the multi-GB load cost exactly once instead of once
per clip. The proven memory choreography from trial_inference.py is preserved and is now
selectable per deployment:

  offload='resident' - UNet and VAE both stay on CUDA. Fastest; needs VRAM headroom.
  offload='swap'     - UNet and VAE take turns on CUDA (the 4 GB / tight-16 GB trick).
  offload='auto'     - pick 'resident' when the card clearly has headroom, else 'swap'.

Version selection (1.5 vs 1.6) only changes the UNet config (256 vs 512 resolution) and
which checkpoint file is loaded. The architecture, VAE, Whisper, and scheduler are shared.
"""
from __future__ import annotations

import gc
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# LatentSync and its vendored InsightFace must resolve exactly as they do for the
# standalone worker, regardless of the current working directory of the caller.
_VENDOR = ROOT / 'vendor' / 'insightface-0.7.3'
if _VENDOR.is_dir() and str(_VENDOR) not in sys.path:
    sys.path.insert(0, str(_VENDOR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# version -> (unet config relative to ROOT, checkpoint relative to ROOT, default guidance)
ENGINE_VERSIONS = {
    '1.5': {
        'config': 'configs/unet/stage2.yaml',
        'checkpoint': 'checkpoints/latentsync_unet.pt',
        'guidance_scale': 1.5,   # matches the official stage2.yaml / gradio default
    },
    '1.6': {
        'config': 'configs/unet/stage2_512.yaml',
        'checkpoint': 'checkpoints/v1.6/latentsync_unet.pt',
        'guidance_scale': 1.5,
    },
}

# Below this much total VRAM (MiB) 'auto' prefers swapping over keeping both models resident.
# 1.5 (256px) peaks at ~3 GB, so keep it resident on any >=5 GB card: swap re-runs
# torch.cuda.empty_cache() on EVERY UNet forward (per-step), which dominates render time on
# small GPUs. 1.6 (512px) genuinely needs the headroom, so its floor stays high.
RESIDENT_VRAM_FLOOR_MIB = {'1.5': 5_000, '1.6': 22_000}


@dataclass
class EngineConfig:
    version: str = '1.6'
    offload: str = 'auto'          # 'auto' | 'resident' | 'swap'
    cpu_threads: int = 4
    seed: int = 1247
    crop_smoothing: float = 0.2    # mouth-crop position EMA history weight (0.2 = LatentSync default)
    face_restore: bool = False     # GFPGAN (ONNX) polish on the generated face
    restore_strength: float = 0.5  # 0..1 blend of restored vs original (lower = subtler, less flicker)
    checkpoint: str | None = None  # override the version default
    config_path: str | None = None # override the version default


class LatentSyncEngine:
    """A long-lived LatentSync pipeline. Build once, then call render_clip() per clip."""

    def __init__(self, config: EngineConfig):
        if config.version not in ENGINE_VERSIONS:
            raise ValueError(f'Unknown engine version {config.version!r}; '
                             f'choose one of {sorted(ENGINE_VERSIONS)}.')
        self.config = config
        self._build()

    # ------------------------------------------------------------------ setup
    def _resolve_paths(self):
        spec = ENGINE_VERSIONS[self.config.version]
        config_path = Path(self.config.config_path or (ROOT / spec['config']))
        checkpoint = Path(self.config.checkpoint or (ROOT / spec['checkpoint']))
        if not config_path.is_file():
            raise FileNotFoundError(f'UNet config not found: {config_path}')
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f'LatentSync {self.config.version} checkpoint not found: {checkpoint}. '
                'Download it before starting the engine.')
        return config_path, checkpoint

    def _choose_offload(self, total_vram_mib: float) -> str:
        if self.config.offload in ('resident', 'swap'):
            return self.config.offload
        floor = RESIDENT_VRAM_FLOOR_MIB.get(self.config.version, 22_000)
        return 'resident' if total_vram_mib >= floor else 'swap'

    def _build(self):
        import torch
        from accelerate import init_empty_weights
        from diffusers import AutoencoderKL, DDIMScheduler
        from omegaconf import OmegaConf
        from latentsync.models.unet import UNet3DConditionModel
        from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
        from latentsync.whisper.audio2feature import Audio2Feature
        from latentsync.utils.affine_transform import AlignRestore

        self.torch = torch
        torch.set_num_threads(max(1, int(self.config.cpu_threads)))
        torch.manual_seed(self.config.seed)
        # The face-crop restorer is re-created inside the pipeline per clip, so steer its
        # temporal smoothing via a class default set once here. Higher = steadier mouth.
        AlignRestore.smooth_history = float(self.config.crop_smoothing)
        os.chdir(ROOT)  # LatentSync reads mask images / temp dirs relative to the repo root.

        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is not available; the LatentSync engine needs a GPU.')
        self.device = torch.device('cuda')
        props = torch.cuda.get_device_properties(0)
        total_vram_mib = props.total_memory / 2 ** 20
        self.offload = self._choose_offload(total_vram_mib)
        # Match the official LatentSync precision policy: fp16 only on Ampere+ (compute
        # capability >= 8). On Turing/Volta (e.g. the Kaggle T4, cc 7.5) the authors use
        # fp32 for numerical stability, so we follow suit instead of forcing fp16 there.
        capability = torch.cuda.get_device_capability(0)
        self.weight_dtype = torch.float16 if capability[0] >= 8 else torch.float32
        print(f'Engine GPU: {props.name} ({total_vram_mib:.0f} MiB total), '
              f'compute {capability[0]}.{capability[1]} -> {self.weight_dtype}. '
              f'LatentSync {self.config.version}, offload={self.offload}.', flush=True)

        config_path, checkpoint_path = self._resolve_paths()
        self._unet_config = OmegaConf.load(config_path)
        self.resolution = int(self._unet_config.data.resolution)
        self._mask_image_path = self._unet_config.data.mask_image_path

        started = time.monotonic()
        print(f'Loading {self.config.version} UNet from {checkpoint_path.name} (mmap)…', flush=True)
        checkpoint = torch.load(checkpoint_path, map_location='cpu', mmap=True, weights_only=True)
        state = checkpoint['state_dict']
        with init_empty_weights():
            unet = UNet3DConditionModel.from_config(OmegaConf.to_container(self._unet_config.model))
        state = {key: value.to(dtype=self.weight_dtype) if value.is_floating_point() else value
                 for key, value in state.items()}
        torch.nn.Module.load_state_dict(unet, state, strict=True, assign=True)
        del checkpoint, state
        gc.collect()
        unet.eval().requires_grad_(False)
        self.unet = unet

        vae = AutoencoderKL.from_pretrained(str(ROOT / 'checkpoints' / 'vae'),
                                            torch_dtype=self.weight_dtype, local_files_only=True)
        vae.config.scaling_factor = 0.18215
        vae.config.shift_factor = 0
        vae.enable_slicing()   # lossless per-item batch decode; tiling is unused (<=512px) and omitted
        self.vae = vae

        # Whisper audio encoding runs on the GPU (frees the CPU, and the model is tiny,
        # ~75 MB). Its features are produced on CPU and moved to CUDA per clip inside the
        # pipeline's __call__.
        self.audio_encoder = Audio2Feature(
            model_path=str(ROOT / 'checkpoints' / 'whisper' / 'tiny.pt'),
            device='cuda', num_frames=16)
        self.pipeline = LipsyncPipeline(
            vae=vae, audio_encoder=self.audio_encoder, unet=unet,
            scheduler=DDIMScheduler.from_pretrained(str(ROOT / 'configs')))
        # Keep CPU Whisper out of the pipeline's device discovery; features are copied to
        # CUDA inside __call__. Other components are staged explicitly below.
        self.pipeline.__class__._execution_device = property(lambda _self: torch.device('cuda'))
        self._install_memory_policy()

        # Optional GFPGAN (ONNX) face restoration. Loaded once; a missing model or any
        # failure degrades to a no-op inside FaceRestorer, so it never breaks a render.
        self.pipeline.face_restorer = None
        if self.config.face_restore:
            from latentsync.utils.face_restore import FaceRestorer
            self.pipeline.face_restorer = FaceRestorer(
                ROOT / 'checkpoints' / 'restore' / 'gfpgan.onnx',
                device='cuda', strength=self.config.restore_strength)

        if self.offload == 'resident':
            unet.to('cuda')
            vae.to('cuda')
        else:
            unet.to('cpu')
            vae.to('cpu')
        torch.cuda.empty_cache()
        print(f'Engine ready in {time.monotonic() - started:.1f}s.', flush=True)

    def _install_memory_policy(self):
        """Stage UNet and VAE on CUDA at the right moments for the chosen offload mode."""
        torch = self.torch
        pipeline = self.pipeline
        original_mask = pipeline.prepare_mask_latents
        original_images = pipeline.prepare_image_latents
        original_decode = pipeline.decode_latents
        original_forward = self.unet.forward

        def to_vae_stage():
            if self.offload == 'swap':
                self.unet.to('cpu')
                torch.cuda.empty_cache()
                self.vae.to('cuda')

        def to_unet_stage():
            if self.offload == 'swap':
                self.vae.to('cpu')
                torch.cuda.empty_cache()
                self.unet.to('cuda')

        def mask(*a, **kw):
            to_vae_stage()
            return original_mask(*a, **kw)

        def images(*a, **kw):
            return original_images(*a, **kw)

        def decode(*a, **kw):
            to_vae_stage()
            return original_decode(*a, **kw)

        def forward(*a, **kw):
            to_unet_stage()
            return original_forward(*a, **kw)

        pipeline.prepare_mask_latents = mask
        pipeline.prepare_image_latents = images
        pipeline.decode_latents = decode
        self.unet.forward = forward

    # ----------------------------------------------------------------- render
    def render_clip(self, video_path, audio_path, output_path, *, frames=16, steps=20,
                    guidance_scale=None, temp_dir=None):
        """Render one clip with the already-loaded model. Returns timing/memory stats."""
        torch = self.torch
        video_path, audio_path, output_path = Path(video_path), Path(audio_path), Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if guidance_scale is None:
            guidance_scale = ENGINE_VERSIONS[self.config.version]['guidance_scale']
        # A temp dir UNIQUE PER CLIP: several workers render in parallel across GPUs and
        # must not share LatentSync's intermediate files, or they corrupt each other's
        # frames (0 faces -> ZeroDivisionError). Cleared first so a re-render starts fresh.
        if temp_dir is None:
            temp_dir = output_path.parent / f'engine_temp_{output_path.stem}'
        temp_dir = Path(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)
        temp_dir.mkdir(parents=True, exist_ok=True)
        self.audio_encoder.num_frames = int(frames)

        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        self.pipeline(
            video_path=str(video_path), audio_path=str(audio_path),
            video_out_path=str(output_path), num_frames=int(frames),
            num_inference_steps=int(steps), guidance_scale=float(guidance_scale),
            weight_dtype=self.weight_dtype, width=self.resolution, height=self.resolution,
            mask_image_path=self._mask_image_path, temp_dir=str(temp_dir))
        if not output_path.is_file():
            raise RuntimeError('No output video was produced by the engine.')
        shutil.rmtree(temp_dir, ignore_errors=True)   # free disk between clips
        return {
            'output': str(output_path),
            'seconds': round(time.monotonic() - started, 2),
            'peak_cuda_allocated_mib': round(torch.cuda.max_memory_allocated() / 2 ** 20),
            'peak_cuda_reserved_mib': round(torch.cuda.max_memory_reserved() / 2 ** 20),
            'frames_per_batch': int(frames), 'steps': int(steps),
            'guidance_scale': float(guidance_scale), 'resolution': self.resolution,
            'version': self.config.version, 'offload': self.offload,
            'crop_smoothing': float(self.config.crop_smoothing), 'seed': int(self.config.seed),
            'face_restore': bool(self.config.face_restore),
            'restore_strength': float(self.config.restore_strength),
        }

    def close(self):
        try:
            self.unet.to('cpu')
            self.vae.to('cpu')
        except Exception:
            pass
        self.torch.cuda.empty_cache()
