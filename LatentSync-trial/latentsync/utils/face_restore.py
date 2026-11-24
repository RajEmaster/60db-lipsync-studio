"""Optional GFPGAN (ONNX) face restoration for the LatentSync-generated face.

Runs through onnxruntime (already a dependency), so it adds no torch / basicsr version
conflicts the way the pip GFPGAN packages do. It enhances the aligned face the UNet
produced; because only the mouth region is pasted back downstream, restoration polishes
exactly the pixels LatentSync generated and leaves the original (already sharp) face alone.

A `strength` in [0, 1] blends restored vs. original to limit temporal flicker / identity
drift. Any load or inference failure degrades to a silent no-op, so enabling restoration can
never break a render — at worst you get the unrestored frame.
"""
from pathlib import Path

import numpy as np

try:
    import onnxruntime as ort
except Exception:   # pragma: no cover - onnxruntime is expected, but never hard-fail here
    ort = None


class FaceRestorer:
    def __init__(self, model_path, device="cuda", strength=0.5):
        self.ok = False
        self.strength = float(max(0.0, min(1.0, strength)))
        self.size = 512
        model_path = Path(model_path)
        if ort is None or self.strength <= 0 or not model_path.is_file():
            if self.strength > 0 and not model_path.is_file():
                print(f"Face restore off: model not found at {model_path}. "
                      "Download it (download_trial.py --restore) to enable.", flush=True)
            return
        providers = ["CPUExecutionProvider"]
        if str(device).startswith("cuda") and "CUDAExecutionProvider" in ort.get_available_providers():
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        try:
            self.session = ort.InferenceSession(str(model_path), providers=providers)
            inp = self.session.get_inputs()[0]
            self.input_name = inp.name
            self.output_name = self.session.get_outputs()[0].name
            # Honor a fixed input size when the export declares one; else assume 512.
            dim = inp.shape[2] if len(inp.shape) == 4 else None
            self.size = int(dim) if isinstance(dim, int) and dim > 0 else 512
            self.ok = True
            print(f"Face restore ready: GFPGAN {self.size}px, strength {self.strength}.", flush=True)
        except Exception as exc:
            print(f"Face restore disabled (failed to load {model_path.name}): {exc}", flush=True)

    def enhance_batch(self, faces):
        """faces: torch tensor (N, 3, res, res) in [-1, 1], RGB. Returns the same shape/range
        with each face blended toward its GFPGAN-restored version by `strength`."""
        if not self.ok:
            return faces
        import torch
        res = faces.shape[-1]
        out = faces.clone()
        for i in range(faces.shape[0]):
            face = faces[i:i + 1].float()   # (1, 3, res, res) in [-1, 1]
            net_in = torch.nn.functional.interpolate(
                face, size=(self.size, self.size), mode="bilinear", align_corners=False)
            arr = np.ascontiguousarray(net_in.clamp(-1, 1).cpu().numpy().astype(np.float32))
            try:
                restored = self.session.run([self.output_name], {self.input_name: arr})[0]
            except Exception as exc:   # one bad frame must not stop the render
                print(f"Face restore skipped a frame: {exc}", flush=True)
                continue
            restored = np.asarray(restored, dtype=np.float32)
            # Some exports emit [0, 1] instead of [-1, 1]; normalize to [-1, 1] if so.
            if restored.min() >= -0.05:
                restored = restored * 2.0 - 1.0
            rt = torch.from_numpy(np.ascontiguousarray(restored)).to(device=face.device)
            if rt.ndim == 3:
                rt = rt.unsqueeze(0)
            rt = torch.nn.functional.interpolate(
                rt, size=(res, res), mode="bilinear", align_corners=False).clamp(-1, 1)
            blended = self.strength * rt + (1.0 - self.strength) * face
            out[i] = blended[0].to(out.dtype)
        return out
