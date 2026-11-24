from insightface.app import FaceAnalysis
import numpy as np
import onnxruntime as ort
import torch

INSIGHTFACE_DETECT_SIZE = 512

# Lip-sync looks wrong on strongly non-frontal faces (the mouth gets pasted onto a
# profile where it doesn't belong). When the detected face is too turned, we skip it so
# the frame passes through UNCHANGED (original mouth kept) instead of showing a bad sync.
# 1.0 == dead frontal; a near-profile approaches 0. Conservative by default so only clearly
# turned heads are skipped; raise toward 0.45 to be stricter on movie footage.
FRONTAL_MIN_SYMMETRY = 0.25


class FaceDetector:
    def __init__(self, device="cuda"):
        # Run detection on the GPU when a CUDA ONNX Runtime is available (big CPU win);
        # fall back to CPU automatically otherwise so nothing breaks on CPU-only installs.
        providers = ["CPUExecutionProvider"]
        if str(device).startswith("cuda") and "CUDAExecutionProvider" in ort.get_available_providers():
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.app = FaceAnalysis(
            allowed_modules=["detection", "landmark_2d_106"],
            root="checkpoints/auxiliary",
            providers=providers,
        )
        self.app.prepare(ctx_id=cuda_to_int(device), det_size=(INSIGHTFACE_DETECT_SIZE, INSIGHTFACE_DETECT_SIZE))

    def __call__(self, frame, threshold=0.5):
        f_h, f_w, _ = frame.shape

        faces = self.app.get(frame)

        get_face_store = None
        max_size = 0

        if len(faces) == 0:
            return None, None
        else:
            for face in faces:
                bbox = face.bbox.astype(np.int_).tolist()
                w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                if w < 50 or h < 80:
                    continue
                if w / h > 1.5 or w / h < 0.2:
                    continue
                if face.det_score < threshold:
                    continue
                size_now = w * h

                if size_now > max_size:
                    max_size = size_now
                    get_face_store = face

        if get_face_store is None:
            return None, None
        else:
            face = get_face_store
            lmk = np.round(face.landmark_2d_106).astype(np.int_)

            # Frontality (yaw) gate: compare the horizontal span nose->left-brow vs
            # nose->right-brow. On a profile, one span collapses toward zero. Faces below
            # the symmetry floor are rejected -> the pipeline keeps the original frame.
            left_brow = lmk[[43, 48, 49, 51, 50]].mean(axis=0)
            right_brow = lmk[101:106].mean(axis=0)
            nose_center = lmk[[74, 77, 83, 86]].mean(axis=0)
            dl = abs(nose_center[0] - left_brow[0])
            dr = abs(right_brow[0] - nose_center[0])
            symmetry = min(dl, dr) / max(dl, dr) if max(dl, dr) > 1e-3 else 0.0
            if symmetry < FRONTAL_MIN_SYMMETRY:
                return None, None

            halk_face_coord = np.mean([lmk[74], lmk[73]], axis=0)  # lmk[73]

            sub_lmk = lmk[LMK_ADAPT_ORIGIN_ORDER]
            halk_face_dist = np.max(sub_lmk[:, 1]) - halk_face_coord[1]
            upper_bond = halk_face_coord[1] - halk_face_dist  # *0.94

            x1, y1, x2, y2 = (np.min(sub_lmk[:, 0]), int(upper_bond), np.max(sub_lmk[:, 0]), np.max(sub_lmk[:, 1]))

            if y2 - y1 <= 0 or x2 - x1 <= 0 or x1 < 0:
                x1, y1, x2, y2 = face.bbox.astype(np.int_).tolist()

            y2 += int((x2 - x1) * 0.1)
            x1 -= int((x2 - x1) * 0.05)
            x2 += int((x2 - x1) * 0.05)

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(f_w, x2)
            y2 = min(f_h, y2)

            return (x1, y1, x2, y2), lmk


def cuda_to_int(cuda_str: str) -> int:
    """
    Convert the string with format "cuda:X" to integer X.
    """
    if cuda_str == "cuda":
        return 0
    device = torch.device(cuda_str)
    if device.type != "cuda":
        raise ValueError(f"Device type must be 'cuda', got: {device.type}")
    return device.index


LMK_ADAPT_ORIGIN_ORDER = [
    1,
    10,
    12,
    14,
    16,
    3,
    5,
    7,
    0,
    23,
    21,
    19,
    32,
    30,
    28,
    26,
    17,
    43,
    48,
    49,
    51,
    50,
    102,
    103,
    104,
    105,
    101,
    73,
    74,
    86,
]
