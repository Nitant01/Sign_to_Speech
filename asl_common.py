"""
asl_common.py  -  code shared by train_landmark_v2.py, collect_webcam_samples.py
and webcam_asl_v2.py.

WHY THIS FILE EXISTS
Your old train_landmark.py had two separate copies of `normalize()` and `LandmarkMLP`
(one for training, one for the webcam). If those two copies ever differ, the webcam model
silently sees different numbers than it was trained on. Now there is only ONE copy.
"""
import numpy as np
import torch
import torch.nn as nn

FEATURE_VERSION = "v2-palm-scale+distances"   # webcam script refuses checkpoints with another value

# ---------- hand landmark helpers ----------
# MediaPipe numbering: 0 wrist | 1-4 thumb | 5-8 index | 9-12 middle | 13-16 ring | 17-20 pinky
# (tip of each finger = 4, 8, 12, 16, 20; knuckle = 5, 9, 13, 17)

def landmarks_to_raw(hand_landmarks, w, h):
    """MediaPipe result -> (21,3) array in pixel units. Used by training, collecting AND webcam,
    so all three always use exactly the same convention (x*w, y*h, z*w)."""
    return np.array([[p.x * w, p.y * h, p.z * w] for p in hand_landmarks], dtype=np.float32)


def normalize_np(pts):
    """(21,3) raw -> (21,3) normalized.
    1) wrist becomes (0,0,0)
    2) divide by the PALM size (wrist->middle knuckle + knuckle-to-knuckle width).
    The old code divided by the farthest landmark, which is a fingertip for open hands but a
    different point for fists, so the scale changed with the sign itself. Palm size does not."""
    pts = pts - pts[0]
    scale = np.linalg.norm(pts[9]) + np.linalg.norm(pts[5] - pts[17])
    return (pts / (scale + 1e-6)).astype(np.float32)


# Extra hand-shape features: distances between landmark pairs. They do not change when the hand is
# mirrored or rotated, and they directly measure what separates the confusing letters:
#   thumb tip (4) to each knuckle/middle joint -> S vs T vs N vs M (where is the thumb tucked?)
#   thumb tip to every fingertip               -> D (thumb touches fingers) vs L (thumb away)
#   neighbouring fingertips / fingertip->wrist -> U vs V vs R, open vs curled fingers
PAIRS = ([(4, i) for i in (5, 6, 9, 10, 13, 14, 17, 18, 8, 12, 16, 20, 2)] +
         [(8, 12), (12, 16), (16, 20), (8, 20), (8, 5), (12, 9), (16, 13), (20, 17),
          (8, 0), (12, 0), (16, 0), (20, 0)])
N_FEATURES = 63 + len(PAIRS)     # 63 coordinates + 25 distances = 88


def add_features(P):
    """torch (B,21,3) normalized -> torch (B, N_FEATURES)"""
    B = P.shape[0]
    d = torch.stack([(P[:, a] - P[:, b]).norm(dim=1) for a, b in PAIRS], 1)
    return torch.cat([P.reshape(B, 63), d], 1)


def features_from_raw(pts):
    """(21,3) raw pixels -> (1, N_FEATURES) torch tensor, ready for the model."""
    P = torch.tensor(normalize_np(pts)).unsqueeze(0)
    return add_features(P)


class LandmarkMLP(nn.Module):
    """Same architecture as before (256 -> 128 -> classes). Only the input size changed 63 -> 88."""
    def __init__(self, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(N_FEATURES, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, n_classes))

    def forward(self, x):
        return self.net(x)


def load_checkpoint(path, device="cpu"):
    """Load a v2 checkpoint and refuse (with a clear message) anything made by the old script."""
    ck = torch.load(path, map_location=device)
    if ck.get("feature_version") != FEATURE_VERSION:
        raise SystemExit(
            f"\n'{path}' was NOT made by train_landmark_v2.py (it has the old 63-number input).\n"
            "Run train_landmark_v2.py first, then use sign_landmark_dataset_c_v2.pt.\n")
    model = LandmarkMLP(len(ck["classes"])).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ck["classes"]
