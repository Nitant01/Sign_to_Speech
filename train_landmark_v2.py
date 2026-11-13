"""
train_landmark_v2.py  -  trains the ASL landmark classifier (28 classes: A-Z, del, space; "nothing" excluded).

Run from your project folder:   python train_landmark_v2.py
Writes ONLY new files (your old ones are never touched):
    dataset_c_landmarks_raw_cache_v2.npz   <- landmark cache
    sign_landmark_dataset_c_v2.pt          <- trained model
    confusion_matrix_v2.csv                <- full confusion matrix of the honest validation run

What changed compared with train_landmark.py (details are in the explanation):
  1. Honest validation: images that come from the same recording burst stay on the SAME side of the split.
  2. Palm-size normalisation + 25 distance features (see asl_common.py).
  3. Augmentation adds camera-tilt (3D rotation) and depth jitter; mirroring is kept (explained below).
  4. Hand detection retries on a bigger copy of the image, so fewer images are lost (I, del, space, ...).
  5. Optional webcam samples (made by collect_webcam_samples.py) are mixed in automatically.
  6. Prints per-class precision / recall / F1 and the confusion matrix.
"""
import argparse, math, os
from pathlib import Path
import numpy as np, cv2, torch, torch.nn as nn
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from asl_common import (FEATURE_VERSION, LandmarkMLP, N_FEATURES, add_features,
                        landmarks_to_raw, normalize_np)

# ====================== SETTINGS ======================
DATA_DIR = r"D:\Nitant\Sem V\DL\Capstone\Dataset C"
TASK_PATH = "hand_landmarker.task"
CACHE = "dataset_c_landmarks_raw_cache_v2.npz"      # NEW name (old: dataset_c_landmarks_cache.npz)
WEBCAM_SAMPLES = "webcam_samples_c.npz"             # created by collect_webcam_samples.py (optional)
OUT = "sign_landmark_dataset_c_v2.pt"               # NEW name (old: sign_landmark_dataset_c.pt)
CONFUSION_CSV = "confusion_matrix_v2.csv"

EXCLUDE_CLASSES = {"nothing"}
EPOCHS, BATCH, LR = 60, 32, 3e-3                    # unchanged from your run
VAL_FRACTION = 0.20      # share of each letter held back for the honest check
BLOCK = 5                # 5 consecutive file names = one recording burst = never split apart
SEED = 42
FINAL_ON_ALL_DATA = True # after the honest check, retrain on 100% of the data for the saved model

# Letters whose DATASET images are ignored once you have recorded your own webcam samples for them.
# Reason: the dataset's "T" images show the thumb sticking out sideways, which is not the standard ASL T
# (thumb tucked between index and middle finger). Delete "T" from this set to keep the dataset T as well.
PREFER_WEBCAM = {"T"}
MIN_WEBCAM_FOR_REPLACE = 30

# Augmentation strengths (each one explained in the answer)
ROLL_DEG = 15     # in-plane rotation (same as before)
TILT_DEG = 20     # NEW: tilt towards/away from the camera (simulates a different camera angle)
SCALE_JIT = 0.10  # +-10 % size (same as before)
Z_JIT = 0.30      # NEW: +-30 % on the depth numbers (MediaPipe's z is the noisiest value)
NOISE = 0.01      # same as before
MIRROR_PROB = 0.5 # kept, see explanation
# ======================================================


# ---------- 1) landmark extraction ----------
def make_landmarker(conf):
    return vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=TASK_PATH),
        running_mode=vision.RunningMode.IMAGE, num_hands=1,
        min_hand_detection_confidence=conf, min_hand_presence_confidence=conf,
        min_tracking_confidence=conf))


def _detect(lm, bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)).hand_landmarks


def extract(lm_strict, lm_loose, path):
    """Returns (21,3) raw landmarks or None. Tries: normal -> lower confidence -> 2x bigger padded copy."""
    img = cv2.imread(str(path))
    if img is None:
        return None
    h, w = img.shape[:2]
    for lm in (lm_strict, lm_loose):
        hl = _detect(lm, img)
        if hl:
            return landmarks_to_raw(hl[0], w, h)
    PAD = 100
    big = cv2.copyMakeBorder(cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC),
                             PAD, PAD, PAD, PAD, cv2.BORDER_REPLICATE)
    H2, W2 = big.shape[:2]
    hl = _detect(lm_loose, big)
    if hl:
        raw = landmarks_to_raw(hl[0], W2, H2)
        raw[:, :2] = (raw[:, :2] - PAD) / 2       # back to the original image's pixel units
        raw[:, 2] = raw[:, 2] / 2
        return raw
    return None


def build_dataset_cache(data_dir, classes):
    if os.path.exists(CACHE):
        z = np.load(CACHE, allow_pickle=True)
        if list(z["classes"]) == classes and str(z["version"]) == "extract-v2":
            print("Loaded cached landmarks:", z["P"].shape)
            return z["P"], z["y"], z["group"], z["names"]
        print("Cache does not match -> rebuilding")
    if not os.path.exists(TASK_PATH):
        raise SystemExit(f"Cannot find {TASK_PATH} in this folder.")
    strict, loose = make_landmarker(0.3), make_landmarker(0.1)
    P, y, group, names = [], [], [], []
    for ci, c in enumerate(classes):
        files = sorted(f for f in (Path(data_dir) / c).iterdir()
                       if f.suffix.lower() in (".jpg", ".jpeg", ".png"))   # sorted = recording order
        ok = 0
        for k, f in enumerate(files):
            raw = extract(strict, loose, f)
            if raw is None:
                continue
            P.append(raw); y.append(ci); group.append(ci * 1000 + k // BLOCK); names.append(f"{c}/{f.name}")
            ok += 1
        print(f"{c:>6}: hand found in {ok}/{len(files)} images")
    P, y, group, names = np.stack(P), np.array(y), np.array(group), np.array(names)
    np.savez(CACHE, P=P, y=y, group=group, names=names, classes=np.array(classes), version="extract-v2")
    print("Saved cache:", P.shape)
    return P, y, group, names


def load_webcam_samples(classes):
    if not os.path.exists(WEBCAM_SAMPLES):
        return np.zeros((0, 21, 3), np.float32), np.zeros(0, int), np.zeros(0, int)
    z = np.load(WEBCAM_SAMPLES, allow_pickle=True)
    P, y, group = [], [], []
    for raw, lab, take in zip(z["P"], z["labels"], z["take"]):
        lab = str(lab)
        if lab not in classes:
            continue
        ci = classes.index(lab)
        P.append(raw); y.append(ci); group.append(10 ** 6 + ci * 1000 + int(take))
    return np.array(P, np.float32).reshape(-1, 21, 3), np.array(y, int), np.array(group, int)


# ---------- 2) honest split ----------
def group_split(y, group, frac, seed):
    """Whole groups (recording bursts / webcam takes) go to validation, never single frames."""
    rng = np.random.RandomState(seed)
    tr, va = [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        gs = np.unique(group[idx]); rng.shuffle(gs)
        want, got, chosen = max(1, round(len(idx) * frac)), 0, []
        for g in gs[:-1]:                         # the last group always stays in training
            if got >= want:
                break
            chosen.append(g); got += int((group[idx] == g).sum())
        m = np.isin(group[idx], chosen)
        va += idx[m].tolist(); tr += idx[~m].tolist()
    return np.array(tr), np.array(va)


# ---------- 3) augmentation ----------
def _rot(ax, ay, az):
    B = ax.shape[0]; o, z = torch.ones(B), torch.zeros(B)
    cx, sx, cy, sy, cz, sz = ax.cos(), ax.sin(), ay.cos(), ay.sin(), az.cos(), az.sin()
    Rx = torch.stack([torch.stack([o, z, z], 1), torch.stack([z, cx, -sx], 1), torch.stack([z, sx, cx], 1)], 1)
    Ry = torch.stack([torch.stack([cy, z, sy], 1), torch.stack([z, o, z], 1), torch.stack([-sy, z, cy], 1)], 1)
    Rz = torch.stack([torch.stack([cz, -sz, z], 1), torch.stack([sz, cz, z], 1), torch.stack([z, z, o], 1)], 1)
    return Rz @ Ry @ Rx


def augment(P):
    """P: (B,21,3) normalized landmarks -> (B, N_FEATURES). Every change keeps the letter the same letter."""
    B = P.shape[0]
    u = lambda deg: (torch.rand(B) * 2 - 1) * math.radians(deg)
    P = P @ _rot(u(TILT_DEG), u(TILT_DEG), u(ROLL_DEG)).transpose(1, 2)
    sign = (torch.rand(B) < MIRROR_PROB).float() * -2 + 1            # +1 or -1 (mirror = other hand)
    P = P * torch.stack([sign, torch.ones(B), torch.ones(B)], 1).unsqueeze(1)
    P = P * (1 + SCALE_JIT * (torch.rand(B, 1, 1) * 2 - 1))
    zs = 1 + Z_JIT * (torch.rand(B, 1, 1) * 2 - 1)
    P = torch.cat([P[:, :, :2], P[:, :, 2:] * zs], 2)
    return add_features(P + NOISE * torch.randn_like(P))


# ---------- 4) training ----------
def fit(P, y, n_classes, epochs, seed):
    torch.manual_seed(seed)
    model = LandmarkMLP(n_classes)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    steps = math.ceil(len(P) / BATCH)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, epochs=epochs, steps_per_epoch=steps)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
    for ep in range(1, epochs + 1):
        model.train()
        perm, tot = torch.randperm(len(P)), 0.0
        for i in range(0, len(P), BATCH):
            idx = perm[i:i + BATCH]
            if len(idx) < 2:
                continue
            opt.zero_grad()
            loss = loss_fn(model(augment(P[idx])), y[idx])
            loss.backward(); opt.step(); sched.step()
            tot += loss.item() * len(idx)
        if ep == 1 or ep % 10 == 0:
            print(f"  epoch {ep:02d}/{epochs}  loss {tot / len(P):.4f}")
    model.eval()
    return model


def predict(model, P):
    with torch.no_grad():
        return model(add_features(P)).argmax(1).numpy()


# ---------- 5) report ----------
def report(y_true, y_pred, classes):
    K = len(classes)
    cm = np.zeros((K, K), int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    print(f"\n{'class':>6} {'support':>7} {'precision':>9} {'recall':>7} {'F1':>6}")
    f1s = []
    for k, c in enumerate(classes):
        sup = cm[k].sum()
        if sup == 0:
            print(f"{c:>6} {0:>7}      (no validation images)"); continue
        prec = cm[k, k] / max(1, cm[:, k].sum()); rec = cm[k, k] / sup
        f1 = 2 * prec * rec / max(1e-9, prec + rec); f1s.append(f1)
        flag = "   <-- focus letter" if c in "DLMNST" and len(c) == 1 else ""
        print(f"{c:>6} {sup:>7} {prec:>9.2f} {rec:>7.2f} {f1:>6.2f}{flag}")
    print(f"\nHONEST validation accuracy: {np.trace(cm) / cm.sum():.3f}   macro-F1: {np.mean(f1s):.3f}   ({cm.sum()} images)")
    errs = sorted(((cm[t, p], classes[t], classes[p]) for t in range(K) for p in range(K) if t != p and cm[t, p]),
                  reverse=True)[:10]
    print("Most common mistakes (true -> predicted : count):", ", ".join(f"{a}->{b}:{n}" for n, a, b in errs))
    for a, b in [("D", "L"), ("L", "D"), ("M", "S"), ("N", "S"), ("T", "S"), ("M", "N"), ("N", "M")]:
        if a in classes and b in classes:
            print(f"  true {a} predicted as {b}: {cm[classes.index(a), classes.index(b)]}")
    with open(CONFUSION_CSV, "w") as f:
        f.write("true\\pred," + ",".join(classes) + "\n")
        for k, c in enumerate(classes):
            f.write(c + "," + ",".join(map(str, cm[k])) + "\n")
    print(f"Full confusion matrix saved to {CONFUSION_CSV} (open it in Excel).")


# ---------- main ----------
def main():
    global DATA_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA_DIR, help="Dataset C folder")
    DATA_DIR = ap.parse_args().data

    classes = sorted(d.name for d in Path(DATA_DIR).iterdir()
                     if d.is_dir() and d.name.lower() not in EXCLUDE_CLASSES)
    print(f"{len(classes)} classes: {classes}")
    assert len(classes) == 28, "expected 28 classes (A-Z, del, space)"

    P, y, group, names = build_dataset_cache(DATA_DIR, classes)
    Pw, yw, gw = load_webcam_samples(classes)
    if len(Pw):
        for c in sorted(PREFER_WEBCAM):
            if c in classes and (yw == classes.index(c)).sum() >= MIN_WEBCAM_FOR_REPLACE:
                keep = y != classes.index(c)
                print(f"Using your webcam samples for '{c}' instead of the {int((~keep).sum())} dataset images.")
                P, y, group = P[keep], y[keep], group[keep]
        P, y, group = np.concatenate([P, Pw]), np.concatenate([y, yw]), np.concatenate([group, gw])
        print(f"Added {len(Pw)} webcam samples from {WEBCAM_SAMPLES}")
    print(f"Total training material: {len(P)} samples")

    Pn = torch.tensor(np.stack([normalize_np(p) for p in P]))
    yt = torch.tensor(y)

    # --- honest check ---
    tr, va = group_split(y, group, VAL_FRACTION, SEED)
    print(f"\nHonest split: {len(tr)} training / {len(va)} validation samples "
          f"(validation comes from different recording bursts than training)")
    model = fit(Pn[tr], yt[tr], len(classes), EPOCHS, SEED)
    report(y[va], predict(model, Pn[va]), classes)

    # --- final model ---
    if FINAL_ON_ALL_DATA:
        print(f"\nRetraining on all {len(P)} samples for the saved model ...")
        model = fit(Pn, yt, len(classes), EPOCHS, SEED)
    torch.save({"model": model.state_dict(), "classes": classes,
                "feature_version": FEATURE_VERSION, "n_features": N_FEATURES}, OUT)
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
