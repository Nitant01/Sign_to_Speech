"""
collect_webcam_samples.py  -  record hand landmarks from YOUR webcam for the letters the model gets wrong.

WHY: Dataset C pictures were taken from odd angles (often from below the hand) and its "T" is not the standard
ASL T. Your webcam sees your hand differently, so the model has never seen your D, M, N or T. Recording ~90
samples of each problem letter from your own camera is the most effective fix.

Run (one or several letters):     python collect_webcam_samples.py D L M N S T
Result is appended to:            webcam_samples_c.npz      (only landmark numbers are stored, no pictures)
Then re-run:                      python train_landmark_v2.py   (it picks the samples up automatically)

For each letter you do 3 "takes" of 6 seconds. Press SPACE to start a take, then you have 3 seconds to get your
hand into position. While recording, keep making the correct sign but slowly move the hand a little: closer /
further, tilt left / right, turn slightly. Keys: SPACE start | s skip this letter | q quit
"""
import argparse, os, sys, time
import cv2
import numpy as np

from asl_common import landmarks_to_raw

# ====================== SETTINGS ======================
TASK_PATH = "hand_landmarker.task"
OUT = "webcam_samples_c.npz"
CAMERA = 0
TAKES = 3            # separate takes per letter (the trainer keeps whole takes together when it validates)
TAKE_SECONDS = 6
SAMPLE_EVERY = 0.2   # one sample every 0.2 s -> about 30 samples per take
COUNTDOWN = 3
# ======================================================
FONT = cv2.FONT_HERSHEY_SIMPLEX
ALL = [chr(c) for c in range(65, 91)] + ["del", "space"]


def next_take_id(label):
    if not os.path.exists(OUT):
        return 1
    z = np.load(OUT, allow_pickle=True)
    t = [int(x) for x, l in zip(z["take"], z["labels"]) if str(l) == label]
    return max(t, default=0) + 1


def save(new_P, label, take):
    labels, takes = np.array([label] * len(new_P)), np.full(len(new_P), take, np.int32)
    P = np.array(new_P, np.float32)
    if os.path.exists(OUT):
        z = np.load(OUT, allow_pickle=True)
        P = np.concatenate([z["P"], P]); labels = np.concatenate([z["labels"], labels]); takes = np.concatenate([z["take"], takes])
    tmp = OUT.replace(".npz", ".tmp.npz")
    np.savez(tmp, P=P, labels=labels, take=takes)
    os.replace(tmp, OUT)                       # saved after every take, so quitting early loses nothing


def put(frame, text, y, scale=0.8, color=(255, 255, 255)):
    cv2.putText(frame, text, (15, y), FONT, scale, (0, 0, 0), 4)
    cv2.putText(frame, text, (15, y), FONT, scale, color, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("letters", nargs="+", help="e.g. D L M N S T   (also: del space)")
    letters = [a.upper() if len(a) == 1 else a.lower() for a in ap.parse_args().letters]
    bad = [l for l in letters if l not in ALL]
    if bad:
        raise SystemExit(f"Unknown letters: {bad}. Use A-Z, del, space.")

    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    lm = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=TASK_PATH),
        running_mode=vision.RunningMode.VIDEO, num_hands=1,
        min_hand_detection_confidence=0.5, min_hand_presence_confidence=0.5, min_tracking_confidence=0.5))
    cap = cv2.VideoCapture(CAMERA)
    if not cap.isOpened():
        raise SystemExit("Cannot open camera. Try CAMERA = 1")

    t0, last_ts = time.time(), -1

    def read():
        nonlocal last_ts
        ok, frame = cap.read()
        if not ok:
            raise SystemExit("Camera stopped.")
        frame = cv2.flip(frame, 1)                       # SAME mirroring as webcam_asl_v2.py
        h, w = frame.shape[:2]
        ts = max(int((time.time() - t0) * 1000), last_ts + 1); last_ts = ts
        res = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), ts)
        raw = landmarks_to_raw(res.hand_landmarks[0], w, h) if res.hand_landmarks else None
        if raw is not None:
            for x, y, _ in raw:
                cv2.circle(frame, (int(x), int(y)), 4, (0, 0, 255), -1)
        return frame, raw

    for letter in letters:
        skip = False
        for take_no in range(1, TAKES + 1):
            if skip:
                break
            # ---- wait for SPACE ----
            while True:
                frame, _ = read()
                put(frame, f"Letter {letter}   take {take_no}/{TAKES}", 35, 1.0, (0, 255, 255))
                put(frame, "Press SPACE to start (then you get 3 s to get ready)", 70, 0.65)
                put(frame, "s = skip letter    q = quit", 100, 0.55, (200, 200, 200))
                cv2.imshow("Collect samples", frame)
                k = cv2.waitKey(1) & 0xFF
                if k == 32: break
                if k == ord("s"): skip = True; break
                if k == ord("q"): cap.release(); cv2.destroyAllWindows(); return
            if skip:
                break
            # ---- countdown ----
            start = time.time()
            while time.time() - start < COUNTDOWN:
                frame, _ = read()
                put(frame, f"Get ready: {COUNTDOWN - int(time.time() - start)}   sign = {letter}", 45, 1.1, (0, 255, 255))
                cv2.imshow("Collect samples", frame); cv2.waitKey(1)
            # ---- record ----
            samples, start, next_at = [], time.time(), 0.0
            while time.time() - start < TAKE_SECONDS:
                frame, raw = read()
                el = time.time() - start
                if raw is not None and el >= next_at:
                    samples.append(raw); next_at = el + SAMPLE_EVERY
                put(frame, f"RECORDING {letter}   {len(samples)} samples", 40, 1.0, (0, 0, 255))
                put(frame, "keep the sign, move the hand slowly: closer/further, tilt, turn" if raw is not None
                    else "NO HAND FOUND - show your hand", 75, 0.6, (255, 255, 255) if raw is not None else (0, 200, 255))
                cv2.rectangle(frame, (15, 90), (15 + int(300 * el / TAKE_SECONDS), 104), (0, 0, 255), -1)
                cv2.imshow("Collect samples", frame); cv2.waitKey(1)
            if len(samples) >= 5:
                save(samples, letter, next_take_id(letter))
                print(f"Saved {len(samples)} samples for {letter} (take {take_no}/{TAKES})")
            else:
                print(f"Only {len(samples)} samples for {letter} - take not saved. Keep your hand in view and retry.")

    cap.release(); cv2.destroyAllWindows(); lm.close()
    if os.path.exists(OUT):
        z = np.load(OUT, allow_pickle=True)
        print("\nSamples now in", OUT)
        for l in sorted(set(map(str, z["labels"]))):
            print(f"  {l:>6}: {int((z['labels'] == l).sum())}")


if __name__ == "__main__":
    main()
