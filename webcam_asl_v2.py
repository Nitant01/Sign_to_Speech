"""
webcam_asl_v2.py  -  live ASL typing with MediaPipe + your trained model.

Run:            python webcam_asl_v2.py
Voice check:    python webcam_asl_v2.py --test-voice
Silent mode:    python webcam_asl_v2.py --no-voice

Hold a letter sign still for HOLD_SECONDS and it is typed.
Hold the SPACE sign  -> a space is added AND the word you just finished is spoken aloud.
Keys:  q = quit | c = clear text | b = backspace | SPACE bar = same as the space sign | d = show/hide top-3 panel
"""
import argparse, os, queue, shutil, subprocess, sys, threading, time
from collections import deque

import cv2
import numpy as np
import torch

from asl_common import features_from_raw, landmarks_to_raw, load_checkpoint

# ====================== SETTINGS ======================
TASK_PATH = "hand_landmarker.task"
MODEL = "sign_landmark_dataset_c_v2.pt"      # NEW model file (never the old sign_landmark*.pt)
CAMERA = 0

CONF = 0.50            # the AVERAGED probability of a sign must reach this (was: each single frame)
SMOOTH_FRAMES = 8      # average the model output over the last 8 frames (was: 10-frame hard vote needing 7/10)
MIN_FRAMES = 3         # need at least this many frames before trusting the average
HOLD_SECONDS = 0.7     # how long a sign must stay the same to be typed (was: 20 frames, which resets on any flicker)
GRACE_SECONDS = 0.25   # tracking may lose the hand for this long without cancelling the hold
RELEASE_SECONDS = 0.4  # after typing, the sign must change / hand must leave this long before the same letter can repeat
SPEAK_WORDS = True
# ======================================================

# colours are BGR
GREEN, BLUE, RED, YELLOW, WHITE, GREY = (0, 200, 0), (255, 140, 0), (0, 0, 220), (0, 200, 255), (255, 255, 255), (170, 170, 170)
FONT = cv2.FONT_HERSHEY_SIMPLEX
CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
               (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]


# ---------------- speech ----------------
class Speaker:
    """Speaks words in a background thread so the video never freezes. Words are spoken one after another.
    Windows: uses the built-in Windows voice through PowerShell (nothing to pip install)."""

    def __init__(self, enabled=True):
        self.enabled, self.backend = enabled, self._pick_backend()
        self.q = queue.Queue()
        self.speaking = ""          # text being spoken right now (shown on screen)
        self.error = ""
        if self.enabled and self.backend is None:
            print("WARNING: no text-to-speech voice found on this computer -> running without voice.")
            self.enabled = False
        if self.enabled:
            threading.Thread(target=self._worker, daemon=True).start()

    @staticmethod
    def _pick_backend():
        if sys.platform.startswith("win"):
            return "powershell"
        if sys.platform == "darwin" and shutil.which("say"):
            return "say"
        for exe in ("espeak-ng", "espeak"):
            if shutil.which(exe):
                return exe
        return None

    def _command(self, text):
        if self.backend == "powershell":
            script = ("Add-Type -AssemblyName System.Speech; "
                      "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                      "$s.Speak($env:ASL_SPEAK_TEXT)")
            return ["powershell", "-NoProfile", "-NonInteractive", "-Command", script], {"ASL_SPEAK_TEXT": text}
        return [self.backend, text], {}

    def _worker(self):
        while True:
            text = self.q.get()
            self.speaking = text
            try:
                cmd, extra = self._command(text)
                flags = subprocess.CREATE_NO_WINDOW if sys.platform.startswith("win") else 0
                r = subprocess.run(cmd, env={**os.environ, **extra}, stdout=subprocess.DEVNULL,
                                   stderr=subprocess.PIPE, text=True, timeout=60, creationflags=flags)
                if r.returncode != 0:
                    self.error = (r.stderr or "voice command failed").strip().splitlines()[0][:80]
                    print("VOICE ERROR:", self.error)
            except Exception as e:                       # never let a voice problem crash the camera loop
                self.error = str(e)[:80]
                print("VOICE ERROR:", e)
            self.speaking = ""
            self.q.task_done()

    def say(self, word):
        if self.enabled and word:
            # single letters stay capital ("I", "A") so they are read as letters, words are lower-case
            self.q.put(word if len(word) == 1 else word.lower())

    def wait_until_done(self):
        self.q.join()


# ---------------- typing logic (no camera, no drawing -> easy to test) ----------------
class Typer:
    def __init__(self, classes, speaker):
        self.classes, self.speaker = classes, speaker
        self.text = ""
        self.window = deque(maxlen=SMOOTH_FRAMES)
        self.cand, self.cand_since = None, 0.0
        self.locked, self.diff_since = None, None
        self.last_hand = -1e9
        self.mean = None            # averaged probabilities (for the top-3 panel)
        self.progress = 0.0
        self.flash_msg, self.flash_sub, self.flash_color, self.flash_until = "", "", GREEN, 0.0
        self.border_until = 0.0

    # ----- feedback -----
    def flash(self, msg, color, seconds, now, sub=""):
        self.flash_msg, self.flash_sub, self.flash_color, self.flash_until = msg, sub, color, now + seconds

    # ----- actions -----
    def commit_space(self, now):
        word = self.text.rsplit(" ", 1)[-1]                 # everything after the last space
        if not word:
            self.flash("SPACE ignored", YELLOW, 1.2, now, sub="no new word to speak yet")
            return
        self.text += " "
        self.speaker.say(word)
        self.border_until = now + 0.6
        self.flash("SPACE ADDED", BLUE, 1.6, now, sub=f'speaking: "{word}"')

    def commit(self, label, now):
        name = label.lower()
        if name == "space":
            self.commit_space(now)
        elif name in ("del", "delete"):
            self.text = self.text[:-1]
            self.flash("DELETE", RED, 0.7, now)
        else:
            self.text += label
            self.flash(f"+ {label}", GREEN, 0.6, now)

    # ----- called once per camera frame; probs = model output (numpy) or None when no hand -----
    def update(self, now, probs):
        if probs is not None:
            self.last_hand = now
            self.window.append(probs)
        elif now - self.last_hand > GRACE_SECONDS:
            self.window.clear()

        cand, self.mean = None, None
        if len(self.window) >= MIN_FRAMES:
            self.mean = np.mean(self.window, axis=0)
            k = int(self.mean.argmax())
            if self.mean[k] >= CONF:
                cand = self.classes[k]
        if cand != self.cand:
            self.cand, self.cand_since = cand, now

        if self.locked is not None:                          # re-arm only after the sign changed / hand left
            if cand == self.locked:
                self.diff_since = None
            else:
                self.diff_since = self.diff_since if self.diff_since is not None else now
                if now - self.diff_since >= RELEASE_SECONDS:
                    self.locked, self.diff_since = None, None

        self.progress = 0.0
        if cand is not None:
            if cand == self.locked:
                self.progress = 1.0
            else:
                self.progress = min(1.0, (now - self.cand_since) / HOLD_SECONDS)
                if self.progress >= 1.0:
                    self.commit(cand, now)
                    self.locked, self.diff_since = cand, None


# ---------------- drawing ----------------
def draw_text_line(img, text, x, y, max_w, scale, now):
    """Draws the typed text. Spaces are drawn as a visible cyan bracket and a blinking cursor shows where the next
    letter goes, so you can always SEE whether a space was added."""
    th = 2
    m_w = cv2.getTextSize("M", FONT, scale, th)[0][0]
    sp_w = int(m_w * 0.8)
    width = lambda ch: sp_w if ch == " " else cv2.getTextSize(ch, FONT, scale, th)[0][0] + 2
    prefix = ""
    while text and sum(width(c) for c in text) + sp_w > max_w:     # keep only the end that fits
        text, prefix = text[1:], "..."
    if prefix:
        cv2.putText(img, prefix, (x, y), FONT, scale, GREY, th)
        x += cv2.getTextSize(prefix, FONT, scale, th)[0][0] + 4
    h = cv2.getTextSize("M", FONT, scale, th)[0][1]
    for ch in text:
        w = width(ch)
        if ch == " ":
            c = (255, 255, 0)
            cv2.line(img, (x + 3, y - 8), (x + 3, y), c, 2); cv2.line(img, (x + 3, y), (x + w - 3, y), c, 2)
            cv2.line(img, (x + w - 3, y), (x + w - 3, y - 8), c, 2)
        else:
            cv2.putText(img, ch, (x, y), FONT, scale, WHITE, th)
        x += w
    if int(now * 2) % 2 == 0:                                       # blinking cursor
        cv2.line(img, (x + 3, y - h - 2), (x + 3, y + 4), (0, 255, 255), 3)


def draw_hud(frame, t, now, show_top3=True):
    h, w = frame.shape[:2]
    s = w / 640.0
    # --- bottom text bar ---
    bar = int(95 * s)
    cv2.rectangle(frame, (0, h - bar), (w, h), (0, 0, 0), -1)
    draw_text_line(frame, t.text, 12, h - int(48 * s), w - 24, 1.0 * s, now)
    hint = "hold a sign to type | SPACE sign = space + speak word | c clear | b back | q quit"
    cv2.putText(frame, hint, (12, h - int(14 * s)), FONT, 0.45 * s, GREY, 1)
    if t.speaker.speaking:
        cv2.putText(frame, f'SPEAKING: "{t.speaker.speaking}"', (12, h - bar - 10), FONT, 0.6 * s, BLUE, 2)
    elif t.speaker.error:
        cv2.putText(frame, f"VOICE PROBLEM: {t.speaker.error}", (12, h - bar - 10), FONT, 0.5 * s, RED, 1)

    # --- hold progress (top-left) ---
    if t.cand is not None:
        is_space = t.cand.lower() == "space"
        col = BLUE if is_space else (RED if t.cand.lower() == "del" else GREEN)
        name = "SPACE" if is_space else ("DELETE" if t.cand.lower() == "del" else t.cand)
        done = t.progress >= 1.0
        cv2.putText(frame, f"{name}  {'typed - lift hand or change sign' if done else 'keep holding...'}",
                    (12, int(32 * s)), FONT, 0.7 * s, col, 2)
        bw, bh = int(260 * s), int(16 * s)
        cv2.rectangle(frame, (12, int(44 * s)), (12 + bw, int(44 * s) + bh), (60, 60, 60), -1)
        cv2.rectangle(frame, (12, int(44 * s)), (12 + int(bw * t.progress), int(44 * s) + bh), col, -1)

    # --- top-3 panel (top-right) ---
    if show_top3 and t.mean is not None:
        order = np.argsort(t.mean)[::-1][:3]
        for r, k in enumerate(order):
            cv2.putText(frame, f"{t.classes[k]:>5} {t.mean[k] * 100:3.0f}%", (w - int(150 * s), int((28 + 24 * r) * s)),
                        FONT, 0.6 * s, WHITE if r else (0, 255, 255), 2 if r == 0 else 1)

    # --- flash banner ---
    if now < t.flash_until:
        bw, bh = int(380 * s), int((90 if t.flash_sub else 60) * s)
        x0, y0 = (w - bw) // 2, int(95 * s)
        cv2.rectangle(frame, (x0, y0), (x0 + bw, y0 + bh), t.flash_color, -1)
        tw = cv2.getTextSize(t.flash_msg, FONT, 1.1 * s, 3)[0][0]
        cv2.putText(frame, t.flash_msg, (x0 + (bw - tw) // 2, y0 + int(40 * s)), FONT, 1.1 * s, WHITE, 3)
        if t.flash_sub:
            tw = cv2.getTextSize(t.flash_sub, FONT, 0.6 * s, 2)[0][0]
            cv2.putText(frame, t.flash_sub, (x0 + (bw - tw) // 2, y0 + int(72 * s)), FONT, 0.6 * s, WHITE, 2)
    if now < t.border_until:                                       # blue frame around the whole picture
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), BLUE, int(10 * s))


# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-voice", action="store_true", help="speak a test sentence and exit")
    ap.add_argument("--no-voice", action="store_true", help="run without spoken words")
    args = ap.parse_args()

    speaker = Speaker(enabled=SPEAK_WORDS and not args.no_voice)
    if args.test_voice:
        if not speaker.enabled:
            raise SystemExit("No voice available on this computer.")
        print("Speaking a test sentence ... you should HEAR it now.")
        speaker.q.put("Voice check. If you can hear this, spoken words will work.")
        speaker.wait_until_done()
        print(("Voice problem: " + speaker.error) if speaker.error else "Finished. If you heard it, voice works.")
        return

    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    device = "cpu"
    model, classes = load_checkpoint(MODEL, device)
    print(f"Loaded {MODEL} with {len(classes)} classes: {classes}")

    landmarker = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=TASK_PATH),
        running_mode=vision.RunningMode.VIDEO, num_hands=1,
        min_hand_detection_confidence=0.5, min_hand_presence_confidence=0.5, min_tracking_confidence=0.5))

    cap = cv2.VideoCapture(CAMERA)
    if not cap.isOpened():
        raise SystemExit("Cannot open camera. Try CAMERA = 1")

    typer = Typer(classes, speaker)
    show_top3 = True
    t0, last_ts = time.time(), -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)       # mirror view; landmarks are taken from this mirrored picture
        h, w = frame.shape[:2]
        now = time.time() - t0
        ts = max(int(now * 1000), last_ts + 1); last_ts = ts
        res = landmarker.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), ts)

        probs = None
        if res.hand_landmarks:
            hl = res.hand_landmarks[0]
            with torch.no_grad():
                probs = torch.softmax(model(features_from_raw(landmarks_to_raw(hl, w, h))), 1)[0].numpy()
            px = [(int(p.x * w), int(p.y * h)) for p in hl]
            for a, b in CONNECTIONS:
                cv2.line(frame, px[a], px[b], (0, 255, 0), 2)
            for p in px:
                cv2.circle(frame, p, 4, (0, 0, 255), -1)
            k = int(probs.argmax())
            cv2.putText(frame, f"{classes[k]} ({probs[k] * 100:.0f}%)", (px[0][0], max(px[0][1] - 20, 30)),
                        FONT, 0.9, (0, 255, 0), 2)
        typer.update(now, probs)
        draw_hud(frame, typer, now, show_top3)
        cv2.imshow("Sign Language (MediaPipe)", frame)

        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        elif k == ord("c"):
            typer.text = ""
        elif k == ord("b"):
            typer.text = typer.text[:-1]
        elif k == ord("d"):
            show_top3 = not show_top3
        elif k == 32:
            typer.commit_space(now)

    cap.release(); cv2.destroyAllWindows(); landmarker.close()
    speaker.wait_until_done()


if __name__ == "__main__":
    main()
