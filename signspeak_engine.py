"""
signspeak_engine.py  -  adapter between the SignSpeak window and your EXISTING recognition code.

Nothing in your four original files is edited. This file only IMPORTS them:
    asl_common.py     -> load_checkpoint, features_from_raw, landmarks_to_raw   (model + preprocessing)
    webcam_asl_v2.py  -> Typer (hold / smoothing / space / delete logic), Speaker (spoken words),
                         and its settings (MODEL, TASK_PATH, CONF, HOLD_SECONDS ...)

Why a separate loop at all? webcam_asl_v2.main() is one big function that owns the camera and its own OpenCV window,
so it cannot be embedded in another window. The loop below does exactly what main() does for each frame
(mirror -> MediaPipe -> existing model -> Typer.update) and hands the picture + state to the UI instead of
cv2.imshow. It is the ONLY place that touches the camera, MediaPipe and the model, so nothing is duplicated.
"""
import sys
import threading
import time
from collections import deque

import cv2
import numpy as np
import torch

import webcam_asl_v2 as W                                   # your existing recognition code (unmodified)
from asl_common import features_from_raw, landmarks_to_raw, load_checkpoint


# ======================= SETTINGS (tune these after testing) =======================
# --- long-hold acceptance of moderately low-confidence signs (used only by this window) ---
WEAK_ENABLED = True
WEAK_MIN_CONF = 0.30          # below this averaged confidence a sign is NEVER accepted
WEAK_MAX_HOLD_SECONDS = 2.5   # hold needed when confidence is right at WEAK_MIN_CONF
                              # (at the normal CONF the hold is just HOLD_SECONDS; in between it grows linearly)
WEAK_MIN_MARGIN = 0.10        # the winning letter must beat the runner-up by at least this much
WEAK_MIN_FRAMES = 15          # ... and be the winner for this many consecutive camera frames
WEAK_ALLOW_SPECIAL = False    # False: the space and del signs are never accepted through this path
# --- camera start-up ---
CAMERA_BACKEND = "dshow"      # Windows only: "dshow" opens much faster than OpenCV's default; "default" = old behaviour
CAMERA_WIDTH, CAMERA_HEIGHT = 640, 480    # ask the camera for 640x480 (0 = leave the camera's own setting)
CAMERA_BACKEND_LINUX = "v4l2" # Linux / Jetson only: "v4l2" talks to /dev/video* directly (OpenCV's default may try GStreamer first); "default" = old behaviour
CAMERA_FOURCC_LINUX = "MJPG"  # Linux / Jetson only: USB webcams give 640x480 at 30 fps more reliably as MJPG ("" = leave the camera's own format)
# --- voice (Linux / Jetson, espeak-ng only; Windows keeps using its own voice) ---
VOICE = "en-us"               # e.g. "en-us", "en-gb", "en-us+f3" (female variant), "en-us+m3" (male variant). List them: espeak-ng --voices=en
VOICE_SPEED = 150             # words per minute; espeak-ng's default is 175, lower = slower and clearer
VOICE_PITCH = 50              # 0-99; 50 is the default, higher = higher voice
# ====================================================================================


_orig_speaker_command = W.Speaker._command


def _speaker_command(self, text):
    """Adds voice / speed / pitch for espeak. Your webcam_asl_v2.py is not edited: this only wraps its Speaker at run time."""
    if self.backend in ("espeak-ng", "espeak"):
        return [self.backend, "-v", VOICE, "-s", str(VOICE_SPEED), "-p", str(VOICE_PITCH), text], {}
    return _orig_speaker_command(self, text)


W.Speaker._command = _speaker_command


def open_camera(index):
    """Opens the camera as quickly as possible. DirectShow avoids OpenCV's slow default (MSMF) start-up on Windows;
    if it cannot open the camera we fall back to the default backend, so nothing that worked before stops working."""
    cap = None
    if sys.platform.startswith("win") and CAMERA_BACKEND == "dshow":
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            cap = None
    linux = sys.platform.startswith("linux")
    if cap is None and linux and CAMERA_BACKEND_LINUX == "v4l2":      # Jetson / Linux: open /dev/video<index> directly
        cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            cap = None
    if cap is None:
        cap = cv2.VideoCapture(index)
    if cap.isOpened():
        if linux and CAMERA_FOURCC_LINUX:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*CAMERA_FOURCC_LINUX))   # must be set before the size
        if CAMERA_WIDTH and CAMERA_HEIGHT:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)          # always the newest frame, no queue of old ones
    return cap


def _hex(bgr):
    b, g, r = bgr
    return f"#{r:02x}{g:02x}{b:02x}"


class UITyper(W.Typer):
    """Your Typer, plus a log of what was typed by gestures (for the History tab). Behaviour is unchanged."""

    def __init__(self, classes, speaker, log):
        super().__init__(classes, speaker)
        self._log = log
        self.weak = False                 # True while the current candidate is being accepted through the long-hold path
        self.strong_since, self.streak, self.was_strong = 0.0, 0, False

    def reset_recognition(self):
        self.window.clear()
        self.cand, self.locked, self.diff_since, self.mean, self.progress = None, None, None, None, 0.0
        self.weak, self.strong_since, self.streak, self.was_strong = False, 0.0, 0, False

    def _weak_hold(self, p):
        """Seconds a weak sign must be held: HOLD_SECONDS at confidence CONF, growing linearly to WEAK_MAX_HOLD_SECONDS
        at confidence WEAK_MIN_CONF. The lower the confidence, the longer (and more consistent) the hold must be."""
        low = min(WEAK_MIN_CONF, W.CONF)
        frac = (W.CONF - p) / max(W.CONF - low, 1e-6)
        return W.HOLD_SECONDS + frac * max(WEAK_MAX_HOLD_SECONDS - W.HOLD_SECONDS, 0.0)

    def update(self, now, probs):
        """Same as webcam_asl_v2.Typer.update for confident signs (the strong path below is a copy of its logic).
        Added: a sign whose averaged confidence is between WEAK_MIN_CONF and CONF can still be typed if it stays the clear
        winner for long enough. Also stricter re-arming: after typing, the sign must change or the hand must leave."""
        if probs is not None:
            self.last_hand = now
            self.window.append(probs)
        elif now - self.last_hand > W.GRACE_SECONDS:
            self.window.clear()

        cand, top, weak, p = None, None, False, 0.0
        self.mean = None
        if len(self.window) >= W.MIN_FRAMES:
            self.mean = np.mean(self.window, axis=0)
            order = np.argsort(self.mean)[::-1]
            top, p = self.classes[order[0]], float(self.mean[order[0]])
            margin = p - float(self.mean[order[1]])
            if p >= W.CONF:
                cand = top
            elif (WEAK_ENABLED and p >= min(WEAK_MIN_CONF, W.CONF) and margin >= WEAK_MIN_MARGIN
                  and (WEAK_ALLOW_SPECIAL or top.lower() not in ("space", "del", "delete"))):
                cand, weak = top, True

        strong = cand is not None and not weak
        if strong and not (self.was_strong and cand == self.cand):
            self.strong_since = now                              # a confident stretch of this letter starts now
        self.was_strong = strong
        if cand != self.cand:                                    # a new streak of the same winning letter starts
            self.cand, self.cand_since, self.streak = cand, now, 0
        if cand is not None and probs is not None:
            self.streak += 1                                     # counts real camera frames only
        self.weak = weak

        if self.locked is not None:                              # re-arm: sign changed or hand left (not just low confidence)
            if top == self.locked:
                self.diff_since = None
            else:
                self.diff_since = self.diff_since if self.diff_since is not None else now
                if now - self.diff_since >= W.RELEASE_SECONDS:
                    self.locked, self.diff_since = None, None

        self.progress = 0.0
        if cand is not None:
            if cand == self.locked:
                self.progress = 1.0
            else:
                if weak:
                    self.progress = min((now - self.cand_since) / self._weak_hold(p), self.streak / WEAK_MIN_FRAMES, 1.0)
                else:
                    self.progress = min(1.0, (now - self.strong_since) / W.HOLD_SECONDS)
                if self.progress >= 1.0:
                    self.commit(cand, now)
                    self.locked, self.diff_since = cand, None

    def commit(self, label, now):
        name = label.lower()
        word = self.text.rsplit(" ", 1)[-1]
        before = self.text
        super().commit(label, now)
        if name == "space":
            self._log(f'Space  -  "{word}"' if self.text != before else "Space ignored (no new word)", "gesture")
        elif name in ("del", "delete"):
            self._log("Delete", "gesture")
        else:
            self._log(f"Letter {label}" + ("  (long hold)" if self.weak else ""), "gesture")


class Engine:
    SKELETON = W.CONNECTIONS

    def __init__(self, camera_factory=None):
        self._camera_factory = camera_factory or open_camera
        self.lock = threading.RLock()
        # --- loaded ONCE ---
        self.model, self.classes = load_checkpoint(W.MODEL, "cpu")
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
        self._mp = mp
        self.landmarker = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=W.TASK_PATH),
            running_mode=vision.RunningMode.VIDEO, num_hands=1,
            min_hand_detection_confidence=0.5, min_hand_presence_confidence=0.5, min_tracking_confidence=0.5))
        self.speaker = W.Speaker(enabled=W.SPEAK_WORDS)
        self.voice_available = self.speaker.enabled
        self.history, self.history_version = deque(maxlen=200), 0
        self.typer = UITyper(self.classes, self.speaker, self._log)
        self.t0, self._last_ts = time.time(), -1
        # --- camera thread state ---
        self.paused, self.starting, self.running = False, False, False
        self.error = ""
        self._stop = threading.Event()
        self._stop.set()                                     # "no camera thread is active yet"
        self._thread, self._gen = None, 0
        self._warm_up()
        self._frame = None                                   # newest RGB picture (numpy)
        self._live = dict(hand=False, label="", conf=0.0, fps=0.0)

    def _warm_up(self):
        """First use of MediaPipe / PyTorch is slower than later uses. Doing one dummy run here (while the window shows
        'Loading model...') means Start Camera does not pay for it."""
        try:
            blank = np.zeros((480, 640, 3), np.uint8)
            self._last_ts = 1
            self.landmarker.detect_for_video(self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=blank), 1)
            with torch.no_grad():
                self.model(features_from_raw(np.random.rand(21, 3).astype(np.float32)))
        except Exception as e:                               # warming up is optional - never block start-up
            print("warm-up skipped:", e)

    # ---------------- helpers ----------------
    def now(self):
        return time.time() - self.t0

    def _log(self, text, source):
        self.history.appendleft((time.strftime("%H:%M:%S"), text, source))
        self.history_version += 1

    # ---------------- camera ----------------
    @property
    def active(self):
        """True while a camera is running or starting (False as soon as Stop has been requested)."""
        return (self.running or self.starting) and not self._stop.is_set()

    def start(self, camera_index=0):
        if self.active:
            return
        self.error = ""
        with self.lock:
            self._gen += 1
            self.running, self.starting, self._frame = False, True, None
        stop = threading.Event()
        self._stop = stop
        prev, self._thread = self._thread, None
        self._thread = threading.Thread(target=self._run, args=(camera_index, self._gen, stop, prev), daemon=True)
        self._thread.start()

    def stop(self, wait=False):
        """Returns immediately (releasing a camera can take a moment, so it happens in the camera thread).
        wait=True is used when the application closes, to be sure the camera is released."""
        self._stop.set()
        if wait and self._thread is not None:
            self._thread.join(timeout=5)

    def _run(self, camera_index, gen, stop, prev):
        cap = None
        try:
            if prev is not None and prev.is_alive():
                prev.join(timeout=5)                         # never have two camera instances open at once
            t_start = time.time()
            cap = self._camera_factory(camera_index)
            if stop.is_set():
                return
            if not cap.isOpened():
                self.error = f"Cannot open camera {camera_index}. Close other apps using it, or pick another camera number."
                return
            t_open = time.time() - t_start
            self.running = True
            self.starting = False
            fps, last, first = 0.0, time.time(), True
            while not stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    self.error = "The camera stopped sending pictures."
                    break
                if first:
                    print(f"[camera] opened in {t_open:.2f} s, first frame after {time.time() - t_start:.2f} s")
                frame = cv2.flip(frame, 1)                  # same mirrored view as webcam_asl_v2.py
                h, w = frame.shape[:2]
                now = self.now()
                ts = max(int(now * 1000), self._last_ts + 1)
                self._last_ts = ts
                res = self.landmarker.detect_for_video(
                    self._mp.Image(image_format=self._mp.ImageFormat.SRGB,
                                   data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), ts)
                probs, label, conf = None, "", 0.0
                if res.hand_landmarks:
                    hl = res.hand_landmarks[0]
                    with torch.no_grad():
                        probs = torch.softmax(self.model(features_from_raw(landmarks_to_raw(hl, w, h))), 1)[0].numpy()
                    k = int(probs.argmax())
                    label, conf = self.classes[k], float(probs[k])
                    px = [(int(p.x * w), int(p.y * h)) for p in hl]
                    for a, b in self.SKELETON:
                        cv2.line(frame, px[a], px[b], (0, 255, 0), 2)
                    for p in px:
                        cv2.circle(frame, p, 4, (0, 0, 255), -1)
                with self.lock:
                    if self._gen != gen or stop.is_set():
                        break
                    if self.paused:
                        self._reset_typer_locked()
                    else:
                        self.typer.update(now, probs)
                    t = time.time()
                    fps = 0.9 * fps + 0.1 / max(t - last, 1e-3)
                    last = t
                    self._live = dict(hand=probs is not None, label=label, conf=conf, fps=fps)
                    self._frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if first:
                    print(f"[camera] first recognised frame after {time.time() - t_start:.2f} s")
                    first = False
        except Exception as e:                               # report instead of dying silently
            self.error = f"Recognition stopped: {e}"
        finally:
            if cap is not None:
                cap.release()
            with self.lock:
                if self._gen == gen:                         # a newer Start may already be under way: do not touch its state
                    self.running = self.starting = False
                    self._frame = None
                    self._live = dict(hand=False, label="", conf=0.0, fps=0.0)
                    self._reset_typer_locked()

    def _reset_typer_locked(self):
        self.typer.reset_recognition()

    # ---------------- state for the UI ----------------
    def frame(self):
        with self.lock:
            return self._frame

    def snapshot(self):
        with self.lock:
            t, now = self.typer, self.now()
            flash = (t.flash_msg, t.flash_sub, _hex(t.flash_color)) if now < t.flash_until else None
            if flash and flash[1].startswith("speaking:") and not self.speaker.enabled:
                flash = (flash[0], "voice is switched off", flash[2])
            top3 = []
            if t.mean is not None:
                top3 = [(self.classes[k], float(t.mean[k])) for k in np.argsort(t.mean)[::-1][:3]]
            stopping = self._stop.is_set()                   # Stop was clicked: show "off" immediately
            return dict(running=self.running and not stopping, starting=self.starting and not stopping,
                        paused=self.paused, error=self.error, weak=t.weak,
                        text=t.text, cand=t.cand, progress=t.progress,
                        typed=t.cand is not None and t.cand == t.locked,
                        flash=flash, border=now < t.border_until, top3=top3,
                        speaking=self.speaker.speaking, voice_error=self.speaker.error, **self._live)

    # ---------------- manual actions (work with the camera on or off) ----------------
    def manual_space(self):
        with self.lock:
            before = self.typer.text
            self.typer.commit_space(self.now())
            self._log('Space button  -  "%s"' % before.rsplit(" ", 1)[-1] if self.typer.text != before
                      else "Space button ignored (no new word)", "manual")

    def manual_delete(self):
        with self.lock:
            t = self.typer
            if t.text:
                t.text = t.text[:-1]
                t.flash("DELETE", W.RED, 0.7, self.now())
                self._log("Delete button", "manual")
            else:
                t.flash("Nothing to delete", W.YELLOW, 1.0, self.now())

    def clear_text(self):
        with self.lock:
            self.typer.text = ""
            self._log("Text cleared", "manual")

    def speak_text(self):
        """Speaks the whole text. Returns a short message for the UI."""
        with self.lock:
            text = self.typer.text.strip()
        if not text:
            return "Nothing to speak yet"
        if not self.voice_available:
            return "No voice available on this computer"
        previous, self.speaker.enabled = self.speaker.enabled, True   # an explicit click always speaks
        self.speaker.say(text)
        self.speaker.enabled = previous
        return "Speaking..."

    def set_auto_speak(self, on):
        """Controls the 'speak each word after a space' behaviour of your Speaker."""
        if self.voice_available:
            self.speaker.enabled = bool(on)

    @property
    def voice_tunable(self):
        """True when the voice / speed can be changed (espeak on Linux / Jetson)."""
        return bool(self.voice_available and self.speaker.backend in ("espeak-ng", "espeak"))

    def set_voice(self, voice=None, speed=None):
        """Changes the espeak voice and / or speed for everything spoken from now on."""
        global VOICE, VOICE_SPEED
        if voice:
            VOICE = voice
        if speed:
            VOICE_SPEED = int(speed)

    def test_voice(self):
        """Speaks a short sentence with the current voice settings. Returns a short message for the UI."""
        if not self.voice_available:
            return "No voice available on this computer"
        previous, self.speaker.enabled = self.speaker.enabled, True
        self.speaker.say("Hello, this is SignSpeak")
        self.speaker.enabled = previous
        return "Playing voice test..."

    def reset_recognition(self):
        with self.lock:
            self._reset_typer_locked()

    def set_paused(self, paused):
        with self.lock:
            self.paused = paused
            self._reset_typer_locked()

    def clear_history(self):
        self.history.clear()
        self.history_version += 1

    def shutdown(self):
        self.stop(wait=True)
        try:
            self.landmarker.close()
        except Exception:
            pass
