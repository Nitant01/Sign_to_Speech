"""
jetson_selftest.py  -  checks every part of SignSpeak step by step and measures speed.  READ-ONLY: it changes nothing.

Run from the project folder with the virtual environment active:
    python jetson_selftest.py --image "Dataset C/A/A0001_test.jpg"            # no camera needed
    python jetson_selftest.py --image "Dataset C/A/A0001_test.jpg" --camera 0 # also tests the camera + speed (~15 s)
    add --speak to also test the voice.

It prints PASS / FAIL for each step and saves everything to  jetson_selftest_results.json
(real measurements for the report: model name, library versions, latency, FPS).
"""
import argparse, glob, json, os, platform, statistics, subprocess, sys, time
import numpy as np

RESULTS = {"when": time.strftime("%Y-%m-%d %H:%M:%S"), "steps": {}}
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def step(name):
    def deco(fn):
        def run(*a, **k):
            print(f"\n=== {name}")
            try:
                out = fn(*a, **k)
                RESULTS["steps"][name] = {"status": "PASS", **(out or {})}
                print("PASS")
                return out
            except Exception as e:                                   # keep going, report at the end
                RESULTS["steps"][name] = {"status": "FAIL", "error": f"{type(e).__name__}: {e}"}
                print(f"FAIL  {type(e).__name__}: {e}")
                return None
        return run
    return deco


def _read(path):
    try:
        with open(path, "rb") as f:
            return f.read().replace(b"\x00", b"").decode(errors="replace").strip()
    except Exception:
        return None


def _cmd(*c):
    try:
        return subprocess.run(c, capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return None


def ms_stats(v):
    v = sorted(v)
    return {"mean_ms": round(statistics.mean(v), 3), "median_ms": round(statistics.median(v), 3),
            "p95_ms": round(v[int(0.95 * (len(v) - 1))], 3), "n": len(v)}


@step("1. Device and library versions")
def versions():
    info = {"python": platform.python_version(), "machine": platform.machine(), "os": platform.platform(),
            "jetson_model": _read("/proc/device-tree/model"), "l4t_release": _read("/etc/nv_tegra_release"),
            "power_mode": _cmd("nvpmodel", "-q")}
    import numpy, cv2, torch
    info.update(numpy=numpy.__version__, opencv=cv2.__version__, opencv_file=cv2.__file__, torch=torch.__version__,
                torch_cuda_build=torch.version.cuda, cuda_available=bool(torch.cuda.is_available()))
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    for mod in ("mediapipe", "customtkinter", "PIL", "tkinter"):
        try:
            m = __import__(mod)
            info[mod] = getattr(m, "__version__", "ok")
        except Exception as e:
            info[mod] = f"MISSING ({type(e).__name__}: {e})"
    try:
        from PIL import ImageTk  # noqa: F401
        info["PIL.ImageTk"] = "ok"
    except Exception as e:
        info["PIL.ImageTk"] = f"MISSING ({e})"
    for k, v in info.items():
        print(f"  {k}: {v}")
    if "MISSING" in str(info.get("mediapipe")):
        raise RuntimeError("mediapipe cannot be imported")
    return info


@step("2. Checkpoint and model")
def checkpoint():
    import torch
    from asl_common import load_checkpoint, N_FEATURES
    path = os.path.join(HERE, "sign_landmark_dataset_c_v2.pt")
    model, classes = load_checkpoint(path)
    params = sum(p.numel() for p in model.parameters())
    print(f"  {len(classes)} classes: {classes}\n  trainable parameters: {params}   input features: {N_FEATURES}")
    x = torch.randn(1, N_FEATURES)
    out = {"classes": len(classes), "parameters": params, "input_features": N_FEATURES}
    with torch.no_grad():
        for _ in range(20):
            model(x)
        t = []
        for _ in range(500):
            s = time.perf_counter(); model(x); t.append((time.perf_counter() - s) * 1000)
        out["mlp_cpu_latency"] = ms_stats(t); print("  MLP on CPU :", out["mlp_cpu_latency"])
        if torch.cuda.is_available():
            mg, xg = model.to("cuda"), x.to("cuda")
            for _ in range(30):
                mg(xg)
            torch.cuda.synchronize(); t = []
            for _ in range(500):
                s = time.perf_counter(); mg(xg); torch.cuda.synchronize(); t.append((time.perf_counter() - s) * 1000)
            out["mlp_cuda_latency"] = ms_stats(t); print("  MLP on CUDA:", out["mlp_cuda_latency"])
    return out


def make_landmarker(video):
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    task = os.path.join(HERE, "hand_landmarker.task")
    return mp, vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=task),
        running_mode=vision.RunningMode.VIDEO if video else vision.RunningMode.IMAGE, num_hands=1,
        min_hand_detection_confidence=0.5, min_hand_presence_confidence=0.5, min_tracking_confidence=0.5))


@step("3. MediaPipe HandLandmarker on one image")
def image_test(image):
    import cv2, torch
    from asl_common import features_from_raw, landmarks_to_raw, load_checkpoint
    if not image or not os.path.exists(image):
        raise RuntimeError(f"image not found: {image!r}  (use --image \"Dataset C/A/A0001_test.jpg\")")
    mp, lm = make_landmarker(video=False)
    img = cv2.imread(image); h, w = img.shape[:2]
    mpimg = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    res = lm.detect(mpimg)
    if not res.hand_landmarks:
        raise RuntimeError("no hand found in that image (try another picture)")
    model, classes = load_checkpoint(os.path.join(HERE, "sign_landmark_dataset_c_v2.pt"))
    with torch.no_grad():
        p = torch.softmax(model(features_from_raw(landmarks_to_raw(res.hand_landmarks[0], w, h))), 1)[0]
    top = int(p.argmax())
    print(f"  hand found; model says '{classes[top]}' with confidence {float(p[top]):.2f}   (folder name: {os.path.basename(os.path.dirname(image))})")
    t = []
    for _ in range(50):
        s = time.perf_counter(); lm.detect(mpimg); t.append((time.perf_counter() - s) * 1000)
    lm.close()
    stats = ms_stats(t); print("  MediaPipe detect (IMAGE mode):", stats)
    return {"image": image, "predicted": classes[top], "confidence": round(float(p[top]), 3), "mediapipe_image_latency": stats}


def benchmark_loop(cap, seconds, save_frame=None):
    """Same steps as the app's camera thread, with each stage timed. cap needs read()."""
    import cv2, torch
    from asl_common import features_from_raw, landmarks_to_raw, load_checkpoint
    model, classes = load_checkpoint(os.path.join(HERE, "sign_landmark_dataset_c_v2.pt"))
    mp, lm = make_landmarker(video=True)
    t_cap, t_mp, t_mlp, t_all, hands, n, last_ts, preds = [], [], [], [], 0, 0, -1, {}
    t0 = time.perf_counter(); saved = False
    while time.perf_counter() - t0 < seconds:
        a = time.perf_counter(); ok, frame = cap.read(); b = time.perf_counter()
        if not ok:
            break
        frame = cv2.flip(frame, 1); h, w = frame.shape[:2]
        ts = max(int((time.perf_counter() - t0) * 1000), last_ts + 1); last_ts = ts
        c = time.perf_counter()
        res = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), ts)
        d = time.perf_counter(); e = d
        if res.hand_landmarks:
            hands += 1
            raw = landmarks_to_raw(res.hand_landmarks[0], w, h)
            with torch.no_grad():
                p = torch.softmax(model(features_from_raw(raw)), 1)[0]
            e = time.perf_counter()
            k = classes[int(p.argmax())]; preds[k] = preds.get(k, 0) + 1
            if save_frame and not saved and hands > 10:
                for x, y, _ in raw:
                    cv2.circle(frame, (int(x), int(y)), 4, (0, 0, 255), -1)
                cv2.putText(frame, f"{k} {float(p.max()):.2f}", (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2)
                cv2.imwrite(save_frame, frame); saved = True
            t_mlp.append((e - d) * 1000)
        t_cap.append((b - a) * 1000); t_mp.append((d - c) * 1000); t_all.append((e - a) * 1000); n += 1
    lm.close()
    if n < 5:
        raise RuntimeError(f"only {n} frames could be read")
    wall = time.perf_counter() - t0
    out = {"frames": n, "seconds": round(wall, 2), "loop_fps": round(n / wall, 1), "frames_with_hand_pct": round(100 * hands / n, 1),
           "capture_latency": ms_stats(t_cap), "mediapipe_video_latency": ms_stats(t_mp),
           "processing_latency_per_frame": ms_stats(t_all), "predicted_letter_counts": preds}
    if t_mlp:
        out["mlp_latency_in_loop"] = ms_stats(t_mlp)
    for k, v in out.items():
        print(f"  {k}: {v}")
    return out


@step("4. Camera + full pipeline speed")
def camera_test(index, seconds):
    import cv2
    devs = sorted(glob.glob("/dev/video*")); print("  video devices:", devs or "none found")
    from signspeak_engine import open_camera
    cap = open_camera(index)
    if not cap.isOpened():
        raise RuntimeError(f"camera {index} did not open (check the device list above and try --camera N)")
    t = time.perf_counter()
    for _ in range(5):
        cap.read()
    print(f"  camera opened, {cap.get(cv2.CAP_PROP_FRAME_WIDTH):.0f}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT):.0f}; show your hand to the camera now ...")
    out = {"devices": devs, "camera_index": index,
           "resolution": f"{cap.get(cv2.CAP_PROP_FRAME_WIDTH):.0f}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT):.0f}"}
    try:
        out.update(benchmark_loop(cap, seconds, save_frame=os.path.join(HERE, "jetson_selftest_frame.jpg")))
    finally:
        cap.release()
    if os.path.exists(os.path.join(HERE, "jetson_selftest_frame.jpg")):
        print("  saved jetson_selftest_frame.jpg (one annotated frame - usable as a screenshot)")
    return out


@step("5. Speech")
def speech_test(speak):
    import webcam_asl_v2 as W
    backend = W.Speaker._pick_backend()
    print("  voice backend:", backend)
    if backend is None:
        raise RuntimeError("no voice program found. Install one with:  sudo apt install espeak-ng")
    if speak:
        sp = W.Speaker(True); sp.say("signspeak test"); sp.wait_until_done()
        if sp.error:
            raise RuntimeError(sp.error)
        print("  (you should have heard 'signspeak test')")
    return {"backend": backend, "spoken": bool(speak)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="", help='e.g. "Dataset C/A/A0001_test.jpg"')
    ap.add_argument("--camera", type=int, default=None, help="camera index to test (0 = /dev/video0)")
    ap.add_argument("--seconds", type=float, default=15)
    ap.add_argument("--speak", action="store_true")
    a = ap.parse_args()
    versions(); checkpoint()
    image_test(a.image)
    if a.camera is not None:
        camera_test(a.camera, a.seconds)
    speech_test(a.speak)
    out = os.path.join(HERE, "jetson_selftest_results.json")
    with open(out, "w") as f:
        json.dump(RESULTS, f, indent=2, default=str)
    print("\n================ SUMMARY")
    for k, v in RESULTS["steps"].items():
        print(f"  {v['status']:>4}  {k}" + (f"   -> {v['error']}" if v["status"] == "FAIL" else ""))
    print(f"\nSaved {out}  (send me this file if anything failed)")


if __name__ == "__main__":
    main()
