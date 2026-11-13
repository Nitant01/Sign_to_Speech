"""
signspeak_app.py  -  SignSpeak: desktop window for your ASL recognition system.

Run:   python signspeak_app.py

Only UI code lives here. All recognition is done by your existing code through signspeak_engine.py.
"""
import os
import sys
import time
import tkinter as tk
from tkinter import messagebox

import cv2
import customtkinter as ctk
if sys.platform.startswith("linux"):
    # Linux / Jetson: customtkinter's default rounded-corner method draws circles with a bundled font, and Tk aborted there
    # with "*** stack smashing detected ***" (seen on the Jetson). Polygon corners need no font and look almost the same.
    from customtkinter.windows.widgets.core_rendering import DrawEngine
    DrawEngine.preferred_drawing_method = "polygon_shapes"
from PIL import Image, ImageTk

# make relative paths in your scripts (model, hand_landmarker.task) work no matter where this is launched from
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import webcam_asl_v2 as W                 # your settings (CONF, HOLD_SECONDS) are adjusted at run time only
import signspeak_engine as SE                      # its VOICE / VOICE_SPEED settings are changed at run time from the Voice tab
from signspeak_engine import Engine

# ---------------- look & feel ----------------
BG, CARD, CARD2 = "#0B1220", "#151E2F", "#1D2940"
TEXT, MUTED = "#E6EBF5", "#8A97B1"
ACCENT, GREEN, RED, AMBER, BLUE = "#6C7CFF", "#22C55E", "#EF4444", "#F59E0B", "#3B82F6"
FONT = "Segoe UI" if sys.platform.startswith("win") else "DejaVu Sans"             # Segoe UI exists only on Windows
MONO = "Consolas" if sys.platform.startswith("win") else "DejaVu Sans Mono"
IS_WIN = sys.platform.startswith("win")
LINUX_SYMBOLS = False   # False = leave symbol glyphs out on Linux / Jetson (they crashed Tk there)


def emo(text):
    """Windows shows all symbols / emoji (arrows, play button, dots ...). On Linux / Jetson, Tk 8.6 + Xft can abort with
    "*** stack smashing detected ***" while measuring such glyphs, so there only plain letters and common punctuation
    (up to U+2026) are kept and the symbols are left out. Set LINUX_SYMBOLS = True to try them again."""
    if IS_WIN or LINUX_SYMBOLS:
        return text
    return "".join(c for c in text if ord(c) <= 0x2026).strip()


SIDE_W = 500          # the right-hand column has a FIXED width, so nothing inside it can ever move the layout
PRED_H = 188          # fixed height of the prediction card
# (name shown in the Voice tab, espeak-ng voice code). Add your own lines here; list all voices with:  espeak-ng --voices=en
VOICES = [
    ("Female", "en-us+f3"),
    ("Male", "en-us+m3"),
]
ctk.set_appearance_mode("dark")


def font(size, weight="normal"):
    return ctk.CTkFont(family=FONT, size=size, weight=weight)


HELP_TEXT = """HOW TO USE
1.  Click  Start Camera  and show one hand, palm towards the camera.
2.  Hold a letter sign still. The blue bar fills; when it is full the letter is typed.
3.  Hold the SPACE sign (or press the Space button) to add a space - the word you just finished is spoken.
4.  Hold the DEL sign (or press Delete) to remove the last character.
5.  To type the same letter twice, lower your hand (or change sign) in between.

KEYBOARD SHORTCUTS
Space bar        add a space and speak the word
Backspace / B    delete last character
C                clear all text (asks first)
Ctrl + C         copy the text
S                speak the whole text
P                pause / resume recognition
D                show / hide the top-3 predictions
F11 / Esc        full screen / leave full screen
F1               this help
Q                quit

SETTINGS (side panel)
Confidence threshold  - how sure the model must be before a sign counts.
Hold time             - how long a sign must stay still before it is typed.
Both only apply while the app is running; your scripts are never changed."""


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("SignSpeak – ASL Recognition System")
        self.geometry("1360x860")
        self.minsize(1100, 780)
        self.configure(fg_color=BG)
        self.engine = None
        self._photo = None
        self._last = {}
        self._history_seen = -1
        self._toast_until = 0.0
        self._shown_error = ""
        self._show_top3 = True
        self._build()
        self._bind_keys()
        self.protocol("WM_DELETE_WINDOW", self._quit_now)
        self._set_status("Loading model…", AMBER)
        self.after(100, self._load_engine)           # let the window appear first

    # ================= layout =================
    def _build(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # ---- header ----
        head = ctk.CTkFrame(self, fg_color="transparent")
        head.grid(row=0, column=0, sticky="ew", padx=24, pady=(18, 8))
        head.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(head, text=emo("🤟"), font=font(34)).grid(row=0, column=0, rowspan=2, padx=(0, 12))
        ctk.CTkLabel(head, text="SignSpeak", font=font(26, "bold"), text_color=TEXT, anchor="w").grid(row=0, column=1, sticky="w")
        ctk.CTkLabel(head, text="ASL Recognition System", font=font(13), text_color=MUTED, anchor="w").grid(row=1, column=1, sticky="w")
        right = ctk.CTkFrame(head, fg_color="transparent")
        right.grid(row=0, column=2, rowspan=2, sticky="e")
        self.status = ctk.CTkLabel(right, text="", font=font(13, "bold"), corner_radius=16, width=190, height=32, text_color="white")
        self.status.pack(side="left", padx=(0, 12))
        self._btn(right, "Help  (F1)", self._show_help, CARD2, width=100).pack(side="left", padx=4)
        self.fs_btn = self._btn(right, "Full screen  (F11)", self._toggle_fullscreen, CARD2, width=150)
        self.fs_btn.pack(side="left", padx=4)

        # ---- main area ----
        main = ctk.CTkFrame(self, fg_color="transparent")
        main.grid(row=1, column=0, sticky="nsew", padx=24, pady=(4, 20))
        main.grid_columnconfigure(0, weight=1)
        main.grid_columnconfigure(1, weight=0, minsize=SIDE_W)
        main.grid_rowconfigure(0, weight=1)

        # ---- camera card ----
        cam = ctk.CTkFrame(main, fg_color=CARD, corner_radius=20)
        cam.grid(row=0, column=0, sticky="nsew", padx=(0, 14))
        cam.grid_columnconfigure(0, weight=1)
        cam.grid_rowconfigure(0, weight=1)
        self.video_box = tk.Frame(cam, bg="#05080F", width=400, height=260)
        self.video_box.grid(row=0, column=0, sticky="nsew", padx=14, pady=(14, 8))
        self.video = tk.Label(self.video_box, bg="#05080F", fg=MUTED, font=(FONT, 15), bd=0,
                              text="Camera is off\n\nClick “Start Camera” to begin", justify="center")
        self.video.place(x=0, y=0, relwidth=1, relheight=1)     # place() => the picture never changes the layout
        bar = ctk.CTkFrame(cam, fg_color="transparent")
        bar.grid(row=1, column=0, sticky="ew", padx=14, pady=(0, 14))
        self.cam_btn = self._btn(bar, emo("▶  Start Camera"), self._toggle_camera, GREEN, width=170, height=42)
        self.cam_btn.pack(side="left")
        self.pause_btn = self._btn(bar, emo("⏸  Pause"), self._toggle_pause, CARD2, width=110, height=42)
        self.pause_btn.pack(side="left", padx=8)
        self.reset_btn = self._btn(bar, emo("↺  Reset"), self._reset_recognition, CARD2, width=100, height=42)
        self.reset_btn.pack(side="left")
        self.fps_label = ctk.CTkLabel(bar, text="", font=font(12), text_color=MUTED, width=56, anchor="e")
        self.fps_label.pack(side="right", padx=(8, 4))
        self.cam_index = ctk.CTkOptionMenu(bar, values=["0", "1", "2", "3"], width=64, height=34, fg_color=CARD2,
                                           button_color=CARD2, button_hover_color="#2A3A58", font=font(13))
        self.cam_index.set(str(W.CAMERA))
        self.cam_index.pack(side="right")
        ctk.CTkLabel(bar, text="Camera", font=font(12), text_color=MUTED).pack(side="right", padx=6)

        # ---- right column ----
        side = ctk.CTkFrame(main, fg_color="transparent", width=SIDE_W)
        side.grid(row=0, column=1, sticky="ns")
        side.grid_propagate(False)                       # content (text lengths, font sizes) can never resize this column
        side.grid_columnconfigure(0, weight=1)
        side.grid_rowconfigure(1, weight=1, minsize=250)
        self._build_prediction(side)
        self._build_text(side)
        self._build_tabs(side)

    def _btn(self, parent, text, command, color, width=120, height=36, text_color="white"):
        hover = {GREEN: "#16A34A", RED: "#DC2626", ACCENT: "#5566F0", BLUE: "#2563EB", AMBER: "#D97706"}.get(color, "#2A3A58")
        return ctk.CTkButton(parent, text=text, command=command, fg_color=color, hover_color=hover, text_color=text_color,
                             corner_radius=12, width=width, height=height, font=font(13, "bold"))

    def _build_prediction(self, side):
        card = ctk.CTkFrame(side, fg_color=CARD, corner_radius=20, height=PRED_H)
        card.grid(row=0, column=0, sticky="ew", pady=(0, 14))
        card.grid_propagate(False)                       # fixed size: labels inside may change text freely
        card.grid_columnconfigure(1, weight=1)
        self.f_letter, self.f_word = font(64, "bold"), font(34, "bold")      # created ONCE (a new font object per frame caused constant redraws)
        self.big = ctk.CTkLabel(card, text="–", font=self.f_letter, text_color=MUTED, width=150, height=84)
        self.big.grid(row=0, column=0, rowspan=3, padx=(14, 6), pady=(8, 4))
        self.hand_label = ctk.CTkLabel(card, text=emo("●  No hand detected"), font=font(14, "bold"), text_color=MUTED, anchor="w")
        self.hand_label.grid(row=0, column=1, sticky="ew", padx=(6, 18), pady=(12, 0))
        self.conf_label = ctk.CTkLabel(card, text="Confidence  0%", font=font(13), text_color=MUTED, anchor="w")
        self.conf_label.grid(row=1, column=1, sticky="ew", padx=(6, 18))
        self.conf_bar = ctk.CTkProgressBar(card, height=10, corner_radius=5, fg_color=CARD2, progress_color=ACCENT)
        self.conf_bar.set(0)
        self.conf_bar.grid(row=2, column=1, sticky="ew", padx=(6, 18), pady=(0, 6))
        self.state_label = ctk.CTkLabel(card, text="Camera off", font=font(13), text_color=TEXT, anchor="w")
        self.state_label.grid(row=3, column=0, columnspan=2, sticky="ew", padx=18)
        self.hold_bar = ctk.CTkProgressBar(card, height=12, corner_radius=6, fg_color=CARD2, progress_color=BLUE)
        self.hold_bar.set(0)
        self.hold_bar.grid(row=4, column=0, columnspan=2, sticky="ew", padx=18, pady=(4, 8))
        self.top3_frame = ctk.CTkFrame(card, fg_color="transparent")
        self.top3_frame.grid(row=5, column=0, columnspan=2, sticky="ew", padx=18, pady=(0, 12))
        self.top3_label = ctk.CTkLabel(self.top3_frame, text="", font=font(12), text_color=MUTED, anchor="w")
        self.top3_label.pack(fill="x")

    def _build_text(self, side):
        card = ctk.CTkFrame(side, fg_color=CARD, corner_radius=20)
        card.grid(row=1, column=0, sticky="nsew", pady=(0, 14))
        card.grid_propagate(False)
        card.grid_columnconfigure((0, 1, 2), weight=1, uniform="b")
        card.grid_rowconfigure(1, weight=1)
        ctk.CTkLabel(card, text="Recognised text", font=font(15, "bold"), text_color=TEXT, anchor="w").grid(
            row=0, column=0, columnspan=2, sticky="w", padx=18, pady=(14, 4))
        self.stats = ctk.CTkLabel(card, text="0 words · 0 characters", font=font(12), text_color=MUTED, anchor="e", width=200)
        self.stats.grid(row=0, column=2, sticky="e", padx=18, pady=(14, 4))
        self.textbox = ctk.CTkTextbox(card, font=font(26, "bold"), fg_color=CARD2, text_color=TEXT, corner_radius=14,
                                      wrap="word", height=90, border_width=0)
        self.textbox.grid(row=1, column=0, columnspan=3, sticky="nsew", padx=16, pady=4)
        self.textbox.tag_config("cursor", foreground=TEXT)    # the blinking cursor is a tagged character (no text rewrite per blink)
        self.textbox.configure(state="disabled")
        self.toast = ctk.CTkLabel(card, text="", font=font(13, "bold"), text_color=BLUE, anchor="w", height=22)
        self.toast.grid(row=2, column=0, columnspan=3, sticky="ew", padx=20)
        self._btn(card, emo("␣  Space"), self._manual_space, BLUE, height=42).grid(row=3, column=0, sticky="ew", padx=(16, 4), pady=4)
        self._btn(card, emo("⌫  Delete"), self._manual_delete, AMBER, height=42, text_color="#1A1204").grid(row=3, column=1, sticky="ew", padx=4, pady=4)
        self._btn(card, emo("🗑  Clear"), self._clear, RED, height=42).grid(row=3, column=2, sticky="ew", padx=(4, 16), pady=4)
        self._btn(card, emo("📋  Copy"), self._copy, CARD2, height=42).grid(row=4, column=0, columnspan=1, sticky="ew", padx=(16, 4), pady=(4, 16))
        self._btn(card, emo("🔊  Speak text"), self._speak, ACCENT, height=42).grid(row=4, column=1, columnspan=2, sticky="ew", padx=(4, 16), pady=(4, 16))

    def _build_tabs(self, side):
        tabs = ctk.CTkTabview(side, fg_color=CARD, corner_radius=20, height=192, segmented_button_fg_color=CARD2,
                              segmented_button_selected_color=ACCENT, segmented_button_unselected_color=CARD2)
        tabs.grid(row=2, column=0, sticky="ew")
        tabs.grid_propagate(False)
        tabs._segmented_button.configure(font=font(13, "bold"))
        s, v, h = tabs.add("Settings"), tabs.add("Voice"), tabs.add("History")

        # ---- Voice tab: voice, speed, test (works with espeak-ng on Linux / Jetson) ----
        v.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(v, text="Voice", font=font(13), text_color=TEXT, anchor="w", width=84).grid(row=0, column=0, sticky="w", pady=(8, 4))
        self._voice_names = ["Female", "Male"]
        self.voice_menu = ctk.CTkOptionMenu(v, values=self._voice_names, command=self._on_voice_pick, height=30, font=font(13),
                                            dropdown_font=font(13), fg_color=CARD2, button_color=ACCENT, button_hover_color="#5566F0")
        self.voice_menu.grid(row=0, column=1, columnspan=2, sticky="ew", padx=(8, 0), pady=(8, 4))
        ctk.CTkLabel(v, text="Speed", font=font(13), text_color=TEXT, anchor="w", width=84).grid(row=1, column=0, sticky="w", pady=4)
        self.speed_slider = ctk.CTkSlider(v, from_=100, to=220, number_of_steps=12, command=self._on_speed, button_color=ACCENT,
                                          progress_color=ACCENT, height=16)
        self.speed_slider.set(SE.VOICE_SPEED)
        self.speed_slider.grid(row=1, column=1, sticky="ew", padx=8, pady=4)
        self.speed_val = ctk.CTkLabel(v, text=f"{SE.VOICE_SPEED} wpm", font=font(13, "bold"), text_color=ACCENT, width=64)
        self.speed_val.grid(row=1, column=2, pady=4)
        self.test_btn = self._btn(v, "Test voice", self._test_voice, ACCENT, width=110, height=30)
        self.test_btn.grid(row=2, column=0, columnspan=2, sticky="w", pady=(8, 0))
        self.voice_note = ctk.CTkLabel(v, text="", font=font(12), text_color=MUTED, anchor="w", justify="left", wraplength=400)
        self.voice_note.grid(row=2, column=1, columnspan=2, sticky="e", pady=(8, 0))

        s.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(s, text="Confidence", font=font(13), text_color=TEXT, anchor="w", width=84).grid(row=0, column=0, sticky="w", pady=(8, 4))
        self.conf_slider = ctk.CTkSlider(s, from_=0.30, to=0.95, number_of_steps=13, command=self._on_conf, button_color=ACCENT,
                                         progress_color=ACCENT, height=16)
        self.conf_slider.set(W.CONF)
        self.conf_slider.grid(row=0, column=1, sticky="ew", padx=8, pady=(8, 4))
        self.conf_val = ctk.CTkLabel(s, text=f"{W.CONF:.2f}", font=font(13, "bold"), text_color=ACCENT, width=44)
        self.conf_val.grid(row=0, column=2, pady=(8, 4))
        ctk.CTkLabel(s, text="Hold time", font=font(13), text_color=TEXT, anchor="w", width=84).grid(row=1, column=0, sticky="w", pady=4)
        self.hold_slider = ctk.CTkSlider(s, from_=0.3, to=1.5, number_of_steps=12, command=self._on_hold, button_color=ACCENT,
                                         progress_color=ACCENT, height=16)
        self.hold_slider.set(W.HOLD_SECONDS)
        self.hold_slider.grid(row=1, column=1, sticky="ew", padx=8, pady=4)
        self.hold_val = ctk.CTkLabel(s, text=f"{W.HOLD_SECONDS:.1f} s", font=font(13, "bold"), text_color=ACCENT, width=44)
        self.hold_val.grid(row=1, column=2, pady=4)
        self.voice_switch = ctk.CTkSwitch(s, text="Speak each word after a space", font=font(13), progress_color=GREEN,
                                          command=self._on_voice)
        self.voice_switch.grid(row=2, column=0, columnspan=3, sticky="w", pady=(6, 0))

        h.grid_columnconfigure(0, weight=1)
        h.grid_rowconfigure(0, weight=1)
        self.hist = ctk.CTkTextbox(h, font=ctk.CTkFont(family=MONO, size=12), fg_color=CARD2, corner_radius=10, height=110)
        self.hist.grid(row=0, column=0, sticky="nsew")
        self.hist.configure(state="disabled")
        self._btn(h, "Clear history", self._clear_history, CARD2, width=110, height=28).grid(row=1, column=0, sticky="e", pady=(6, 0))

    # ================= engine =================
    def _load_engine(self):
        try:
            self.engine = Engine()
        except SystemExit as e:                      # asl_common refuses an old checkpoint with this
            self._fatal(str(e).strip())
            return
        except Exception as e:
            if not (os.path.exists(W.MODEL) and os.path.exists(W.TASK_PATH)):
                hint = f"\n\nA project file is missing. Run the app from the folder that contains {W.MODEL} and {W.TASK_PATH}."
            elif "4551" in str(e) or "Application Control" in str(e):
                hint = ("\n\nWindows (Smart App Control / Code Integrity) blocked a MediaPipe file. "
                        "Your project files are fine - this is a Windows policy, not a missing file.")
            else:
                hint = ""
            self._fatal(f"Could not start the recognition engine:\n\n{type(e).__name__}: {e}{hint}")
            return
        if self.engine.voice_available:
            self.voice_switch.select()
        else:
            self.voice_switch.deselect(); self.voice_switch.configure(state="disabled", text="Voice not available on this computer")
        self._setup_voice_tab()
        self._set_status("Camera off", MUTED)
        self._tick()

    def _setup_voice_tab(self):
        """Configure the two available English voices and speech speed."""
        e = self.engine

        current = next(
            ((name, code) for name, code in VOICES if code == SE.VOICE),
            None
        )

        if current is None:
            current = ("Female", "en-us+f3")
            e.set_voice(voice=current[1])

        self.voice_menu.set(current[0])

        if e.voice_tunable:
            self.voice_menu.configure(state="normal")
            self.speed_slider.configure(state="normal")
            self.test_btn.configure(state="normal")
            self.voice_note.configure(
                text="Choose Male or Female and adjust speaking speed."
            )
        else:
            self.voice_menu.configure(state="disabled")
            self.speed_slider.configure(state="disabled")
            self.test_btn.configure(
                state="normal" if e.voice_available else "disabled"
            )
            self.voice_note.configure(
                text="Voice settings require espeak-ng on Linux / Jetson."
            )

    def _on_voice_pick(self, name):
        code = dict(VOICES).get(name)
        if self.engine and code:
            self.engine.set_voice(voice=code)

    def _on_speed(self, value):
        wpm = int(round(float(value) / 10) * 10)
        if self.speed_val.cget("text") != f"{wpm} wpm":
            self.speed_val.configure(text=f"{wpm} wpm")
        if self.engine:
            self.engine.set_voice(speed=wpm)

    def _test_voice(self):
        if self.engine:
            self._say(self.engine.test_voice())

    def _fatal(self, msg):
        self._set_status("Engine error", RED)
        self.cam_btn.configure(state="disabled")
        messagebox.showerror("SignSpeak", msg)

    def _toggle_camera(self):
        e = self.engine
        if e is None:
            return
        if e.active:
            e.stop()
            self.cam_btn.configure(text=emo("▶  Start Camera"), fg_color=GREEN, hover_color="#16A34A")
            self.cam_index.configure(state="normal")
            self.pause_btn.configure(text=emo("⏸  Pause"), fg_color=CARD2, hover_color="#2A3A58", text_color="white")
            e.set_paused(False)
        else:
            self._shown_error = ""
            e.start(int(self.cam_index.get()))
            self.cam_btn.configure(text=emo("■  Stop Camera"), fg_color=RED, hover_color="#DC2626")
            self.cam_index.configure(state="disabled")

    def _toggle_pause(self):
        e = self.engine
        if e is None or not e.active:
            self._say("Start the camera first")
            return
        e.set_paused(not e.paused)
        self.pause_btn.configure(text=emo("▶  Resume") if e.paused else emo("⏸  Pause"), fg_color=AMBER if e.paused else CARD2,
                                 hover_color="#D97706" if e.paused else "#2A3A58", text_color="#1A1204" if e.paused else "white")

    def _reset_recognition(self):
        if self.engine:
            self.engine.reset_recognition()
            self._say("Recognition reset")

    # ================= text actions =================
    def _manual_space(self):
        if self.engine:
            self.engine.manual_space()

    def _manual_delete(self):
        if self.engine:
            self.engine.manual_delete()

    def _clear(self):
        if not self.engine or not self.engine.typer.text:
            self._say("Nothing to clear")
            return
        if messagebox.askyesno("Clear text", "Delete all recognised text?"):
            self.engine.clear_text()
            self._say("Text cleared")

    def _copy(self):
        text = self.engine.typer.text if self.engine else ""
        if not text:
            self._say("Nothing to copy")
            return
        self.clipboard_clear(); self.clipboard_append(text); self.update()
        self._say("Copied to clipboard")

    def _speak(self):
        if self.engine:
            self._say(self.engine.speak_text())

    def _say(self, msg, seconds=2.5):
        self._set(self.toast, "toast", text=msg, text_color=BLUE)
        self._toast_until = time.time() + seconds

    # ================= settings =================
    def _on_conf(self, v):
        W.CONF = round(float(v), 2)                  # your Typer reads this value on every frame
        self.conf_val.configure(text=f"{W.CONF:.2f}")

    def _on_hold(self, v):
        W.HOLD_SECONDS = round(float(v), 1)
        self.hold_val.configure(text=f"{W.HOLD_SECONDS:.1f} s")

    def _on_voice(self):
        if self.engine:
            self.engine.set_auto_speak(bool(self.voice_switch.get()))

    def _clear_history(self):
        if self.engine:
            self.engine.clear_history()

    # ================= window helpers =================
    def _show_help(self):
        win = ctk.CTkToplevel(self)
        win.title("SignSpeak – Help")
        win.geometry("560x640")
        win.configure(fg_color=BG)
        win.after(150, win.lift)
        box = ctk.CTkTextbox(win, font=font(14), fg_color=CARD, corner_radius=14, wrap="word")
        box.pack(fill="both", expand=True, padx=16, pady=16)
        box.insert("1.0", HELP_TEXT)
        box.configure(state="disabled")

    def _toggle_fullscreen(self, _e=None):
        full = not bool(self.attributes("-fullscreen"))
        self.attributes("-fullscreen", full)
        self.fs_btn.configure(text="Exit full screen  (Esc)" if full else "Full screen  (F11)")

    def _exit_fullscreen(self, _e=None):
        if self.attributes("-fullscreen"):
            self._toggle_fullscreen()

    def _toggle_top3(self):
        self._show_top3 = not self._show_top3
        if self._show_top3:
            self.top3_frame.grid()
        else:
            self.top3_frame.grid_remove()

    def _quit(self):
        if self.engine and self.engine.typer.text and not messagebox.askyesno("Quit", "Quit SignSpeak? The text will be lost."):
            return
        self._quit_now()

    def _quit_now(self):
        if self.engine:
            self.engine.shutdown()
        self.destroy()

    def _bind_keys(self):
        def key(fn):
            def handler(e):
                if isinstance(self.focus_get(), tk.Entry):                       # never steal typing from an input field
                    return None
                fn()
                return "break"
            return handler
        for k, fn in {"<space>": self._manual_space, "<BackSpace>": self._manual_delete, "<b>": self._manual_delete,
                      "<c>": self._clear, "<s>": self._speak, "<p>": self._toggle_pause, "<d>": self._toggle_top3,
                      "<q>": self._quit, "<F1>": self._show_help}.items():
            self.bind(k, key(fn))
        self.bind("<Control-c>", lambda e: (self._copy(), "break")[1])
        self.bind("<F11>", self._toggle_fullscreen)
        self.bind("<Escape>", self._exit_fullscreen)

    # ================= live updates =================
    def _set_status(self, text, color):
        if self._last.get("status") != (text, color):
            self.status.configure(text=text, fg_color=color)
            self._last["status"] = (text, color)

    def _set(self, widget, key, **kw):
        """configure a widget only when its values actually changed (prevents flicker)"""
        sig = tuple(sorted(kw.items()))
        if self._last.get(key) != sig:
            widget.configure(**kw)
            self._last[key] = sig

    def _bar(self, bar, key, value):
        """progress bars are redrawn only when the value moved by at least 2 % (otherwise they repaint 30x a second)"""
        v = round(min(max(value, 0.0), 1.0) * 50) / 50
        if self._last.get(key) != v:
            bar.set(v)
            self._last[key] = v

    def _tick(self):
        e = self.engine
        try:
            s = e.snapshot()
            self._update_video(e, s)
            self._update_panels(s)
        except Exception as ex:                          # a drawing problem must never kill the update loop
            print("UI update error:", ex, file=sys.stderr)
        self.after(30, self._tick)

    def _update_video(self, e, s):
        frame = e.frame() if s["running"] else None
        if frame is None:
            if self._photo is not None:
                self._photo = None
                self.video.configure(image="", text="Starting camera…" if s["starting"] else
                                     "Camera is off\n\nClick “Start Camera” to begin")
            return
        vw, vh = max(self.video_box.winfo_width(), 50), max(self.video_box.winfo_height(), 50)
        fh, fw = frame.shape[:2]
        scale = min(vw / fw, vh / fh)
        img = cv2.resize(frame, (max(1, int(fw * scale)), max(1, int(fh * scale))), interpolation=cv2.INTER_AREA)
        self._overlay(img, s)
        self._photo = ImageTk.PhotoImage(Image.fromarray(img))
        self.video.configure(image=self._photo, text="")

    def _overlay(self, img, s):
        """banner / border drawn on the camera picture so the user sees events where they are looking"""
        h, w = img.shape[:2]
        if s["border"]:
            cv2.rectangle(img, (0, 0), (w - 1, h - 1), (59, 130, 246), max(6, w // 80))
        if s["paused"]:
            cv2.putText(img, "PAUSED", (16, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (245, 158, 11), 3, cv2.LINE_AA)
        if s["flash"]:
            msg, sub, hexcol = s["flash"]
            col = tuple(int(hexcol[i:i + 2], 16) for i in (1, 3, 5))
            sc = w / 760
            bw, bh = int(min(w * 0.8, 420 * max(sc, 0.8) * 1.2)), int((92 if sub else 62) * max(sc, 0.8))
            x0, y0 = (w - bw) // 2, int(h * 0.08)
            ov = img.copy()
            cv2.rectangle(ov, (x0, y0), (x0 + bw, y0 + bh), col, -1)
            cv2.addWeighted(ov, 0.88, img, 0.12, 0, img)
            f = cv2.FONT_HERSHEY_SIMPLEX
            tw = cv2.getTextSize(msg, f, 1.0 * max(sc, 0.8), 2)[0][0]
            cv2.putText(img, msg, (x0 + (bw - tw) // 2, y0 + int(bh * (0.62 if not sub else 0.45))), f, 1.0 * max(sc, 0.8), (255, 255, 255), 2, cv2.LINE_AA)
            if sub:
                tw = cv2.getTextSize(sub, f, 0.6 * max(sc, 0.8), 1)[0][0]
                cv2.putText(img, sub, (x0 + (bw - tw) // 2, y0 + int(bh * 0.8)), f, 0.6 * max(sc, 0.8), (255, 255, 255), 1, cv2.LINE_AA)

    def _update_panels(self, s):
        # ---- status pill ----
        if s["error"] and s["error"] != self._shown_error and not (s["running"] or s["starting"]):
            self._shown_error = s["error"]
            self.cam_btn.configure(text=emo("▶  Start Camera"), fg_color=GREEN, hover_color="#16A34A")
            self.cam_index.configure(state="normal")
            messagebox.showerror("Camera", s["error"])
        if s["starting"]:
            self._set_status("Starting camera…", AMBER)
        elif not s["running"]:
            self._set_status("Camera off", "#475569")
        elif s["paused"]:
            self._set_status("Paused", AMBER)
        else:
            self._set_status(emo("● Live"), GREEN)

        # ---- prediction card ----
        live = s["running"] and not s["paused"]
        hand = live and s["hand"]
        if hand:
            lab = s["label"]
            big = {"space": "SPACE", "del": "DEL"}.get(lab, lab)
            self._set(self.big, "big", text=big, text_color=TEXT, font=self.f_letter if len(big) == 1 else self.f_word)
            self._set(self.hand_label, "hand", text=emo("●  Hand detected"), text_color=GREEN)
            c = s["conf"]
            self._bar(self.conf_bar, "confbar", c)
            self._set(self.conf_bar, "confcol", progress_color=GREEN if c >= W.CONF else AMBER)
            self._set(self.conf_label, "conf", text=f"Confidence  {c * 100:.0f}%", text_color=TEXT)
        else:
            self._set(self.big, "big", text="–", text_color=MUTED, font=self.f_letter)
            self._set(self.hand_label, "hand", text=emo("●  No hand detected"), text_color=MUTED if live or not s["running"] else AMBER)
            self._bar(self.conf_bar, "confbar", 0)
            self._set(self.conf_label, "conf", text="Confidence  –", text_color=MUTED)

        if not s["running"]:
            state = "Starting camera…" if s["starting"] else "Camera off"
        elif s["paused"]:
            state = "Paused – recognition is not typing"
        elif not s["hand"]:
            state = "Show your hand to the camera"
        elif s["cand"] is None:
            state = "Not sure yet – keep the sign steady"
        else:
            name = {"space": "SPACE", "del": "DELETE"}.get(s["cand"], s["cand"])
            state = f"Typed {name} – lift hand or change sign" if s["typed"] else f"Hold steady: {name}"
        self._set(self.state_label, "state", text=state)
        self._bar(self.hold_bar, "holdbar", s["progress"] if live else 0)
        self._set(self.hold_bar, "holdcol", progress_color=GREEN if s["typed"] else (AMBER if s["weak"] else BLUE))

        if live and s["top3"]:
            self._set(self.top3_label, "top3", text="Top 3:   " + "     ".join(f"{l} {p * 100:.0f}%" for l, p in s["top3"]))
        else:
            self._set(self.top3_label, "top3", text="")
        self.fps_label.configure(text=f"{s['fps']:.0f} fps" if s["running"] else "")

        # ---- text area: rewritten only when the text changes; the cursor blinks by changing one tag colour ----
        text = s["text"]
        if self._last.get("text") != text:
            self.textbox.configure(state="normal")
            self.textbox.delete("1.0", "end")
            self.textbox.insert("1.0", text)
            self.textbox.insert("end", (emo("▏") or "|"), "cursor")
            self.textbox.configure(state="disabled")
            self.textbox.see("end")
            words = len(text.split())
            self.stats.configure(text=f"{words} word{'s' if words != 1 else ''} · {len(text)} character{'s' if len(text) != 1 else ''}")
            self._last["text"] = text
        blink = int(time.time() * 2) % 2 == 0
        if self._last.get("blink") != blink:
            self.textbox.tag_config("cursor", foreground=TEXT if blink else CARD2)
            self._last["blink"] = blink

        # ---- toast: temporary messages + engine events (changed only when the message changes) ----
        if time.time() < self._toast_until:
            pass                                            # a button message ("Copied...") is still showing
        elif s["flash"]:
            msg, sub, col = s["flash"]
            self._set(self.toast, "toast", text=f"{msg}   {sub}".strip(), text_color=col)
        elif s["speaking"]:
            self._set(self.toast, "toast", text=emo(f'🔊 Speaking: "{s["speaking"]}"'), text_color=BLUE)
        elif s["voice_error"]:
            self._set(self.toast, "toast", text=f"Voice problem: {s['voice_error']}", text_color=RED)
        else:
            self._set(self.toast, "toast", text="", text_color=BLUE)

        # ---- history tab ----
        if self.engine.history_version != self._history_seen:
            self._history_seen = self.engine.history_version
            self.hist.configure(state="normal")
            self.hist.delete("1.0", "end")
            rows = [f"{t}  {'[gesture]' if src == 'gesture' else '[button] '}  {txt}" for t, txt, src in self.engine.history]
            self.hist.insert("1.0", "\n".join(rows) if rows else "Nothing typed yet.")
            self.hist.configure(state="disabled")


if __name__ == "__main__":
    App().mainloop()
