import os
import sys
import time
import wave
import json
import tempfile
import threading
import tkinter as tk
from tkinter import ttk, filedialog, colorchooser, messagebox, font as tkfont
from pathlib import Path

try:
    import sounddevice as sd
    import soundfile as sf
    _HAS_AUDIO = True
except Exception:
    _HAS_AUDIO = False
    sd = None
    sf = None

# Ensure the src directory is on the path so local imports work.
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from video_generator import (
    get_wav_duration,
    parse_lyrics,
    generate_lyrics_timing,
    align_subtitles_to_audio,
    write_srt,
    build_subtitle_clips,
    create_video,
    VIDEO_WIDTH,
    VIDEO_HEIGHT,
    VIDEO_FPS,
    FONT_SIZE as DEFAULT_FONT_SIZE,
    FONT_PATH as DEFAULT_FONT_PATH,
    SUBTITLE_Y_RATIO,
)
from fancy_text import (
    create_word_fancytext_adv,
    create_static_fancytext,
    _load_font,
    _find_font_by_name,
    _build_font_cache,
    _font_cache,
    apply_advanced_effects,
    generate_procedural_texture,
    generate_molten_texture,
)
from audio_fx import (
    DEFAULT_AUDIO_FX as _DEFAULT_AUDIO_FX,
    EQ_BAND_FREQS,
    EQ_BAND_LABELS,
    EQ_NUM_BANDS,
    EQ_PRESETS,
    DEFAULT_EQ_BANDS,
    MELODY_NAMES,
    apply_eq,
    process_audio_clip,
    mix_audio_clips,
    read_wav,
    write_wav,
)

# ── Colour helpers ──────────────────────────────────────────────
def _rgb_to_hex(rgb):
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"

def _hex_to_rgb(hexstr):
    hexstr = hexstr.lstrip("#")
    return (int(hexstr[0:2], 16), int(hexstr[2:4], 16), int(hexstr[4:6], 16))

_VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv", ".webm", ".m4v"}
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}


# ═══════════════════════════════════════════════════════════════
# Custom FillBar widget  (progress-bar-style slider)
# ═══════════════════════════════════════════════════════════════
class FillBar(tk.Canvas):
    """A horizontal fill-bar control that acts like a slider.

    Displays a rounded trough with a lighter-grey filled portion
    representing the current value (0.0 – 1.0 by default).
    """

    def __init__(self, parent, variable=None, from_=0.0, to=1.0,
                 length=120, height=14, command=None, **kw):
        super().__init__(parent, width=length, height=height,
                         bg=kw.pop("bg", C.BG), bd=0,
                         highlightthickness=0, **kw)
        self._var = variable
        self._from = from_
        self._to = to
        self._cmd = command
        self._bar_h = max(6, height - 6)
        self._radius = self._bar_h // 2
        self._dragging = False

        # colors
        self._trough_color = "#1a1a1a"
        self._fill_color = "#6a6a6a"
        self._fill_hover = "#7e7e7e"
        self._border_color = C.BORDER
        self._cur_fill = self._fill_color

        self.bind("<Configure>", self._draw)
        self.bind("<ButtonPress-1>", self._on_press)
        self.bind("<B1-Motion>", self._on_drag)
        self.bind("<ButtonRelease-1>", self._on_release)
        self.bind("<Enter>", lambda e: self._set_hover(True))
        self.bind("<Leave>", lambda e: self._set_hover(False))

        if self._var is not None:
            self._var.trace_add("write", self._var_changed)

    # ── helpers ────────────────────────────────────────────
    def _ratio(self):
        if self._var is None:
            return 0.0
        v = self._var.get()
        span = self._to - self._from
        if span == 0:
            return 0.0
        return max(0.0, min(1.0, (v - self._from) / span))

    def _set_hover(self, on):
        self._cur_fill = self._fill_hover if on else self._fill_color
        self._draw()

    def _var_changed(self, *_):
        self._draw()

    def _x_to_val(self, x):
        w = self.winfo_width()
        pad = self._radius + 2
        usable = w - 2 * pad
        ratio = max(0.0, min(1.0, (x - pad) / max(usable, 1)))
        return self._from + ratio * (self._to - self._from)

    # ── events ─────────────────────────────────────────────
    def _on_press(self, e):
        self._dragging = True
        self._apply(e.x)

    def _on_drag(self, e):
        if self._dragging:
            self._apply(e.x)

    def _on_release(self, e):
        self._dragging = False

    def _apply(self, x):
        val = self._x_to_val(x)
        if self._var is not None:
            self._var.set(round(val, 3))
        if self._cmd:
            self._cmd(val)
        self._draw()

    # ── drawing ────────────────────────────────────────────
    def _draw(self, _event=None):
        self.delete("all")
        w = self.winfo_width()
        h = self.winfo_height()
        if w < 4 or h < 4:
            return
        pad = self._radius + 2
        y0 = (h - self._bar_h) // 2
        y1 = y0 + self._bar_h
        r = self._radius

        # Trough (rounded rect)
        self._rounded_rect(pad, y0, w - pad, y1, r,
                           fill=self._trough_color,
                           outline=self._border_color)

        # Fill portion
        ratio = self._ratio()
        usable = w - 2 * pad
        fill_w = int(usable * ratio)
        if fill_w > 2:
            fx1 = pad
            fx2 = pad + fill_w
            self._rounded_rect(fx1, y0, fx2, y1, r,
                               fill=self._cur_fill, outline="")

        # Knob indicator (small vertical line at fill edge)
        kx = pad + fill_w
        if r < kx < w - r:
            self.create_line(kx, y0 + 2, kx, y1 - 2,
                             fill="#b0b0b0", width=2)

    def _rounded_rect(self, x0, y0, x1, y1, r, **kw):
        """Draw a rounded rectangle on the canvas."""
        pts = [
            x0 + r, y0,
            x1 - r, y0,
            x1, y0,
            x1, y0 + r,
            x1, y1 - r,
            x1, y1,
            x1 - r, y1,
            x0 + r, y1,
            x0, y1,
            x0, y1 - r,
            x0, y0 + r,
            x0, y0,
        ]
        self.create_polygon(pts, smooth=True, **kw)


# ═══════════════════════════════════════════════════════════════
# Dark Theme Palette
# ═══════════════════════════════════════════════════════════════
class _Colors:
    BG              = "#1e1e1e"
    BG_LIGHT        = "#252526"
    BG_MID          = "#333333"
    SURFACE         = "#2d2d2d"
    SURFACE_LIGHT   = "#353535"
    SURFACE_HOVER   = "#3e3e3e"
    BORDER          = "#474747"
    BORDER_LIGHT    = "#555555"
    TEXT            = "#e0e0e0"
    TEXT_DIM        = "#909090"
    TEXT_MUTED      = "#686868"
    ACCENT          = "#e94560"
    ACCENT_HOVER    = "#ff6b81"
    ACCENT_2        = "#3a3a3a"
    SUCCESS         = "#27ae60"
    WARNING         = "#f39c12"
    ERROR           = "#e74c3c"
    RULER_BG        = "#1a1a1a"
    RULER_FG        = "#888888"
    RULER_TICK      = "#5a5a5a"
    TRACK_BG        = "#242424"
    TRACK_BORDER    = "#333333"
    VIDEO_TRACK     = "#1e2e1e"
    VIDEO_CLIP      = "#2d6b4a"
    VIDEO_CLIP_OUT  = "#3d9b6a"
    AUDIO_TRACK     = "#1e2430"
    AUDIO_CLIP      = "#2a5a8e"
    AUDIO_CLIP_OUT  = "#3a7abe"
    SUB_TRACK       = "#302818"
    SUB_CLIP        = "#c89030"
    SUB_CLIP_SEL    = "#f0b848"
    SUB_CLIP_OUT    = "#a07020"
    SUB_CLIP_TEXT   = "#1e1e1e"
    SUB_HANDLE      = "#8B6914"
    LABEL_BG        = "#1a1a1a"
    PREVIEW_BG      = "#0a0a0a"
    PLAYHEAD        = "#e94560"

C = _Colors

# ═══════════════════════════════════════════════════════════════
# Timeline constants
# ═══════════════════════════════════════════════════════════════
TRACK_HEIGHT = 48
TRACK_PADDING = 2
RULER_HEIGHT = 22
TRACK_LABEL_W = 60

# Preview
PREVIEW_W = 640
PREVIEW_H = 360


# ═══════════════════════════════════════════════════════════════
# Dark ttk theme
# ═══════════════════════════════════════════════════════════════
def _apply_dark_theme(root):
    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure(".", background=C.BG, foreground=C.TEXT,
                    bordercolor=C.BORDER, darkcolor=C.BG,
                    lightcolor=C.SURFACE, troughcolor=C.SURFACE,
                    fieldbackground=C.SURFACE, font=("Segoe UI", 9),
                    arrowcolor=C.TEXT_DIM, insertcolor=C.TEXT)
    style.configure("TFrame", background=C.BG)
    style.configure("TLabelframe", background=C.BG, foreground=C.TEXT_DIM,
                    bordercolor=C.BORDER)
    style.configure("TLabelframe.Label", background=C.BG, foreground=C.TEXT_DIM,
                    font=("Segoe UI", 9, "bold"))
    style.configure("TLabel", background=C.BG, foreground=C.TEXT)
    style.configure("Dim.TLabel", foreground=C.TEXT_DIM)
    style.configure("Header.TLabel", font=("Segoe UI", 9, "bold"),
                    foreground=C.TEXT)
    style.configure("TButton", background=C.SURFACE_LIGHT, foreground=C.TEXT,
                    bordercolor=C.BORDER, padding=(8, 3),
                    font=("Segoe UI", 9))
    style.map("TButton",
              background=[("active", C.SURFACE_HOVER), ("pressed", C.BG_MID)],
              foreground=[("active", C.TEXT)])
    style.configure("Accent.TButton", background=C.ACCENT, foreground="#ffffff",
                    bordercolor=C.ACCENT, padding=(10, 4),
                    font=("Segoe UI", 9, "bold"))
    style.map("Accent.TButton",
              background=[("active", C.ACCENT_HOVER), ("pressed", "#cc3344")])
    style.configure("TEntry", fieldbackground=C.SURFACE, foreground=C.TEXT,
                    bordercolor=C.BORDER, insertcolor=C.TEXT, padding=3)
    style.map("TEntry",
              fieldbackground=[("focus", C.SURFACE_LIGHT)],
              bordercolor=[("focus", C.ACCENT)])
    style.configure("TSpinbox", fieldbackground=C.SURFACE, foreground=C.TEXT,
                    bordercolor=C.BORDER, arrowcolor=C.TEXT_DIM,
                    insertcolor=C.TEXT, padding=2)
    style.map("TSpinbox",
              fieldbackground=[("focus", C.SURFACE_LIGHT)],
              bordercolor=[("focus", C.ACCENT)])
    style.configure("TCheckbutton", background=C.BG, foreground=C.TEXT,
                    padding=(2, 2))
    style.map("TCheckbutton",
              background=[("active", C.BG)],
              foreground=[("active", C.TEXT)])
    # Image-based checkbox indicators: grey unchecked, green checkmark when checked
    _cb_size = 14
    _img_unchecked = tk.PhotoImage(width=_cb_size, height=_cb_size)
    _img_checked = tk.PhotoImage(width=_cb_size, height=_cb_size)
    _img_hover = tk.PhotoImage(width=_cb_size, height=_cb_size)
    _img_checked_hover = tk.PhotoImage(width=_cb_size, height=_cb_size)
    # Fill unchecked: dark grey with lighter border
    _img_unchecked.put("#3a3a3a", to=(0, 0, _cb_size, _cb_size))
    _img_unchecked.put("#555555", to=(0, 0, _cb_size, 1))  # top
    _img_unchecked.put("#555555", to=(0, 0, 1, _cb_size))  # left
    _img_unchecked.put("#555555", to=(_cb_size-1, 0, _cb_size, _cb_size))  # right
    _img_unchecked.put("#555555", to=(0, _cb_size-1, _cb_size, _cb_size))  # bottom
    # Fill hover: slightly lighter grey
    _img_hover.put("#454545", to=(0, 0, _cb_size, _cb_size))
    _img_hover.put("#666666", to=(0, 0, _cb_size, 1))
    _img_hover.put("#666666", to=(0, 0, 1, _cb_size))
    _img_hover.put("#666666", to=(_cb_size-1, 0, _cb_size, _cb_size))
    _img_hover.put("#666666", to=(0, _cb_size-1, _cb_size, _cb_size))
    # Checked: dark box with bright green checkmark
    _img_checked.put("#3a3a3a", to=(0, 0, _cb_size, _cb_size))
    _img_checked.put("#555555", to=(0, 0, _cb_size, 1))
    _img_checked.put("#555555", to=(0, 0, 1, _cb_size))
    _img_checked.put("#555555", to=(_cb_size-1, 0, _cb_size, _cb_size))
    _img_checked.put("#555555", to=(0, _cb_size-1, _cb_size, _cb_size))
    # Draw green checkmark on checked (brighter green #4ade80)
    _check_color = "#4ade80"
    # Checkmark path: starts lower-left, goes down to bottom-middle, then up to top-right
    # Using 2px wide lines for visibility
    for dx in range(2):
        _img_checked.put(_check_color, to=(3+dx, 7, 4+dx, 8))
        _img_checked.put(_check_color, to=(4+dx, 8, 5+dx, 9))
        _img_checked.put(_check_color, to=(5+dx, 9, 6+dx, 10))
        _img_checked.put(_check_color, to=(6+dx, 10, 7+dx, 11))
        _img_checked.put(_check_color, to=(7+dx, 9, 8+dx, 10))
        _img_checked.put(_check_color, to=(8+dx, 8, 9+dx, 9))
        _img_checked.put(_check_color, to=(9+dx, 7, 10+dx, 8))
        _img_checked.put(_check_color, to=(10+dx, 6, 11+dx, 7))
        _img_checked.put(_check_color, to=(11+dx, 5, 12+dx, 6))
        _img_checked.put(_check_color, to=(12+dx, 4, 13+dx, 5))
    # Checked hover: lighter background with even brighter checkmark
    _img_checked_hover.put("#454545", to=(0, 0, _cb_size, _cb_size))
    _img_checked_hover.put("#666666", to=(0, 0, _cb_size, 1))
    _img_checked_hover.put("#666666", to=(0, 0, 1, _cb_size))
    _img_checked_hover.put("#666666", to=(_cb_size-1, 0, _cb_size, _cb_size))
    _img_checked_hover.put("#666666", to=(0, _cb_size-1, _cb_size, _cb_size))
    _check_hover = "#86efac"  # Even brighter green on hover
    for dx in range(2):
        _img_checked_hover.put(_check_hover, to=(3+dx, 7, 4+dx, 8))
        _img_checked_hover.put(_check_hover, to=(4+dx, 8, 5+dx, 9))
        _img_checked_hover.put(_check_hover, to=(5+dx, 9, 6+dx, 10))
        _img_checked_hover.put(_check_hover, to=(6+dx, 10, 7+dx, 11))
        _img_checked_hover.put(_check_hover, to=(7+dx, 9, 8+dx, 10))
        _img_checked_hover.put(_check_hover, to=(8+dx, 8, 9+dx, 9))
        _img_checked_hover.put(_check_hover, to=(9+dx, 7, 10+dx, 8))
        _img_checked_hover.put(_check_hover, to=(10+dx, 6, 11+dx, 7))
        _img_checked_hover.put(_check_hover, to=(11+dx, 5, 12+dx, 6))
        _img_checked_hover.put(_check_hover, to=(12+dx, 4, 13+dx, 5))
    # Keep references alive
    root._cb_images = [_img_unchecked, _img_checked, _img_hover, _img_checked_hover]
    style.element_create("custom_indicator", "image", _img_unchecked,
                         ("active selected", _img_checked_hover),
                         ("selected", _img_checked),
                         ("active", _img_hover),
                         sticky="", border=0, padding=1)
    style.layout("TCheckbutton", [
        ("Checkbutton.padding", {"sticky": "nswe", "children": [
            ("custom_indicator", {"side": "left", "sticky": ""}),
            ("Checkbutton.focus", {"side": "left", "sticky": "", "children": [
                ("Checkbutton.label", {"sticky": "nswe"})
            ]})
        ]})
    ])
    style.configure("TScale", background=C.BG, troughcolor=C.SURFACE,
                    bordercolor=C.BORDER)
    style.configure("Horizontal.TScale", sliderlength=14)
    style.configure("TScrollbar", background=C.SURFACE, troughcolor=C.BG,
                    bordercolor=C.BG, arrowcolor=C.TEXT_DIM)
    style.map("TScrollbar", background=[("active", C.SURFACE_HOVER)])
    style.configure("TCombobox", fieldbackground=C.SURFACE,
                    background=C.SURFACE_LIGHT, foreground=C.TEXT,
                    bordercolor=C.BORDER, arrowcolor=C.TEXT_DIM,
                    insertcolor=C.TEXT, padding=2)
    style.map("TCombobox",
              fieldbackground=[("focus", C.SURFACE_LIGHT),
                               ("readonly", C.SURFACE)],
              foreground=[("readonly", C.TEXT)],
              selectbackground=[("readonly", C.SURFACE)],
              selectforeground=[("readonly", C.TEXT)],
              bordercolor=[("focus", C.ACCENT)])
    # Dark dropdown list for Combobox popdown
    root.option_add("*TCombobox*Listbox.background", C.SURFACE)
    root.option_add("*TCombobox*Listbox.foreground", C.TEXT)
    root.option_add("*TCombobox*Listbox.selectBackground", C.ACCENT)
    root.option_add("*TCombobox*Listbox.selectForeground", "#ffffff")
    root.option_add("*TCombobox*Listbox.borderWidth", "0")
    style.configure("TSeparator", background=C.BORDER)
    style.configure("TPanedwindow", background=C.BG)
    style.configure("Sash", sashthickness=6, gripcount=0,
                    sashrelief="flat", background=C.BORDER)
    return style


# ═══════════════════════════════════════════════════════════════
# Subtitle clip model
# ═══════════════════════════════════════════════════════════════
class SubtitleClip:
    __slots__ = ("text", "start", "end", "idx", "font_size")
    def __init__(self, text, start, end, idx=0, font_size=None):
        self.text = text
        self.start = start
        self.end = end
        self.idx = idx
        self.font_size = font_size  # None means use global default

    @property
    def duration(self):
        return self.end - self.start

    def as_dict(self):
        d = {"text": self.text, "start": self.start, "end": self.end}
        if self.font_size is not None:
            d["font_size"] = self.font_size
        return d


# ═══════════════════════════════════════════════════════════════
# Media clip model (for video/audio track segments)
# ═══════════════════════════════════════════════════════════════
class MediaClip:
    """Represents a segment of a video/audio file placed on the timeline."""
    __slots__ = ("source_path", "source_type", "start", "end",
                 "source_offset", "source_duration", "audio_effects",
                 "track_index")

    def __init__(self, source_path, source_type, start, end,
                 source_offset=0.0, source_duration=0.0,
                 audio_effects=None, track_index=0):
        self.source_path = source_path      # Path to media file
        self.source_type = source_type      # "video", "image", or "audio"
        self.start = start                  # Timeline start position
        self.end = end                      # Timeline end position
        self.source_offset = source_offset  # Offset within source file
        self.source_duration = source_duration  # Full duration of source
        self.audio_effects = dict(_DEFAULT_AUDIO_FX) if audio_effects is None else dict(audio_effects)
        self.track_index = track_index      # Which audio track layer (0-based)

    @property
    def duration(self):
        return self.end - self.start

    def as_dict(self):
        return {
            "source_path": self.source_path,
            "source_type": self.source_type,
            "start": self.start,
            "end": self.end,
            "source_offset": self.source_offset,
            "source_duration": self.source_duration,
            "audio_effects": dict(self.audio_effects),
            "track_index": self.track_index,
        }


# ═══════════════════════════════════════════════════════════════
# Built-in style presets (defaults seeded into presets.json)
# ═══════════════════════════════════════════════════════════════
_IMPACT_FONT = r"C:\Windows\Fonts\impact.ttf"

_DEFAULT_PRESETS = {
    "Default White": {
        "font_path": _IMPACT_FONT,
        "font_size": 85, "text_case": "uppercase", "text_color": [255, 255, 255],
        "highlight_color": [255, 255, 0],
        "stroke_enabled": True, "stroke_color": [0, 0, 0], "stroke_width": 2,
        "shadow_enabled": True, "shadow_offset": [15, 15], "shadow_blur": 20,
        "highlight_box_enabled": False,
        "glow_enabled": False, "gradient_enabled": False,
        "bevel_enabled": False, "chrome_enabled": False,
        "max_words_per_line": 6, "position_y_ratio": 0.85,
        "highlight_mode": "word",
    },
    "Karaoke Classic": {
        "font_path": _IMPACT_FONT,
        "font_size": 85, "text_case": "uppercase", "text_color": [255, 255, 255],
        "highlight_color": [0, 200, 255],
        "stroke_enabled": True, "stroke_color": [0, 0, 80], "stroke_width": 3,
        "shadow_enabled": True, "shadow_offset": [3, 3], "shadow_blur": 5,
        "highlight_box_enabled": False,
        "glow_enabled": True, "glow_color": [0, 150, 255], "glow_size": 8,
        "gradient_enabled": False,
        "bevel_enabled": False, "chrome_enabled": False,
        "max_words_per_line": 5, "position_y_ratio": 0.82,
        "highlight_mode": "word",
    },
    "Neon Glow": {
        "font_path": _IMPACT_FONT,
        "font_size": 50, "text_case": "uppercase", "text_color": [0, 255, 180],
        "highlight_color": [255, 0, 255],
        "stroke_enabled": True, "stroke_color": [0, 60, 40], "stroke_width": 2,
        "shadow_enabled": False,
        "highlight_box_enabled": False,
        "glow_enabled": True, "glow_color": [0, 255, 180], "glow_size": 14,
        "gradient_enabled": False,
        "bevel_enabled": False, "chrome_enabled": False,
        "max_words_per_line": 5, "position_y_ratio": 0.85,
        "highlight_mode": "character",
    },
    "Cinema Gold": {
        "font_path": _IMPACT_FONT,
        "font_size": 46, "text_case": "uppercase", "text_color": [255, 215, 0],
        "highlight_color": [255, 255, 255],
        "stroke_enabled": True, "stroke_color": [80, 50, 0], "stroke_width": 2,
        "shadow_enabled": True, "shadow_offset": [2, 3], "shadow_blur": 6,
        "highlight_box_enabled": False,
        "glow_enabled": False,
        "gradient_enabled": True, "gradient_color1": [255, 215, 0],
        "gradient_color2": [200, 120, 0], "gradient_opacity": 0.8,
        "bevel_enabled": True, "bevel_depth": 3,
        "chrome_enabled": False,
        "max_words_per_line": 6, "position_y_ratio": 0.88,
        "highlight_mode": "word",
    },
    "Minimal Clean": {
        "font_path": _IMPACT_FONT,
        "font_size": 40, "text_case": "normal", "text_color": [240, 240, 240],
        "highlight_color": [100, 200, 255],
        "stroke_enabled": False,
        "shadow_enabled": True, "shadow_offset": [1, 1], "shadow_blur": 2,
        "highlight_box_enabled": False,
        "glow_enabled": False, "gradient_enabled": False,
        "bevel_enabled": False, "chrome_enabled": False,
        "max_words_per_line": 7, "position_y_ratio": 0.90,
        "highlight_mode": "word",
    },
    "Bold Red": {
        "font_path": _IMPACT_FONT,
        "font_size": 56, "text_case": "uppercase", "text_color": [255, 40, 40],
        "highlight_color": [255, 255, 0],
        "stroke_enabled": True, "stroke_color": [0, 0, 0], "stroke_width": 4,
        "shadow_enabled": True, "shadow_offset": [3, 3], "shadow_blur": 4,
        "highlight_box_enabled": False,
        "glow_enabled": False, "gradient_enabled": False,
        "bevel_enabled": False, "chrome_enabled": False,
        "max_words_per_line": 4, "position_y_ratio": 0.85,
        "highlight_mode": "word",
    },
    "Molten Text": {
        "font_path": _IMPACT_FONT,
        "font_size": 65, "text_case": "uppercase", "text_color": [100, 30, 10],
        "highlight_color": [255, 100, 30],
        "stroke_enabled": True, "stroke_color": [20, 5, 0], "stroke_width": 2,
        "shadow_enabled": False,
        "highlight_box_enabled": False,
        "glow_enabled": True, "glow_color": [255, 80, 20], "glow_size": 10,
        "glow_opacity": 150,
        "gradient_enabled": False,
        "bevel_enabled": True, "bevel_depth": 7,
        "chrome_enabled": False, "texture_enabled": True, "texture_scale": 1.0,
        "texture_opacity": 0.85,
        "max_words_per_line": 5, "position_y_ratio": 0.72,
        "highlight_mode": "character",
    },
}

_PRESETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "presets")


def _ensure_preset_defaults():
    """Seed the presets folder with built-in defaults if they don't exist."""
    os.makedirs(_PRESETS_DIR, exist_ok=True)
    for name, data in _DEFAULT_PRESETS.items():
        path = os.path.join(_PRESETS_DIR, f"{name}.json")
        if not os.path.isfile(path):
            try:
                with open(path, "w", encoding="utf-8") as fp:
                    json.dump(data, fp, indent=2)
            except Exception:
                pass


def _load_all_presets():
    """Scan the presets folder and return {name: data} for every .json file."""
    _ensure_preset_defaults()
    presets = {}
    for fname in os.listdir(_PRESETS_DIR):
        if not fname.lower().endswith(".json"):
            continue
        path = os.path.join(_PRESETS_DIR, fname)
        try:
            with open(path, "r", encoding="utf-8") as fp:
                presets[os.path.splitext(fname)[0]] = json.load(fp)
        except Exception:
            pass
    return presets


# ═══════════════════════════════════════════════════════════════
# Main Application
# ═══════════════════════════════════════════════════════════════
class SubtitleEditorApp(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("SubEditor \u2014 Subtitle Video Editor")
        self.geometry("1400x780")
        self.state("zoomed")
        self.configure(bg=C.BG)
        self.minsize(900, 600)
        _apply_dark_theme(self)

        # Unbind space from buttons/checkbuttons to prevent activation
        # (space should only work in text entries, otherwise play/pause)
        self.unbind_class("TButton", "<space>")
        self.unbind_class("Button", "<space>")
        self.unbind_class("TCheckbutton", "<space>")
        self.unbind_class("Checkbutton", "<space>")
        self.unbind_class("TRadiobutton", "<space>")
        self.unbind_class("Radiobutton", "<space>")

        # ── State ──
        self.audio_path: str | None = None
        self.lyrics_path: str | None = None
        self.video_source_path: str | None = None
        self.video_source_type: str | None = None
        self.video_source_duration: float = 0.0
        self.audio_duration: float = 0.0
        self.subtitles: list[SubtitleClip] = []
        self.video_clips: list[MediaClip] = []  # Video track segments
        self.audio_clips: list[MediaClip] = []  # Audio track segments (multi-layer)
        self._video_track_count = 1  # Number of video tracks
        self._audio_track_count = 1  # Number of audio tracks
        self.selected_clip: SubtitleClip | None = None
        self.selected_media_clip: MediaClip | None = None  # Currently selected video/audio clip
        self._selected_track: str | None = None  # "video", "audio", "subtitle", or None
        self._clipboard_clip: dict | None = None  # copied clip data
        self._multi_selected: set = set()  # Multi-selected clips (any type)

        # Drag state
        self._drag_mode = None
        self._drag_clip = None
        self._drag_start_x = 0
        self._drag_orig_start = 0.0
        self._drag_orig_end = 0.0
        self._drag_multi_orig = {}  # {clip: (orig_start, orig_end)} for multi-drag

        # Box selection state
        self._box_select_start = None  # (x, y) start of box selection
        self._box_select_rect = None   # Canvas item ID for selection rectangle

        # Zoom / scroll
        self._px_per_sec = 60.0
        self._scroll_x = 0.0

        # Playback state
        self._playing = False
        self._playback_time = 0.0
        self._playback_speed = 1.0
        self._playback_after_id = None
        self._playback_wall_start = 0.0   # time.perf_counter at play start
        self._playback_pos_start = 0.0    # _playback_time at play start
        self._video_clip = None        # moviepy VideoFileClip for preview
        self._bg_image_pil = None      # PIL Image for image bg
        self._preview_photo = None     # current tk PhotoImage

        # Track preview caches
        self._video_thumbnails = []    # list of (time, PhotoImage) for video track
        self._video_thumb_strip = None # single PhotoImage for image bg track
        self._audio_waveforms = {}     # {source_path: list of (frac, amplitude)} per-clip cache
        self._waveform_images = {}     # {cache_key: PhotoImage} rendered waveform images
        self._processed_audio_cache = {}  # {clip_cache_key: (samples, sr, nch)} processed audio

        # Resize debounce state
        self._preview_resize_after = None
        self._timeline_resize_after = None

        # Style defaults
        self._font_path = _IMPACT_FONT
        self._font_size = DEFAULT_FONT_SIZE
        self._text_case = "normal"  # "uppercase", "normal", "lowercase"
        self._text_color = (255, 255, 255)
        self._highlight_color = (220, 20, 20)
        self._stroke_color = (0, 0, 0)
        self._stroke_width = 3
        self._stroke_enabled = True
        self._shadow_enabled = True
        self._shadow_offset = (15, 15)
        self._shadow_blur = 20
        self._highlight_box_enabled = False
        self._highlight_box_color = (0, 0, 255)
        self._highlight_box_opacity = 0.3
        self._max_words_per_line = 6
        self._position_x_ratio = 0.5
        self._position_y_ratio = SUBTITLE_Y_RATIO
        self._text_justify = "center"  # "left", "center", "right"
        self._highlight_mode = "word"  # "word" or "character"
        self._glow_enabled = False
        self._glow_color = (255, 255, 0)
        self._glow_size = 6
        self._glow_opacity = 60  # 0-300, default 60 (interpreted as percentage)
        self._gradient_enabled = False
        self._gradient_color1 = (255, 215, 0)
        self._gradient_color2 = (255, 140, 0)
        self._gradient_opacity = 0.8
        self._bevel_enabled = False
        self._bevel_depth = 3
        self._chrome_enabled = False
        self._chrome_opacity = 0.9
        self._texture_enabled = False
        self._texture_scale = 1.0
        self._texture_opacity = 0.85
        self._texture_blend_mode = "screen"  # screen works well for bright textures
        self._texture_path = None  # path to loaded texture file
        # Generate default molten lava texture
        self._texture_image = generate_molten_texture(width=256, height=256, seed=77)

        self._build_ui()
        self._bind_shortcuts()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _on_close(self):
        self._audio_cleanup()
        self.destroy()

    # ────────────────────────────────────────────────────────────
    @property
    def _total_track_count(self):
        """N video + N audio + 1 subtitle."""
        return self._video_track_count + self._audio_track_count + 1

    def _update_timeline_height(self):
        h = RULER_HEIGHT + self._total_track_count * (TRACK_HEIGHT + TRACK_PADDING) + 20
        self._timeline_frame.configure(height=h)

    @property
    def timeline_duration(self):
        candidates = [10.0]
        if self.audio_duration > 0:
            candidates.append(self.audio_duration)
        if self.video_source_type == "video" and self.video_source_duration > 0:
            candidates.append(self.video_source_duration)
        if self.subtitles:
            candidates.append(max(c.end for c in self.subtitles) + 1.0)
        if self.video_clips:
            candidates.append(max(c.end for c in self.video_clips))
        if self.audio_clips:
            candidates.append(max(c.end for c in self.audio_clips))
        return max(candidates)

    # ────────────────────────────────────────────────────────────
    # UI Construction
    # ────────────────────────────────────────────────────────────
    def _build_ui(self):
        self._build_menu()

        # ── Main vertical paned: top section + timeline ────────
        main_vpane = ttk.PanedWindow(self, orient=tk.VERTICAL)
        main_vpane.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)

        # ── Top section container ──────────────────────────────
        top_container = tk.Frame(main_vpane, bg=C.BG)

        # ── Top section: style panel (left) + preview (right) ──
        top_paned = ttk.PanedWindow(top_container, orient=tk.HORIZONTAL)
        top_paned.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)

        left_container = self._build_left_panel(top_paned)
        top_paned.add(left_container, weight=0)

        preview_frame = self._build_preview_panel(top_paned)
        top_paned.add(preview_frame, weight=1)

        main_vpane.add(top_container, weight=1)

        # ── Bottom section container (toolbar + timeline) ──────
        bottom_container = tk.Frame(main_vpane, bg=C.BG)

        # ── Toolbar row (just above timeline) ──────────────────
        self._build_toolbar(bottom_container)

        # ── Timeline (compact tracks) ──────────────────────────
        self._timeline_frame = tk.Frame(bottom_container, bg=C.BG)
        self._timeline_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=(0, 2))
        self._build_timeline(self._timeline_frame)

        main_vpane.add(bottom_container, weight=1)

        # ── Detail panel ─────────────────────────────────────────
        detail_frame = ttk.LabelFrame(self, text=" Selected Clip ")
        detail_frame.pack(fill=tk.X, padx=4, pady=(0, 2))
        self._build_detail_panel(detail_frame)

        # ── Status bar ─────────────────────────────────────────
        status_bar = tk.Frame(self, bg=C.SURFACE, height=22)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)
        status_bar.pack_propagate(False)
        self._status_var = tk.StringVar(value="Ready")
        tk.Label(status_bar, textvariable=self._status_var,
                 bg=C.SURFACE, fg=C.TEXT_DIM,
                 font=("Segoe UI", 8), anchor="w",
                 padx=8).pack(fill=tk.X, expand=True)

    # ── Menu bar (custom dark frame-based) ────────────────────
    def _build_menu(self):
        menu_bar = tk.Frame(self, bg=C.SURFACE, height=26)
        menu_bar.pack(fill=tk.X, side=tk.TOP)
        menu_bar.pack_propagate(False)

        _menu_kw = dict(
            tearoff=0, bg=C.SURFACE, fg=C.TEXT,
            activebackground=C.ACCENT, activeforeground="#fff",
            relief="flat", bd=0, font=("Segoe UI", 9),
        )
        _btn_kw = dict(
            bg=C.SURFACE, fg=C.TEXT_DIM, activebackground=C.SURFACE_HOVER,
            activeforeground=C.TEXT, relief="flat", bd=0, padx=10, pady=3,
            font=("Segoe UI", 9), highlightthickness=0, cursor="hand2",
            direction="below",
        )

        # ── File menu ──
        file_btn = tk.Menubutton(menu_bar, text="File", **_btn_kw)
        file_btn.pack(side=tk.LEFT)
        fm = tk.Menu(file_btn, **_menu_kw)
        fm.add_command(label="  New Project",
                       command=self._new_project, accelerator="Ctrl+N")
        fm.add_command(label="  Open Project\u2026",
                       command=self._open_project, accelerator="Ctrl+O")
        fm.add_command(label="  Save Project",
                       command=self._save_project, accelerator="Ctrl+S")
        fm.add_command(label="  Save Project As\u2026",
                       command=self._save_project_as, accelerator="Ctrl+Shift+S")
        fm.add_separator()
        fm.add_command(label="  Open Audio (.wav)\u2026",
                       command=self._open_audio)
        fm.add_command(label="  Add Audio Layer\u2026",
                       command=self._add_audio_layer)
        fm.add_command(label="  Open Lyrics (.txt)\u2026",
                       command=self._open_lyrics)
        fm.add_command(label="  Open Video / Image\u2026",
                       command=self._open_video_source)
        fm.add_separator()
        fm.add_command(label="  Import SRT\u2026", command=self._import_srt)
        fm.add_command(label="  Export SRT\u2026",
                       command=self._export_srt)
        fm.add_separator()
        fm.add_command(label="  Export Video\u2026", command=self._export_video)
        fm.add_separator()
        fm.add_command(label="  Exit", command=self.destroy)
        file_btn["menu"] = fm

        # ── Edit menu ──
        edit_btn = tk.Menubutton(menu_bar, text="Edit", **_btn_kw)
        edit_btn.pack(side=tk.LEFT)
        em = tk.Menu(edit_btn, **_menu_kw)
        em.add_command(label="  Delete Selected Clip",
                       command=self._delete_selected, accelerator="Del")
        em.add_command(label="  Split Clip at Playhead",
                       command=self._split_selected, accelerator="S")
        em.add_separator()
        em.add_command(label="  Clear All Subtitles",
                       command=self._clear_subtitles)
        em.add_command(label="  Clear Video Track",
                       command=self._clear_video_track)
        em.add_command(label="  Clear Audio Track",
                       command=self._clear_audio_track)
        em.add_separator()
        em.add_command(label="  Add Video Track",
                       command=self._add_video_track)
        em.add_command(label="  Remove Video Track",
                       command=self._remove_video_track)
        edit_btn["menu"] = em

        # ── Effects menu ──
        fx_btn = tk.Menubutton(menu_bar, text="Effects", **_btn_kw)
        fx_btn.pack(side=tk.LEFT)
        fxm = tk.Menu(fx_btn, **_menu_kw)
        fxm.add_command(label="  Audio Effects\u2026",
                        command=self._show_audio_fx_dialog)
        fx_btn["menu"] = fxm

        # Thin bottom border
        tk.Frame(self, bg=C.BORDER, height=1).pack(fill=tk.X, side=tk.TOP)

    # ── Toolbar (above timeline) ───────────────────────────────
    def _build_toolbar(self, parent=None):
        if parent is None:
            parent = self
        tb = tk.Frame(parent, bg=C.SURFACE, padx=6, pady=3)
        tb.pack(fill=tk.X, padx=0, pady=(2, 0))

        btn_kw = dict(bg=C.SURFACE_LIGHT, fg=C.TEXT, relief="flat",
                      font=("Segoe UI", 8), padx=8, pady=2,
                      activebackground=C.SURFACE_HOVER,
                      activeforeground=C.TEXT,
                      cursor="hand2", bd=0, highlightthickness=0)

        tk.Button(tb, text="\U0001F3B5 Audio",
                  command=self._open_audio, **btn_kw).pack(
            side=tk.LEFT, padx=(0, 2))
        tk.Button(tb, text="\U0001F4DD Lyrics",
                  command=self._open_lyrics, **btn_kw).pack(
            side=tk.LEFT, padx=2)
        tk.Button(tb, text="\U0001F3AC Video/Image",
                  command=self._open_video_source, **btn_kw).pack(
            side=tk.LEFT, padx=2)

        tk.Frame(tb, bg=C.BORDER, width=1).pack(
            side=tk.LEFT, fill=tk.Y, padx=6, pady=1)

        tk.Button(tb, text="\u2795 Insert Clip",
                  command=self._insert_clip, **btn_kw).pack(
            side=tk.LEFT, padx=2)

        tk.Frame(tb, bg=C.BORDER, width=1).pack(
            side=tk.LEFT, fill=tk.Y, padx=6, pady=1)

        # Subtitle generation controls
        tk.Label(tb, text="Vocal Start:", bg=C.SURFACE, fg=C.TEXT_DIM,
                 font=("Segoe UI", 7)).pack(side=tk.LEFT, padx=(0, 2))
        self._vocal_start_var = tk.StringVar(value="0.0")
        vocal_entry = tk.Entry(tb, textvariable=self._vocal_start_var,
                               width=5, bg=C.SURFACE_LIGHT, fg=C.TEXT,
                               insertbackground=C.TEXT, relief="flat",
                               font=("Segoe UI", 8))
        vocal_entry.pack(side=tk.LEFT, padx=(0, 4))
        self._vocal_entry = vocal_entry  # Store reference for unfocusing
        # Space unfocuses and triggers play/pause
        vocal_entry.bind("<space>", lambda e: (self._tl_canvas.focus_set(), self._play_pause(), "break")[-1])
        # Clicking anywhere on toolbar (buttons, etc.) unfocuses the entry
        tb.bind("<Button-1>", lambda e: self._tl_canvas.focus_set() if e.widget != vocal_entry else None, add="+")

        tk.Button(tb, text="\U0001F3A4 Generate Subs",
                  command=self._generate_subtitles, **btn_kw).pack(
            side=tk.LEFT, padx=2)
        tk.Button(tb, text="\U0001F5D1 Clear Subs",
                  command=self._clear_subtitles, **btn_kw).pack(
            side=tk.LEFT, padx=2)

        tk.Frame(tb, bg=C.BORDER, width=1).pack(
            side=tk.LEFT, fill=tk.Y, padx=6, pady=1)

        tk.Button(tb, text="\U0001F4F9 Export Video",
                  command=self._export_video,
                  bg=C.SUCCESS, fg="#fff", relief="flat",
                  font=("Segoe UI", 8, "bold"), padx=10, pady=2,
                  activebackground="#2ecc71",
                  activeforeground="#fff", cursor="hand2",
                  bd=0, highlightthickness=0).pack(
            side=tk.LEFT, padx=2)

        tk.Frame(tb, bg=C.BORDER, width=1).pack(
            side=tk.LEFT, fill=tk.Y, padx=6, pady=1)

        # Zoom
        tk.Label(tb, text="Zoom", bg=C.SURFACE, fg=C.TEXT_DIM,
                 font=("Segoe UI", 7)).pack(side=tk.LEFT, padx=(0, 2))
        self._zoom_var = tk.DoubleVar(value=self._px_per_sec)
        ttk.Scale(tb, from_=10, to=150, variable=self._zoom_var,
                  orient=tk.HORIZONTAL, length=90,
                  command=self._on_zoom_change).pack(side=tk.LEFT)

        # File info (right side)
        info = tk.Frame(tb, bg=C.SURFACE)
        info.pack(side=tk.RIGHT)
        self._lbl_video = tk.Label(info, text="Video: \u2014",
                                   bg=C.SURFACE, fg=C.TEXT_DIM,
                                   font=("Segoe UI", 7))
        self._lbl_video.pack(side=tk.RIGHT, padx=(8, 0))
        self._lbl_audio = tk.Label(info, text="Audio: \u2014",
                                   bg=C.SURFACE, fg=C.TEXT_DIM,
                                   font=("Segoe UI", 7))
        self._lbl_audio.pack(side=tk.RIGHT, padx=(8, 0))
        self._lbl_lyrics = tk.Label(info, text="Lyrics: \u2014",
                                    bg=C.SURFACE, fg=C.TEXT_DIM,
                                    font=("Segoe UI", 7))
        self._lbl_lyrics.pack(side=tk.RIGHT, padx=(8, 0))

    # ── Preview Panel ──────────────────────────────────────────
    def _build_preview_panel(self, parent):
        outer = ttk.LabelFrame(parent, text=" Preview ")

        # Preview canvas
        self._preview_canvas = tk.Canvas(
            outer, bg=C.PREVIEW_BG, highlightthickness=0,
            width=PREVIEW_W, height=PREVIEW_H)
        self._preview_canvas.pack(fill=tk.BOTH, expand=True,
                                  padx=2, pady=(2, 0))
        self._preview_canvas.bind("<Configure>", self._on_preview_resize)
        self._preview_canvas.bind("<ButtonPress-1>", self._on_preview_click)

        # "No preview" text
        self._preview_canvas.create_text(
            PREVIEW_W // 2, PREVIEW_H // 2,
            text="No media loaded", fill=C.TEXT_MUTED,
            font=("Segoe UI", 12), tags="placeholder")

        # Transport controls
        transport = tk.Frame(outer, bg=C.SURFACE, pady=3, padx=4)
        transport.pack(fill=tk.X, padx=2, pady=(2, 2))

        tb_kw = dict(bg=C.SURFACE_LIGHT, fg=C.TEXT, relief="flat",
                     font=("Segoe UI", 9), padx=6, pady=1,
                     activebackground=C.SURFACE_HOVER,
                     activeforeground=C.TEXT,
                     cursor="hand2", bd=0, highlightthickness=0)

        # Rewind
        tk.Button(transport, text="\u23EE", command=self._rewind,
                  **tb_kw).pack(side=tk.LEFT, padx=1)
        # Step back
        tk.Button(transport, text="\u23EA", command=self._step_back,
                  **tb_kw).pack(side=tk.LEFT, padx=1)
        # Play/Pause
        self._play_btn = tk.Button(transport, text="\u25B6",
                                   command=self._play_pause, **tb_kw)
        self._play_btn.pack(side=tk.LEFT, padx=1)
        # Step forward
        tk.Button(transport, text="\u23E9", command=self._step_forward,
                  **tb_kw).pack(side=tk.LEFT, padx=1)

        # Time display
        self._time_var = tk.StringVar(value="0:00.00 / 0:00.00")
        tk.Label(transport, textvariable=self._time_var,
                 bg=C.SURFACE, fg=C.TEXT_DIM,
                 font=("Consolas", 8)).pack(side=tk.LEFT, padx=(8, 4))

        # Seek bar
        self._seek_var = tk.DoubleVar(value=0.0)
        self._seek_bar = ttk.Scale(transport, from_=0.0, to=1.0,
                                   variable=self._seek_var,
                                   orient=tk.HORIZONTAL,
                                   command=self._on_seek)
        self._seek_bar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)

        # Speed
        tk.Label(transport, text="Speed:", bg=C.SURFACE, fg=C.TEXT_DIM,
                 font=("Segoe UI", 7)).pack(side=tk.LEFT, padx=(6, 2))
        self._speed_var = tk.StringVar(value="1.0x")
        speed_menu = tk.Menubutton(transport, textvariable=self._speed_var,
                                   bg=C.SURFACE_LIGHT, fg=C.TEXT,
                                   relief="flat", font=("Segoe UI", 8),
                                   activebackground=C.SURFACE_HOVER,
                                   activeforeground=C.TEXT,
                                   cursor="hand2", bd=0,
                                   highlightthickness=0, padx=4)
        speed_menu.pack(side=tk.LEFT, padx=1)
        sm = tk.Menu(speed_menu, tearoff=0, bg=C.SURFACE, fg=C.TEXT,
                     activebackground=C.ACCENT, activeforeground="#fff")
        for spd in [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0]:
            sm.add_command(label=f"{spd}x",
                           command=lambda s=spd: self._set_speed(s))
        speed_menu.config(menu=sm)

        return outer

    # ── Left Panel (Style / EQ tabs) ─────────────────────────────
    def _build_left_panel(self, parent):
        """Container with Style / EQ tab buttons at top."""
        container = tk.Frame(parent, bg=C.BG, width=210)

        # ── Tab bar ──
        tab_bar = tk.Frame(container, bg=C.SURFACE, height=28)
        tab_bar.pack(fill=tk.X, side=tk.TOP)
        tab_bar.pack_propagate(False)

        self._left_tab_var = tk.StringVar(value="style")

        tab_kw = dict(
            relief="flat", bd=0, padx=14, pady=3,
            font=("Segoe UI", 9, "bold"), highlightthickness=0,
            cursor="hand2",
        )
        self._style_tab_btn = tk.Button(
            tab_bar, text="Style",
            bg=C.ACCENT, fg="#fff",
            activebackground=C.ACCENT_HOVER, activeforeground="#fff",
            command=lambda: self._switch_left_tab("style"), **tab_kw)
        self._style_tab_btn.pack(side=tk.LEFT, padx=(2, 0), pady=2)

        self._eq_tab_btn = tk.Button(
            tab_bar, text="EQ",
            bg=C.SURFACE_HOVER, fg=C.TEXT_DIM,
            activebackground=C.SURFACE_HOVER, activeforeground=C.TEXT,
            command=lambda: self._switch_left_tab("eq"), **tab_kw)
        self._eq_tab_btn.pack(side=tk.LEFT, padx=(2, 0), pady=2)

        # ── Content frames ──
        self._left_content = tk.Frame(container, bg=C.BG)
        self._left_content.pack(fill=tk.BOTH, expand=True)

        self._style_panel = self._build_style_panel(self._left_content)
        self._style_panel.pack(fill=tk.BOTH, expand=True)

        self._eq_panel = self._build_eq_panel(self._left_content)
        # EQ panel starts hidden

        return container

    def _switch_left_tab(self, tab):
        self._left_tab_var.set(tab)
        if tab == "style":
            self._eq_panel.pack_forget()
            self._style_panel.pack(fill=tk.BOTH, expand=True)
            self._style_tab_btn.config(bg=C.ACCENT, fg="#fff")
            self._eq_tab_btn.config(bg=C.SURFACE_HOVER, fg=C.TEXT_DIM)
        else:
            self._style_panel.pack_forget()
            self._eq_panel.pack(fill=tk.BOTH, expand=True)
            self._eq_tab_btn.config(bg=C.ACCENT, fg="#fff")
            self._style_tab_btn.config(bg=C.SURFACE_HOVER, fg=C.TEXT_DIM)
            self._refresh_eq_panel()

    # ── EQ Panel ───────────────────────────────────────────────
    def _build_eq_panel(self, parent):
        outer = tk.Frame(parent, bg=C.BG, width=210)

        SML = ("Segoe UI", 8)
        TITLE = ("Segoe UI", 9, "bold")
        DIM = {"bg": C.BG, "fg": C.TEXT_DIM, "font": SML}
        HDR = {"bg": C.BG, "fg": C.TEXT, "font": TITLE}

        # ── Header row: Title + scope radio ──
        hdr = tk.Frame(outer, bg=C.BG)
        hdr.pack(fill=tk.X, padx=6, pady=(8, 2))
        tk.Label(hdr, text="10-Band EQ", **HDR).pack(side=tk.LEFT)

        self._eq_scope_var = tk.StringVar(value="clip")
        tk.Radiobutton(hdr, text="Clip", variable=self._eq_scope_var,
                       value="clip", bg=C.BG, fg=C.TEXT, font=SML,
                       selectcolor=C.SURFACE, activebackground=C.BG,
                       activeforeground=C.TEXT, highlightthickness=0,
                       command=self._refresh_eq_panel).pack(
            side=tk.RIGHT, padx=(4, 0))
        tk.Radiobutton(hdr, text="Project", variable=self._eq_scope_var,
                       value="project", bg=C.BG, fg=C.TEXT, font=SML,
                       selectcolor=C.SURFACE, activebackground=C.BG,
                       activeforeground=C.TEXT, highlightthickness=0,
                       command=self._refresh_eq_panel).pack(
            side=tk.RIGHT, padx=(4, 0))

        # ── Preset row ──
        preset_row = tk.Frame(outer, bg=C.BG)
        preset_row.pack(fill=tk.X, padx=6, pady=(2, 4))
        tk.Label(preset_row, text="Preset:", **DIM).pack(side=tk.LEFT)
        self._eq_preset_var = tk.StringVar(value="Flat")
        eq_combo = ttk.Combobox(
            preset_row, textvariable=self._eq_preset_var,
            values=list(EQ_PRESETS.keys()), state="readonly",
            width=14, font=SML)
        eq_combo.pack(side=tk.LEFT, padx=(4, 4))
        eq_combo.bind("<<ComboboxSelected>>", self._on_eq_preset)

        ttk.Button(preset_row, text="Reset",
                   command=self._reset_eq).pack(side=tk.RIGHT)

        # ── Separator ──
        tk.Frame(outer, bg=C.BORDER, height=1).pack(
            fill=tk.X, padx=6, pady=2)

        # ── Sliders frame ──
        sliders_frame = tk.Frame(outer, bg=C.BG)
        sliders_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        self._eq_band_vars = []
        self._eq_band_sliders = []
        self._eq_band_labels = []

        # dB labels column on left
        db_labels = ["+12", "+6", "0", "-6", "-12"]
        db_label_frame = tk.Frame(sliders_frame, bg=C.BG, width=24)
        db_label_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 2))
        db_label_frame.pack_propagate(False)
        for i, dbt in enumerate(db_labels):
            tk.Label(db_label_frame, text=dbt, bg=C.BG,
                     fg=C.TEXT_MUTED, font=("Segoe UI", 7),
                     anchor="e").place(
                relx=1.0, rely=i / (len(db_labels) - 1),
                anchor="e", x=-2)

        # Band sliders
        bands_frame = tk.Frame(sliders_frame, bg=C.BG)
        bands_frame.pack(fill=tk.BOTH, expand=True)

        for i in range(EQ_NUM_BANDS):
            col = tk.Frame(bands_frame, bg=C.BG)
            col.pack(side=tk.LEFT, fill=tk.Y, expand=True, padx=1)

            # dB value label at top
            val_var = tk.DoubleVar(value=0.0)
            self._eq_band_vars.append(val_var)

            val_lbl = tk.Label(col, text="0", bg=C.BG,
                               fg=C.TEXT_DIM, font=("Segoe UI", 7),
                               width=3)
            val_lbl.pack(side=tk.TOP, pady=(0, 1))
            self._eq_band_labels.append(val_lbl)

            # Vertical slider
            slider = tk.Scale(
                col, from_=12.0, to=-12.0,
                resolution=0.5, orient=tk.VERTICAL,
                variable=val_var, showvalue=False,
                bg=C.BG, fg=C.TEXT,
                troughcolor=C.BORDER,        # visible trough
                highlightthickness=0,
                sliderrelief="raised", bd=1,
                activebackground=C.ACCENT,
                length=160, width=14,
                sliderlength=16,
                command=lambda v, idx=i: self._on_eq_slider(idx),
            )
            slider.pack(side=tk.TOP, fill=tk.Y, expand=True)
            self._eq_band_sliders.append(slider)

            # Frequency label at bottom
            tk.Label(col, text=EQ_BAND_LABELS[i], bg=C.BG,
                     fg=C.TEXT_DIM, font=("Segoe UI", 7)).pack(
                side=tk.BOTTOM, pady=(1, 0))

        # ── Bottom controls ──
        tk.Frame(outer, bg=C.BORDER, height=1).pack(
            fill=tk.X, padx=6, pady=(4, 2))

        bottom = tk.Frame(outer, bg=C.BG)
        bottom.pack(fill=tk.X, padx=6, pady=(2, 6))
        ttk.Button(bottom, text="Apply EQ",
                   command=self._apply_eq).pack(side=tk.RIGHT)
        self._eq_preview_btn = ttk.Button(bottom, text="▶ Preview",
                   command=self._preview_eq)
        self._eq_preview_btn.pack(side=tk.RIGHT, padx=(0, 5))

        # ── Project-level EQ state ──
        self._project_eq = list(DEFAULT_EQ_BANDS)
        self._eq_preview_playing = False

        return outer

    def _on_eq_slider(self, idx):
        """Update the dB label when a slider changes."""
        val = self._eq_band_vars[idx].get()
        sign = "+" if val > 0 else ""
        self._eq_band_labels[idx].config(
            text=f"{sign}{val:.0f}" if val == int(val)
            else f"{sign}{val:.1f}")

    def _on_eq_preset(self, event=None):
        """Apply an EQ preset to the sliders."""
        name = self._eq_preset_var.get()
        if name in EQ_PRESETS:
            bands = EQ_PRESETS[name]
            for i, g in enumerate(bands):
                self._eq_band_vars[i].set(g)
                self._on_eq_slider(i)

    def _reset_eq(self):
        """Reset all EQ bands to 0 dB."""
        self._eq_preset_var.set("Flat")
        for i in range(EQ_NUM_BANDS):
            self._eq_band_vars[i].set(0.0)
            self._on_eq_slider(i)

    def _refresh_eq_panel(self):
        """Load EQ values from the active scope (clip or project)."""
        scope = self._eq_scope_var.get()
        if scope == "clip" and self._selected_track == "audio" \
                and self.selected_media_clip:
            bands = self.selected_media_clip.audio_effects.get(
                "eq_bands", None) or list(DEFAULT_EQ_BANDS)
        else:
            bands = list(self._project_eq)
        for i in range(EQ_NUM_BANDS):
            self._eq_band_vars[i].set(bands[i] if i < len(bands) else 0.0)
            self._on_eq_slider(i)

    def _apply_eq(self):
        """Write current slider values to clip or project EQ."""
        bands = [self._eq_band_vars[i].get() for i in range(EQ_NUM_BANDS)]
        scope = self._eq_scope_var.get()
        if scope == "clip":
            if self._selected_track == "audio" and self.selected_media_clip:
                self._invalidate_audio_cache(self.selected_media_clip)
                self.selected_media_clip.audio_effects["eq_bands"] = bands
                self._build_audio_cache()
                self._status_var.set("Clip EQ applied")
            else:
                self._status_var.set("No audio clip selected")
        else:
            self._project_eq = list(bands)
            self._status_var.set("Project EQ applied")

    def _preview_eq(self):
        """Preview audio with current EQ slider settings."""
        if not _HAS_AUDIO:
            self._status_var.set("Audio playback not available")
            return

        # Stop any current preview
        if self._eq_preview_playing:
            sd.stop()
            self._eq_preview_playing = False
            self._eq_preview_btn.config(text="\u25b6 Preview")
            self._status_var.set("Preview stopped")
            return

        # Get current EQ bands from sliders
        bands = [self._eq_band_vars[i].get() for i in range(EQ_NUM_BANDS)]

        # Determine audio source based on scope
        scope = self._eq_scope_var.get()
        audio_source = None

        if scope == "clip" and self._selected_track == "audio" and self.selected_media_clip:
            audio_source = self.selected_media_clip.source_path
        elif self.audio_path:
            audio_source = self.audio_path
        elif self.audio_clips:
            audio_source = self.audio_clips[0].source_path

        if not audio_source or not os.path.isfile(audio_source):
            self._status_var.set("No audio to preview")
            return

        try:
            import numpy as np
            # Load 5 seconds of audio from current playback position
            audio_data, sr = sf.read(audio_source, dtype='float32')

            # Start from current playback position
            start_sample = int(self._playback_time * sr)
            # Preview duration: 5 seconds
            preview_samples = int(5.0 * sr)
            end_sample = min(start_sample + preview_samples, len(audio_data))

            if start_sample >= len(audio_data):
                start_sample = 0
                end_sample = min(preview_samples, len(audio_data))

            preview_audio = audio_data[start_sample:end_sample]

            # Apply EQ
            if any(abs(g) > 0.01 for g in bands):
                preview_audio = apply_eq(preview_audio, sr, bands)
                preview_audio = np.clip(preview_audio, -1.0, 1.0)

            # Play preview
            sd.play(preview_audio.astype(np.float32), sr)
            self._eq_preview_playing = True
            self._eq_preview_btn.config(text="\u25a0 Stop")
            self._status_var.set("Playing EQ preview (5s)...")

            # Stop after preview duration
            def _stop_preview():
                if self._eq_preview_playing:
                    self._eq_preview_playing = False
                    self._eq_preview_btn.config(text="\u25b6 Preview")
                    self._status_var.set("Preview complete")
            self.after(5000, _stop_preview)

        except Exception as e:
            self._status_var.set(f"Preview error: {e}")
            self._eq_preview_playing = False
            self._eq_preview_btn.config(text="\u25b6 Preview")

    # ── Style Panel ────────────────────────────────────────────
    def _build_style_panel(self, parent):
        outer = ttk.LabelFrame(parent, text=" Style ", width=210)

        canvas = tk.Canvas(outer, highlightthickness=0, bg=C.BG, bd=0,
                           width=200)
        scrollbar = ttk.Scrollbar(outer, orient=tk.VERTICAL,
                                  command=canvas.yview)
        scroll_frame = ttk.Frame(canvas)
        scroll_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.bind("<Enter>",
                    lambda e: canvas.bind_all(
                        "<MouseWheel>",
                        lambda ev: canvas.yview_scroll(
                            int(-ev.delta / 120), "units")))
        canvas.bind("<Leave>",
                    lambda e: canvas.unbind_all("<MouseWheel>"))
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        f = scroll_frame
        row = 0
        SML = ("Segoe UI", 8)

        def sect(text, r):
            ttk.Label(f, text=text, style="Header.TLabel").grid(
                row=r, column=0, columnspan=2, sticky="w",
                padx=4, pady=(8, 2))
            return r + 1

        def sep(r):
            tk.Frame(f, bg=C.BORDER, height=1).grid(
                row=r, column=0, columnspan=2, sticky="we",
                padx=4, pady=5)
            return r + 1

        def lbl(text, r, indent=4):
            ttk.Label(f, text=text, style="Dim.TLabel",
                      font=SML).grid(
                row=r, column=0, sticky="w", padx=indent)
            return r

        def cswatch(name, attr, r, indent=4):
            lbl(name, r, indent)
            btn = tk.Button(
                f, width=2, bg=_rgb_to_hex(getattr(self, attr)),
                relief="flat", bd=1, highlightthickness=1,
                highlightbackground=C.BORDER, cursor="hand2")
            btn.config(command=lambda a=attr, b=btn: self._pick_color(a, b))
            btn.grid(row=r, column=1, sticky="w", padx=4, pady=1)
            setattr(self, f"{attr}_btn", btn)
            return r + 1

        # ── Preset selector at top ──
        row = sect("Preset", row)
        self._all_presets = _load_all_presets()
        _default_preset = "Karaoke Classic"
        self._preset_var = tk.StringVar(
            value=_default_preset if _default_preset in self._all_presets
            else "")
        self._preset_combo = ttk.Combobox(
            f, textvariable=self._preset_var,
            values=sorted(self._all_presets.keys()), state="readonly",
            width=16, font=SML)
        self._preset_combo.grid(row=row, column=0, columnspan=2,
                                sticky="we", padx=4, pady=2)
        self._preset_combo.bind("<<ComboboxSelected>>",
                                self._on_preset_combo)
        # Apply default preset on startup
        if _default_preset in self._all_presets:
            self.after_idle(
                lambda: self._apply_preset(
                    self._all_presets[_default_preset]))
        row += 1
        ttk.Button(f, text="Save Preset\u2026",
                   command=self._save_preset).grid(
            row=row, column=0, columnspan=2,
            sticky="we", padx=4, pady=(0, 2))
        row += 1
        row = sep(row)

        # Font
        row = sect("Font", row)
        lbl("Family:", row)
        # Build font cache and get available font names
        _build_font_cache()
        font_names = sorted(set(
            os.path.splitext(os.path.basename(p))[0]
            for p in _font_cache.values()
        ), key=str.lower)
        self._font_family_var = tk.StringVar(
            value=os.path.splitext(os.path.basename(self._font_path))[0])
        font_combo = ttk.Combobox(
            f, textvariable=self._font_family_var,
            values=font_names, width=14, font=SML, state="readonly")
        font_combo.grid(row=row, column=1, sticky="we", padx=4, pady=1)
        font_combo.bind("<<ComboboxSelected>>", lambda e: self._on_font_change())
        self._font_combo = font_combo
        row += 1
        lbl("Size:", row)
        self._font_size_var = tk.IntVar(value=self._font_size)
        ttk.Spinbox(f, from_=12, to=200,
                    textvariable=self._font_size_var,
                    width=4, font=SML, state="readonly").grid(
            row=row, column=1, sticky="w", padx=4, pady=1)
        row += 1
        lbl("Case:", row)
        self._text_case_var = tk.StringVar(value=self._text_case)
        case_combo = ttk.Combobox(
            f, textvariable=self._text_case_var,
            values=["UPPERCASE", "Normal", "lowercase"],
            state="readonly", width=10, font=SML)
        case_combo.grid(row=row, column=1, sticky="w", padx=4, pady=1)
        case_combo.bind("<<ComboboxSelected>>",
                        lambda e: self._on_text_case_change())
        row += 1

        # colors
        row = sep(row)
        row = sect("Colors", row)
        row = cswatch("Text Color:", "_text_color", row)
        row = cswatch("Highlight:", "_highlight_color", row)

        # Stroke
        row = sep(row)
        self._stroke_var = tk.BooleanVar(value=self._stroke_enabled)
        stroke_cb = ttk.Checkbutton(f, text="Stroke",
                        variable=self._stroke_var,
                        takefocus=False)
        stroke_cb.grid(row=row, column=0, columnspan=2, sticky="w",
            padx=4, pady=2)
        stroke_cb.bind("<ButtonRelease-1>", lambda e: self.after(10, self._trigger_style_refresh))
        row += 1
        row = cswatch("Color:", "_stroke_color", row, indent=12)
        lbl("Width:", row, indent=12)
        self._stroke_width_var = tk.IntVar(value=self._stroke_width)
        ttk.Spinbox(f, from_=0, to=12,
                    textvariable=self._stroke_width_var,
                    width=3, font=SML, state="readonly").grid(
            row=row, column=1, sticky="w", padx=4, pady=1)
        row += 1

        # Shadow
        row = sep(row)
        self._shadow_var = tk.BooleanVar(value=self._shadow_enabled)
        shadow_cb = ttk.Checkbutton(f, text="Shadow",
                        variable=self._shadow_var,
                        takefocus=False)
        shadow_cb.grid(row=row, column=0, columnspan=2, sticky="w",
            padx=4, pady=2)
        shadow_cb.bind("<ButtonRelease-1>", lambda e: self.after(10, self._trigger_style_refresh))
        row += 1
        for lt, vn, dv in [
            ("Offset X:", "_shadow_ox_var", self._shadow_offset[0]),
            ("Offset Y:", "_shadow_oy_var", self._shadow_offset[1]),
            ("Blur:", "_shadow_blur_var", self._shadow_blur),
        ]:
            lbl(lt, row, indent=12)
            var = tk.IntVar(value=dv)
            setattr(self, vn, var)
            ttk.Spinbox(f, from_=0, to=20, textvariable=var,
                        width=3, font=SML, state="readonly").grid(
                row=row, column=1, sticky="w", padx=4, pady=1)
            row += 1

        # Highlight Box
        row = sep(row)
        self._hibox_var = tk.BooleanVar(value=self._highlight_box_enabled)
        hibox_cb = ttk.Checkbutton(f, text="Highlight Box",
                        variable=self._hibox_var,
                        takefocus=False)
        hibox_cb.grid(row=row, column=0, columnspan=2, sticky="w",
            padx=4, pady=2)
        hibox_cb.bind("<ButtonRelease-1>", lambda e: self.after(10, self._trigger_style_refresh))
        row += 1
        row = cswatch("Color:", "_highlight_box_color", row, indent=12)
        lbl("Opacity:", row, indent=12)
        self._hibox_opacity_var = tk.DoubleVar(value=self._highlight_box_opacity)
        FillBar(f, variable=self._hibox_opacity_var,
                from_=0.0, to=1.0, length=120, height=14).grid(
            row=row, column=1, sticky="we", padx=4, pady=1)
        row += 1

        # Glow
        row = sep(row)
        self._glow_var = tk.BooleanVar(value=self._glow_enabled)
        glow_cb = ttk.Checkbutton(f, text="Glow",
                        variable=self._glow_var,
                        takefocus=False)
        glow_cb.grid(row=row, column=0, columnspan=2, sticky="w",
            padx=4, pady=2)
        glow_cb.bind("<ButtonRelease-1>", lambda e: self.after(10, self._trigger_style_refresh))
        row += 1
        row = cswatch("Color:", "_glow_color", row, indent=12)
        lbl("Size:", row, indent=12)
        self._glow_size_var = tk.IntVar(value=self._glow_size)
        ttk.Spinbox(f, from_=1, to=20,
                    textvariable=self._glow_size_var,
                    width=3, font=SML, state="readonly").grid(
            row=row, column=1, sticky="w", padx=4, pady=1)
        row += 1
        lbl("Opacity:", row, indent=12)
        self._glow_opacity_var = tk.IntVar(value=self._glow_opacity)
        FillBar(f, variable=self._glow_opacity_var,
                from_=0, to=300, length=120, height=14).grid(
            row=row, column=1, sticky="we", padx=4, pady=1)
        row += 1

        # Gradient
        row = sep(row)
        self._gradient_var = tk.BooleanVar(value=self._gradient_enabled)
        grad_cb = ttk.Checkbutton(f, text="Gradient",
                        variable=self._gradient_var,
                        takefocus=False)
        grad_cb.grid(row=row, column=0, columnspan=2, sticky="w",
            padx=4, pady=2)
        grad_cb.bind("<ButtonRelease-1>", lambda e: self.after(10, self._trigger_style_refresh))
        row += 1
        row = cswatch("Color 1:", "_gradient_color1", row, indent=12)
        row = cswatch("Color 2:", "_gradient_color2", row, indent=12)
        lbl("Opacity:", row, indent=12)
        self._grad_opacity_var = tk.DoubleVar(value=self._gradient_opacity)
        FillBar(f, variable=self._grad_opacity_var,
                from_=0.0, to=1.0, length=120, height=14).grid(
            row=row, column=1, sticky="we", padx=4, pady=1)
        row += 1

        # Bevel
        row = sep(row)
        self._bevel_var = tk.BooleanVar(value=self._bevel_enabled)
        bevel_cb = ttk.Checkbutton(f, text="Bevel",
                        variable=self._bevel_var,
                        takefocus=False)
        bevel_cb.grid(row=row, column=0, columnspan=2, sticky="w",
            padx=4, pady=2)
        bevel_cb.bind("<ButtonRelease-1>", lambda e: self.after(10, self._trigger_style_refresh))
        row += 1
        lbl("Depth:", row, indent=12)
        self._bevel_depth_var = tk.IntVar(value=self._bevel_depth)
        ttk.Spinbox(f, from_=1, to=15,
                    textvariable=self._bevel_depth_var,
                    width=3, font=SML, state="readonly").grid(
            row=row, column=1, sticky="w", padx=4, pady=1)
        row += 1

        # Chrome
        row = sep(row)
        self._chrome_var = tk.BooleanVar(value=self._chrome_enabled)
        chrome_cb = ttk.Checkbutton(f, text="Chrome",
                        variable=self._chrome_var,
                        takefocus=False)
        chrome_cb.grid(row=row, column=0, columnspan=2, sticky="w",
            padx=4, pady=2)
        chrome_cb.bind("<ButtonRelease-1>", lambda e: self.after(10, self._trigger_style_refresh))
        row += 1
        lbl("Opacity:", row, indent=12)
        self._chrome_opacity_var = tk.DoubleVar(value=self._chrome_opacity)
        FillBar(f, variable=self._chrome_opacity_var,
                from_=0.0, to=1.0, length=120, height=14).grid(
            row=row, column=1, sticky="we", padx=4, pady=1)
        row += 1

        # Texture
        row = sep(row)
        tex_row = tk.Frame(f, bg=C.BG)
        tex_row.grid(row=row, column=0, columnspan=2, sticky="w", padx=4, pady=2)
        self._texture_var = tk.BooleanVar(value=self._texture_enabled)
        texture_cb = ttk.Checkbutton(tex_row, text="Texture",
                        variable=self._texture_var,
                        takefocus=False)
        texture_cb.pack(side="left")
        texture_cb.bind("<ButtonRelease-1>", lambda e: self.after(10, self._trigger_style_refresh))
        ttk.Button(tex_row, text="Load...", width=6,
                   command=self._load_texture_file).pack(side="left", padx=(8, 0))
        row += 1
        lbl("Blend:", row, indent=12)
        self._texture_blend_var = tk.StringVar(value=self._texture_blend_mode)
        blend_combo = ttk.Combobox(f, textvariable=self._texture_blend_var,
                                   values=["normal", "screen", "multiply", "overlay"],
                                   width=8, state="readonly", font=SML)
        blend_combo.grid(row=row, column=1, sticky="w", padx=4, pady=1)
        blend_combo.bind("<<ComboboxSelected>>", lambda e: self._trigger_style_refresh())
        row += 1
        lbl("Scale:", row, indent=12)
        self._texture_scale_var = tk.DoubleVar(value=self._texture_scale)
        FillBar(f, variable=self._texture_scale_var,
                from_=0.5, to=3.0, length=120, height=14).grid(
            row=row, column=1, sticky="we", padx=4, pady=1)
        row += 1
        lbl("Opacity:", row, indent=12)
        self._texture_opacity_var = tk.DoubleVar(value=self._texture_opacity)
        FillBar(f, variable=self._texture_opacity_var,
                from_=0.0, to=1.0, length=120, height=14).grid(
            row=row, column=1, sticky="we", padx=4, pady=1)
        row += 1

        # Layout
        row = sep(row)
        row = sect("Layout", row)
        lbl("Words/Line:", row)
        self._max_wpl_var = tk.IntVar(value=self._max_words_per_line)
        ttk.Spinbox(f, from_=1, to=12,
                    textvariable=self._max_wpl_var,
                    width=4, font=SML, state="readonly").grid(
            row=row, column=1, sticky="w", padx=4, pady=1)
        row += 1
        lbl("X Position:", row)
        self._x_ratio_var = tk.DoubleVar(value=self._position_x_ratio)
        FillBar(f, variable=self._x_ratio_var,
                from_=0.0, to=1.0, length=120, height=14).grid(
            row=row, column=1, sticky="we", padx=4, pady=1)
        row += 1
        lbl("Y Position:", row)
        self._y_ratio_var = tk.DoubleVar(value=self._position_y_ratio)
        FillBar(f, variable=self._y_ratio_var,
                from_=0.0, to=1.0, length=120, height=14).grid(
            row=row, column=1, sticky="we", padx=4, pady=1)
        row += 1
        lbl("Justify:", row)
        self._justify_var = tk.StringVar(value=self._text_justify.capitalize())
        justify_combo = ttk.Combobox(f, textvariable=self._justify_var,
                                      values=["Left", "Center", "Right"],
                                      width=10, state="readonly", font=SML)
        justify_combo.grid(row=row, column=1, sticky="w", padx=4, pady=1)
        justify_combo.bind("<<ComboboxSelected>>", lambda e: self._trigger_style_refresh())
        row += 1
        lbl("Highlight:", row)
        self._highlight_mode_var = tk.StringVar(value=self._highlight_mode.capitalize())
        highlight_combo = ttk.Combobox(f, textvariable=self._highlight_mode_var,
                                       values=["Word", "Character"],
                                       width=10, state="readonly", font=SML)
        highlight_combo.grid(row=row, column=1, sticky="w", padx=4, pady=1)
        highlight_combo.bind("<<ComboboxSelected>>", lambda e: self._trigger_style_refresh())
        row += 1

        tk.Frame(f, bg=C.BG, height=8).grid(row=row, column=0, columnspan=2)
        f.columnconfigure(1, weight=1)

        # Bind numeric/slider style vars to auto-update preview (not checkboxes - they use Button-1 binding)
        # Store as instance method to prevent garbage collection
        self._style_var_trace_cb = lambda *args: self._trigger_style_refresh()
        for var in [
            self._font_size_var,
            self._stroke_var, self._stroke_width_var,
            self._shadow_var, self._shadow_ox_var, self._shadow_oy_var, self._shadow_blur_var,
            self._hibox_var, self._hibox_opacity_var,
            self._glow_var, self._glow_size_var, self._glow_opacity_var,
            self._gradient_var, self._grad_opacity_var,
            self._bevel_var, self._bevel_depth_var,
            self._chrome_var, self._chrome_opacity_var,
            self._texture_var, self._texture_scale_var, self._texture_opacity_var,
            self._texture_blend_var,
            self._max_wpl_var,
            self._x_ratio_var,
            self._y_ratio_var,
            self._justify_var,
            self._highlight_mode_var,
        ]:
            var.trace_add("write", self._style_var_trace_cb)

        return outer

    # ── Timeline ───────────────────────────────────────────────
    def _build_timeline(self, parent):
        self._tl_xscroll = ttk.Scrollbar(
            parent, orient=tk.HORIZONTAL,
            command=self._on_timeline_scroll)
        self._tl_xscroll.pack(side=tk.BOTTOM, fill=tk.X)

        self._tl_canvas = tk.Canvas(
            parent, bg=C.TRACK_BG, highlightthickness=0)
        self._tl_canvas.pack(fill=tk.BOTH, expand=True)

        self._tl_canvas.bind("<Configure>", self._on_tl_configure)
        self._tl_canvas.bind("<ButtonPress-1>", self._on_tl_press)
        self._tl_canvas.bind("<B1-Motion>", self._on_tl_drag)
        self._tl_canvas.bind("<ButtonRelease-1>", self._on_tl_release)
        self._tl_canvas.bind("<MouseWheel>", self._on_tl_mousewheel)
        self._tl_canvas.bind("<Motion>", self._on_tl_hover)
        self._tl_canvas.bind("<Button-3>", self._on_tl_right_click)

    # ── Detail panel ───────────────────────────────────────────
    def _build_detail_panel(self, parent):
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, padx=6, pady=(4, 4))

        ttk.Label(row, text="Text:", style="Dim.TLabel").pack(
            side=tk.LEFT, padx=(0, 3))
        self._detail_text_var = tk.StringVar()
        self._detail_text_entry = ttk.Entry(row, textvariable=self._detail_text_var)
        self._detail_text_entry.pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))
        
        # Click anywhere in the detail panel (except entries) unfocuses text entry
        def _unfocus_text_entry(e):
            # Don't unfocus if clicking on an Entry widget
            widget_class = e.widget.winfo_class()
            if widget_class != "TEntry" and widget_class != "Entry":
                self._tl_canvas.focus_set()
        parent.bind("<Button-1>", _unfocus_text_entry, add="+")
        row.bind("<Button-1>", _unfocus_text_entry, add="+")

        ttk.Label(row, text="Start:", style="Dim.TLabel").pack(
            side=tk.LEFT, padx=(0, 2))
        self._detail_start_var = tk.StringVar()
        ttk.Entry(row, textvariable=self._detail_start_var,
                  width=8).pack(side=tk.LEFT, padx=(0, 6))

        ttk.Label(row, text="End:", style="Dim.TLabel").pack(
            side=tk.LEFT, padx=(0, 2))
        self._detail_end_var = tk.StringVar()
        ttk.Entry(row, textvariable=self._detail_end_var,
                  width=8).pack(side=tk.LEFT, padx=(0, 6))

        tk.Label(row, text="Dur:", bg=C.BG, fg=C.TEXT_DIM,
                 font=("Segoe UI", 9)).pack(side=tk.LEFT, padx=(0, 2))
        self._detail_dur_var = tk.StringVar()
        tk.Label(row, textvariable=self._detail_dur_var,
                 bg=C.BG, fg=C.TEXT_DIM, font=("Segoe UI", 9),
                 width=6, anchor="w").pack(side=tk.LEFT, padx=(0, 4))

        ttk.Label(row, text="Size:", style="Dim.TLabel").pack(
            side=tk.LEFT, padx=(0, 2))
        self._detail_fontsize_var = tk.StringVar()
        ttk.Entry(row, textvariable=self._detail_fontsize_var,
                  width=4).pack(side=tk.LEFT, padx=(0, 6))

        ttk.Button(row, text="Apply",
                   command=self._apply_detail).pack(
            side=tk.RIGHT, padx=(6, 0))

    # ────────────────────────────────────────────────────────────
    # Shortcuts
    # ────────────────────────────────────────────────────────────
    def _bind_shortcuts(self):
        self.bind("<Delete>", self._on_delete_key)
        self.bind("<Escape>", lambda e: self._clear_selection())
        self.bind("<Control-n>", lambda e: self._new_project())
        self.bind("<Control-o>", lambda e: self._open_project())
        self.bind("<Control-s>", lambda e: self._save_project())
        self.bind("<Control-Shift-S>", lambda e: self._save_project_as())
        self.bind("<Control-i>", lambda e: self._insert_clip())
        self.bind("<Control-c>", self._on_ctrl_c)
        self.bind("<Control-v>", self._on_ctrl_v)
        self.bind("<Control-d>", lambda e: self._duplicate_clip())
        self.bind("<s>", self._on_s_key)
        self.bind("<S>", self._on_s_key)
        # Use bind_all to catch space before it reaches widgets like buttons
        self.bind_all("<space>", self._on_space_key)

    def _on_s_key(self, event):
        """Handle S key - type 's' if in textbox, otherwise split clip."""
        focused = self.focus_get()
        # If focus is in Entry, Text, Spinbox, or Combobox, let the default typing happen
        if isinstance(focused, (tk.Entry, tk.Text, tk.Spinbox, ttk.Entry, ttk.Spinbox, ttk.Combobox)):
            return  # Don't consume event, let widget handle typing
        self._split_selected()
        return "break"  # Consume event

    def _on_delete_key(self, event):
        """Handle Delete - delete text if in textbox, otherwise delete selected clip."""
        focused = self.focus_get()
        # If focus is in Entry, Text, Spinbox, or Combobox, let the default delete happen
        if isinstance(focused, (tk.Entry, tk.Text, tk.Spinbox, ttk.Entry, ttk.Spinbox, ttk.Combobox)):
            return  # Don't consume event, let widget handle delete
        self._delete_selected()
        return "break"  # Consume event

    def _on_ctrl_c(self, event):
        """Handle Ctrl+C - copy text if in textbox, otherwise copy clip."""
        focused = self.focus_get()
        # If focus is in Entry, Text, Spinbox, or Combobox, let the default copy happen
        if isinstance(focused, (tk.Entry, tk.Text, tk.Spinbox, ttk.Entry, ttk.Spinbox, ttk.Combobox)):
            return  # Don't consume event, let widget handle copy
        self._copy_clip()
        return "break"  # Consume event

    def _on_ctrl_v(self, event):
        """Handle Ctrl+V - paste into textbox if focused, otherwise paste clip."""
        focused = self.focus_get()
        # If focus is in Entry, Text, Spinbox, or Combobox, let the default paste happen
        if isinstance(focused, (tk.Entry, tk.Text, tk.Spinbox, ttk.Entry, ttk.Spinbox, ttk.Combobox)):
            return  # Don't consume event, let widget handle paste
        self._paste_clip()
        return "break"  # Consume event

    # ────────────────────────────────────────────────────────────
    # Preview / Playback
    # ────────────────────────────────────────────────────────────
    def _on_space_key(self, event):
        """Handle space key - only allow in text inputs, otherwise play/pause."""
        focused = self.focus_get()
        if focused is None:
            self._play_pause()
            return "break"
        widget_class = focused.winfo_class()
        # Only let space work in text input widgets (entries, text boxes)
        if widget_class in ("TEntry", "Entry", "Text"):
            return  # Let the widget handle space for typing
        # For all other widgets (buttons, checkbuttons, spinboxes, etc.), 
        # block default behavior and trigger play/pause
        self._play_pause()
        return "break"

    def _on_preview_resize(self, event):
        """Debounced preview resize - only refresh after dragging stops."""
        if self._preview_resize_after:
            self.after_cancel(self._preview_resize_after)
        self._preview_resize_after = self.after(50, self._do_preview_resize)

    def _do_preview_resize(self):
        """Actually refresh the preview after resize debounce."""
        self._preview_resize_after = None
        self._refresh_preview()

    def _on_preview_click(self, event):
        """Click on preview to seek (relative x position)."""
        self._preview_canvas.focus_set()  # Defocus any textboxes
        if self.timeline_duration <= 0:
            return
        cw = self._preview_canvas.winfo_width()
        ratio = max(0.0, min(1.0, event.x / max(cw, 1)))
        self._playback_time = ratio * self.timeline_duration
        self._seek_var.set(ratio)
        self._refresh_preview()
        self._redraw_timeline()

    def _format_time(self, t):
        m = int(t) // 60
        s = t - m * 60
        return f"{m}:{s:05.2f}"

    def _refresh_preview(self):
        """Draw the current frame on the preview canvas."""
        from PIL import Image, ImageTk

        cw = self._preview_canvas.winfo_width()
        ch = self._preview_canvas.winfo_height()
        if cw < 10 or ch < 10:
            return

        frame_img = None
        t = self._playback_time

        # Find active video clip at time t
        active_vclip = None
        for vc in self.video_clips:
            if vc.start <= t < vc.end:
                active_vclip = vc
                break

        if active_vclip is not None:
            source_time = active_vclip.source_offset + (t - active_vclip.start)
            if active_vclip.source_type == "video" and self._video_clip is not None:
                try:
                    ct = max(0, min(source_time, self._video_clip.duration - 0.01))
                    arr = self._video_clip.get_frame(ct)
                    frame_img = Image.fromarray(arr)
                except Exception:
                    pass
            elif active_vclip.source_type == "image" and self._bg_image_pil is not None:
                frame_img = self._bg_image_pil.copy()

        if frame_img is None:
            self._preview_canvas.delete("preview")
            self._preview_canvas.delete("placeholder")
            self._preview_canvas.create_text(
                cw // 2, ch // 2,
                text="No media loaded", fill=C.TEXT_MUTED,
                font=("Segoe UI", 11), tags="placeholder")
            self._update_time_display()
            return

        # Scale to fit canvas
        iw, ih = frame_img.size
        scale = min(cw / iw, ch / ih)
        nw, nh = int(iw * scale), int(ih * scale)
        # Use fast resize during playback, high-quality when paused
        resample = Image.BILINEAR if self._playing else Image.LANCZOS
        frame_img = frame_img.resize((nw, nh), resample)

        # Overlay subtitle text for the current time
        frame_img = self._overlay_subtitle_on_frame(frame_img, t)

        self._preview_photo = ImageTk.PhotoImage(frame_img)
        self._preview_canvas.delete("preview")
        self._preview_canvas.delete("placeholder")
        self._preview_canvas.create_image(
            cw // 2, ch // 2, image=self._preview_photo,
            anchor="center", tags="preview")

        self._update_time_display()

    def _overlay_subtitle_on_frame(self, frame_img, t):
        """Render the active subtitle text at time *t* onto *frame_img*."""
        from PIL import Image, ImageDraw, ImageFont, ImageFilter
        import numpy as np

        # Find the active clip(s) at this time
        active_clips = [c for c in self.subtitles
                        if c.start <= t < c.end and c.text.strip()]
        if not active_clips:
            return frame_img

        frame_img = frame_img.convert("RGBA")
        fw, fh = frame_img.size

        # Scale font size proportionally to preview size
        # Original design resolution
        ref_h = VIDEO_HEIGHT if VIDEO_HEIGHT else 1080
        font_scale = fh / ref_h

        for clip in active_clips:
            # Use clip-specific font size if set, otherwise global
            clip_font_size = clip.font_size if clip.font_size else self._font_size
            font_size = max(12, int(clip_font_size * font_scale))
            
            # Load font for this clip
            try:
                pil_font = ImageFont.truetype(self._font_path, font_size)
            except Exception:
                try:
                    pil_font = ImageFont.truetype("impact.ttf", font_size)
                except Exception:
                    pil_font = ImageFont.load_default()

            text = clip.text.strip()
            # Apply text case transformation
            if self._text_case == "uppercase":
                text = text.upper()
            elif self._text_case == "lowercase":
                text = text.lower()
            
            # Handle explicit line breaks (\n from \r input)
            # Split by newlines first, then apply word wrapping to each segment
            mwpl = max(1, self._max_words_per_line)
            lines = []
            for segment in text.split('\n'):
                segment = segment.strip()
                if not segment:
                    continue
                words = segment.split()
                if not words:
                    continue
                for i in range(0, len(words), mwpl):
                    lines.append(" ".join(words[i:i + mwpl]))
            
            if not lines:
                continue

            # Calculate highlight progress within this clip
            clip_dur = clip.end - clip.start
            elapsed = t - clip.start
            progress = max(0.0, min(1.0, elapsed / clip_dur)) if clip_dur > 0 else 1.0

            # Create a transparent overlay for the text
            overlay = Image.new("RGBA", (fw, fh), (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)

            # Measure all lines (store bbox top offset for proper positioning)
            line_bboxes = []
            for line in lines:
                bbox = draw.textbbox((0, 0), line, font=pil_font)
                lw = bbox[2] - bbox[0]
                lh = bbox[3] - bbox[1]
                ltop = bbox[1]  # Y offset from draw position to actual text top
                line_bboxes.append((lw, lh, ltop))

            line_spacing = int(font_size * 0.25)
            total_h = sum(h for _, h, _ in line_bboxes) + line_spacing * (len(lines) - 1)
            y_pos = int(fh * self._position_y_ratio) - total_h // 2

            # Compute max line width for justification reference
            max_lw = max(lw for lw, _, _ in line_bboxes) if line_bboxes else 0

            def _line_x(lw):
                """Return line X position based on justify and X ratio."""
                if self._text_justify == "left":
                    # Anchor left edge of text block at X ratio position
                    block_x = int(fw * self._position_x_ratio) - max_lw // 2
                    return block_x
                elif self._text_justify == "right":
                    # Anchor right edge of text block at X ratio position
                    block_x = int(fw * self._position_x_ratio) + max_lw // 2
                    return block_x - lw
                else:  # center
                    return int(fw * self._position_x_ratio) - lw // 2

            # Determine colours
            text_rgb = self._text_color
            hi_rgb = self._highlight_color
            stroke_rgb = self._stroke_color
            stroke_w = int(max(1, self._stroke_width * font_scale)) if self._stroke_enabled else 0

            # Count total characters for highlight sweep
            total_chars = sum(len(line) for line in lines)
            if self._highlight_mode == "word":
                # Word mode: highlight whole words at a time
                all_words = text.split()
                n_words = len(all_words)
                highlighted_words = int(progress * n_words)
                # Build a per-character map: True if char should be highlighted
                # Walk through each line's characters, tracking word boundaries
                _hi_flags = []
                word_count = 0
                for line in lines:
                    in_word = False
                    for ch in line:
                        if ch == ' ':
                            # Spaces are highlighted if they fall between highlighted words
                            _hi_flags.append(word_count <= highlighted_words and word_count > 0)
                            in_word = False
                        else:
                            if not in_word:
                                word_count += 1
                                in_word = True
                            _hi_flags.append(word_count <= highlighted_words)
                highlight_char = None  # sentinel: use _hi_flags instead
            else:
                # Character mode: smooth per-character sweep
                highlight_char = int(progress * total_chars)
                _hi_flags = None

            # Draw highlight box FIRST (behind everything else)
            if self._highlight_box_enabled:
                box_alpha = int(255 * self._highlight_box_opacity)
                pad = int(font_size * 0.1)
                # Find current word being highlighted
                words_in_text = text.split()
                word_progress = progress * len(words_in_text)
                current_word_idx = min(int(word_progress), len(words_in_text) - 1)
                
                # Track position to find current word
                word_count = 0
                box_y = y_pos
                for i, line in enumerate(lines):
                    lw, lh, ltop = line_bboxes[i]
                    line_x = _line_x(lw)
                    line_words = line.split()
                    cx = line_x
                    for word in line_words:
                        word_bbox = draw.textbbox((0, 0), word, font=pil_font)
                        word_w = word_bbox[2] - word_bbox[0]
                        word_top = word_bbox[1]
                        word_h = word_bbox[3] - word_bbox[1]
                        if word_count == current_word_idx:
                            # Account for font top offset when drawing box
                            draw.rectangle(
                                [cx - pad, box_y + word_top - pad, cx + word_w + pad, box_y + word_top + word_h + pad],
                                fill=(*self._highlight_box_color, box_alpha))
                        cx += word_w
                        # Add space width
                        space_bbox = draw.textbbox((0, 0), " ", font=pil_font)
                        cx += space_bbox[2] - space_bbox[0]
                        word_count += 1
                    box_y += lh + line_spacing

            # Draw glow effect (behind text but after highlight box) - character by character
            if self._glow_enabled:
                glow_overlay = Image.new("RGBA", (fw, fh), (0, 0, 0, 0))
                glow_draw = ImageDraw.Draw(glow_overlay)
                glow_y = y_pos
                # Glow opacity: 0-300 range, use full alpha for source; blur will soften
                glow_alpha = min(255, int(self._glow_opacity * 255 / 100))
                # Thick stroke makes the glow source fat before blur
                glow_stroke_w = max(2, int(self._glow_size * font_scale * 0.4))
                for i, line in enumerate(lines):
                    lw, lh, ltop = line_bboxes[i]
                    gx = _line_x(lw)
                    gcx = gx
                    for ch_char in line:
                        ch_bbox = glow_draw.textbbox((0, 0), ch_char, font=pil_font)
                        ch_w = ch_bbox[2] - ch_bbox[0]
                        glow_draw.text((gcx, glow_y), ch_char, font=pil_font,
                                       fill=(*self._glow_color, glow_alpha),
                                       stroke_width=glow_stroke_w,
                                       stroke_fill=(*self._glow_color, glow_alpha))
                        gcx += ch_w
                    glow_y += lh + line_spacing
                glow_size = max(1, int(self._glow_size * font_scale * 0.5))
                glow_overlay = glow_overlay.filter(ImageFilter.GaussianBlur(glow_size))
                # Amplify glow alpha so colour is clearly visible
                g_arr = np.array(glow_overlay)
                g_arr[:, :, 3] = np.minimum(
                    255, (g_arr[:, :, 3].astype(np.uint16) * 3))
                g_arr[:, :, 3] = g_arr[:, :, 3].astype(np.uint8)
                glow_overlay = Image.fromarray(g_arr)
                overlay = Image.alpha_composite(overlay, glow_overlay)
                draw = ImageDraw.Draw(overlay)

            # Draw shadow - on separate layer with blur
            if self._shadow_enabled:
                shadow_overlay = Image.new("RGBA", (fw, fh), (0, 0, 0, 0))
                shadow_draw = ImageDraw.Draw(shadow_overlay)
                shadow_y = y_pos
                sx = int(self._shadow_offset[0] * font_scale)
                sy = int(self._shadow_offset[1] * font_scale)
                for i, line in enumerate(lines):
                    lw, lh, ltop = line_bboxes[i]
                    shx = _line_x(lw) + sx
                    scx = shx
                    for ch_char in line:
                        ch_bbox = shadow_draw.textbbox((0, 0), ch_char, font=pil_font)
                        ch_w = ch_bbox[2] - ch_bbox[0]
                        shadow_draw.text((scx, shadow_y + sy), ch_char, font=pil_font,
                                  fill=(0, 0, 0, 180))
                        scx += ch_w
                    shadow_y += lh + line_spacing
                # Apply blur to shadow
                shadow_blur = max(1, int(self._shadow_blur * font_scale * 0.5))
                shadow_overlay = shadow_overlay.filter(ImageFilter.GaussianBlur(shadow_blur))
                overlay = Image.alpha_composite(overlay, shadow_overlay)
                draw = ImageDraw.Draw(overlay)

            # Draw stroke/outline character by character to match text positions
            if stroke_w > 0:
                stroke_y = y_pos
                for i, line in enumerate(lines):
                    lw, lh, ltop = line_bboxes[i]
                    stx = _line_x(lw)
                    scx = stx
                    for ch_char in line:
                        ch_bbox = draw.textbbox((0, 0), ch_char, font=pil_font)
                        ch_w = ch_bbox[2] - ch_bbox[0]
                        for dx in range(-stroke_w, stroke_w + 1):
                            for dy in range(-stroke_w, stroke_w + 1):
                                if dx == 0 and dy == 0:
                                    continue
                                draw.text((scx + dx, stroke_y + dy), ch_char, font=pil_font,
                                          fill=(*stroke_rgb, 255))
                        scx += ch_w
                    stroke_y += lh + line_spacing

            # Draw text character by character with highlight
            char_idx = 0
            text_y = y_pos
            for i, line in enumerate(lines):
                lw, lh, ltop = line_bboxes[i]
                x = _line_x(lw)
                cx = x
                for ch_char in line:
                    ch_bbox = draw.textbbox((0, 0), ch_char, font=pil_font)
                    ch_w = ch_bbox[2] - ch_bbox[0]

                    if _hi_flags is not None:
                        is_hi = char_idx < len(_hi_flags) and _hi_flags[char_idx]
                    else:
                        is_hi = char_idx < highlight_char

                    if is_hi:
                        color = (*hi_rgb, 255)
                    else:
                        color = (*text_rgb, 255)

                    draw.text((cx, text_y), ch_char, font=pil_font, fill=color)
                    cx += ch_w
                    char_idx += 1

                text_y += lh + line_spacing

            # Create a fill-only mask for effects (text only, no stroke)
            # This ensures gradient/chrome only apply to text fill, not stroke
            text_mask = Image.new("L", (fw, fh), 0)
            mask_draw = ImageDraw.Draw(text_mask)
            mask_y = y_pos
            for i, line in enumerate(lines):
                lw, lh, ltop = line_bboxes[i]
                mx = _line_x(lw)
                mcx = mx
                for ch_char in line:
                    ch_bbox = mask_draw.textbbox((0, 0), ch_char, font=pil_font)
                    ch_w = ch_bbox[2] - ch_bbox[0]
                    # Draw only the text fill position (no stroke offsets)
                    mask_draw.text((mcx, mask_y), ch_char, font=pil_font, fill=255)
                    mcx += ch_w
                mask_y += lh + line_spacing

            # Apply advanced effects using FancyText module
            has_advanced = (self._gradient_enabled or self._bevel_enabled or
                            self._chrome_enabled or self._texture_enabled)
            if has_advanced:
                import numpy as np
                overlay_arr = np.array(overlay)
                mask_arr = np.array(text_mask)
                # Pass stroke_width=0 since mask is already fill-only
                overlay_arr = apply_advanced_effects(
                    overlay_arr, mask_arr,
                    stroke_width=0,
                    gradient_overlay_enabled=self._gradient_enabled,
                    gradient_overlay_colors=(self._gradient_color1, self._gradient_color2),
                    gradient_overlay_angle=-90,  # color1 at bottom, color2 at top
                    gradient_overlay_opacity=self._gradient_opacity,
                    bevel_emboss_enabled=self._bevel_enabled,
                    bevel_emboss_style="inner_bevel",
                    bevel_emboss_depth=max(3, int(self._bevel_depth * font_scale)),
                    bevel_emboss_angle=135,
                    bevel_emboss_highlight_color=(255, 255, 255),
                    bevel_emboss_shadow_color=(0, 0, 0),
                    bevel_emboss_highlight_opacity=0.75,
                    bevel_emboss_shadow_opacity=0.75,
                    bevel_emboss_soften=3,
                    chrome_enabled=self._chrome_enabled,
                    chrome_opacity=self._chrome_opacity,
                    texture_enabled=self._texture_enabled,
                    texture_scale=self._texture_scale,
                    texture_opacity=self._texture_opacity,
                    texture_blend_mode=self._texture_blend_mode,
                    texture_image=self._texture_image,
                )
                overlay = Image.fromarray(overlay_arr)

            frame_img = Image.alpha_composite(frame_img, overlay)

        return frame_img.convert("RGB")

    def _update_time_display(self):
        dur = self.timeline_duration
        cur = self._format_time(self._playback_time)
        tot = self._format_time(dur)
        self._time_var.set(f"{cur} / {tot}")
        if dur > 0:
            self._seek_var.set(self._playback_time / dur)

    def _play_pause(self):
        if self._playing:
            self._stop_playback()
        else:
            self._start_playback()

    def _start_playback(self):
        if self.timeline_duration <= 0:
            return
        if self._playback_time >= self.timeline_duration:
            self._playback_time = 0.0
        self._playing = True
        self._play_btn.config(text="\u23F8")
        # Draw the full timeline once so clips/waveforms are current
        self._redraw_timeline()
        self._audio_play_from(self._playback_time)
        # Wall-clock anchor set AFTER audio processing completes
        # (prevents desync when effects like autotune/vocoder take seconds)
        import time as _time
        self._playback_wall_start = _time.perf_counter()
        self._playback_pos_start = self._playback_time
        self._playback_tick()

    def _stop_playback(self):
        self._playing = False
        self._play_btn.config(text="\u25B6")
        if self._playback_after_id:
            self.after_cancel(self._playback_after_id)
            self._playback_after_id = None
        self._audio_stop()
        # Full redraw so clips + playhead are clean
        self._redraw_timeline()

    def _playback_tick(self):
        if not self._playing:
            return
        # sounddevice handles its own threading, no pump needed

        # Wall-clock sync: compute _playback_time from real elapsed time
        import time as _time
        wall_now = _time.perf_counter()
        elapsed = (wall_now - self._playback_wall_start) * self._playback_speed
        self._playback_time = self._playback_pos_start + elapsed

        if self._playback_time >= self.timeline_duration:
            self._playback_time = self.timeline_duration
            self._stop_playback()
            self._redraw_timeline()
            return

        self._refresh_preview()
        # Only move the playhead — skip full timeline redraw during playback
        self._update_playhead()

        if self._playing:
            self._playback_after_id = self.after(
                33, self._playback_tick)

    def _update_playhead(self):
        """Move the playhead marker without redrawing the entire timeline."""
        c = self._tl_canvas
        c.delete("playhead")
        draw_w = c.winfo_width() or 800
        ph_x = self._sec_to_x(self._playback_time)
        y_off = RULER_HEIGHT
        track_bottom = y_off + self._total_track_count * (TRACK_HEIGHT + TRACK_PADDING)
        if TRACK_LABEL_W <= ph_x <= draw_w:
            c.create_line(ph_x, 0, ph_x, track_bottom,
                          fill=C.PLAYHEAD, width=1, tags="playhead")
            c.create_polygon(
                ph_x - 4, 0, ph_x + 4, 0, ph_x, 6,
                fill=C.PLAYHEAD, outline="", tags="playhead")

    def _rewind(self):
        self._playback_time = 0.0
        self._refresh_preview()
        self._redraw_timeline()
        if self._playing:
            self._audio_play_from(0.0)
            import time as _time
            self._playback_wall_start = _time.perf_counter()
            self._playback_pos_start = 0.0

    def _step_back(self):
        self._playback_time = max(0, self._playback_time - 1.0)
        self._refresh_preview()
        self._redraw_timeline()

    def _step_forward(self):
        self._playback_time = min(
            self.timeline_duration, self._playback_time + 1.0)
        self._refresh_preview()
        self._redraw_timeline()

    def _on_seek(self, value):
        if self.timeline_duration > 0:
            self._playback_time = float(value) * self.timeline_duration
            self._refresh_preview()
            self._redraw_timeline()
            # If playing, restart audio from new position and re-anchor wall clock
            if self._playing:
                self._audio_play_from(self._playback_time)
                import time as _time
                self._playback_wall_start = _time.perf_counter()
                self._playback_pos_start = self._playback_time

    def _set_speed(self, speed):
        self._playback_speed = speed
        self._speed_var.set(f"{speed}x")

    # ────────────────────────────────────────────────────────────
    # Audio playback (sounddevice)
    # ────────────────────────────────────────────────────────────
    _audio_data = None
    _audio_sr = 44100

    def _get_audio_cache_key(self, clip):
        """Generate a stable cache key for an audio clip based on its effects."""
        import hashlib
        fx_str = str(sorted(clip.audio_effects.items()))
        fx_hash = hashlib.md5(fx_str.encode()).hexdigest()[:8]
        return (clip.source_path, clip.source_offset, clip.source_duration, fx_hash)

    def _invalidate_audio_cache(self, clip=None):
        """Clear processed audio cache. If clip is provided, clear only that clip."""
        if clip is None:
            self._processed_audio_cache.clear()
        else:
            key = self._get_audio_cache_key(clip)
            self._processed_audio_cache.pop(key, None)

    def _ensure_audio_cached(self, clip):
        """Ensure a clip's processed audio is in cache, return (samples, sr, nch)."""
        key = self._get_audio_cache_key(clip)
        if key in self._processed_audio_cache:
            return self._processed_audio_cache[key]

        # Process and cache
        samples, sr, nch = process_audio_clip(
            clip.source_path,
            clip.audio_effects,
            source_offset=clip.source_offset,
            source_duration=clip.source_duration,
        )
        self._processed_audio_cache[key] = (samples, sr, nch)
        return samples, sr, nch

    def _build_audio_cache(self):
        """Pre-process all audio clips in background thread."""
        if not self.audio_clips:
            return

        clips_to_cache = []
        for clip in self.audio_clips:
            if not os.path.isfile(clip.source_path):
                continue
            key = self._get_audio_cache_key(clip)
            if key not in self._processed_audio_cache:
                clips_to_cache.append(clip)

        if not clips_to_cache:
            return

        def _cache_worker():
            for clip in clips_to_cache:
                try:
                    key = self._get_audio_cache_key(clip)
                    if key not in self._processed_audio_cache:
                        samples, sr, nch = process_audio_clip(
                            clip.source_path,
                            clip.audio_effects,
                            source_offset=clip.source_offset,
                            source_duration=clip.source_duration,
                        )
                        self._processed_audio_cache[key] = (samples, sr, nch)
                except Exception as e:
                    print(f"[AudioCache] Failed to cache {clip.source_path}: {e}")
            self.after(0, lambda: self._status_var.set("Audio effects cached"))

        self._status_var.set("Caching audio effects...")
        threading.Thread(target=_cache_worker, daemon=True).start()

    def _audio_play_from(self, start_sec):
        """Start playing all audio clip layers mixed from *start_sec*."""
        if not _HAS_AUDIO:
            print("Audio not available (_HAS_AUDIO=False)")
            return
        if not self.audio_clips and not self.audio_path:
            print("No audio clips or audio_path to play")
            return
        try:
            self._audio_stop()
            import numpy as np

            total_dur = self.timeline_duration
            if start_sec >= total_dur:
                return

            # Mix all audio clips together
            clips_data = []
            for clip in self.audio_clips:
                if not os.path.isfile(clip.source_path):
                    continue
                # Skip clips that end before start_sec
                if clip.end <= start_sec:
                    continue
                try:
                    # Use cached processed audio
                    samples, sr, nch = self._ensure_audio_cached(clip)
                    clips_data.append({
                        'samples': samples,
                        'sr': sr,
                        'nch': nch,
                        'start': clip.start,
                    })
                except Exception as e:
                    print(f"Warning: audio clip processing failed "
                          f"({clip.source_path}): {e}")

            if not clips_data:
                # Fallback: play raw audio_path if available
                if self.audio_path:
                    print(f"Loading audio from: {self.audio_path}")
                    audio_data, sr = sf.read(self.audio_path, dtype='float32')
                    print(f"Loaded audio: shape={audio_data.shape}, sr={sr}")
                    start_frame = max(0, int(start_sec * sr))
                    if start_frame >= len(audio_data):
                        return
                    audio_data = audio_data[start_frame:]
                    self._audio_data = audio_data
                    self._audio_sr = sr
                else:
                    return
            else:
                # Mix and trim from start_sec
                mixed, sr, nch = mix_audio_clips(clips_data, total_dur)

                # Apply project-level EQ
                if any(abs(g) > 0.01 for g in self._project_eq):
                    mixed = apply_eq(mixed, sr, self._project_eq)
                    mixed = np.clip(mixed, -1.0, 1.0)

                # Trim to start from start_sec
                start_sample = int(start_sec * sr)
                if start_sample >= mixed.shape[0]:
                    return
                mixed = mixed[start_sample:]

                self._audio_data = mixed.astype(np.float32)
                self._audio_sr = sr

            # Play using sd.play() - simple and reliable
            print(f"Playing audio: shape={self._audio_data.shape}, sr={self._audio_sr}")
            sd.play(self._audio_data, self._audio_sr)
            print("Audio playback started")
        except Exception as e:
            import traceback
            print(f"Warning: audio playback failed: {e}")
            traceback.print_exc()

    def _audio_stop(self):
        """Stop audio playback."""
        if not _HAS_AUDIO:
            return
        try:
            sd.stop()
        except Exception:
            pass

    def _audio_cleanup(self):
        """Clean up audio resources on exit."""
        self._audio_stop()
        self._audio_data = None

    # ────────────────────────────────────────────────────────────
    # Project save / load
    # ────────────────────────────────────────────────────────────
    _project_path: str | None = None

    def _get_project_dict(self):
        """Serialize the full project state to a dict."""
        self._sync_style_vars()
        return {
            "version": 2,
            "audio_path": self.audio_path,
            "lyrics_path": self.lyrics_path,
            "video_source_path": self.video_source_path,
            "video_source_type": self.video_source_type,
            "video_clips": [c.as_dict() for c in self.video_clips],
            "audio_clips": [c.as_dict() for c in self.audio_clips],
            "video_track_count": self._video_track_count,
            "subtitles": [
                {"text": c.text, "start": c.start, "end": c.end, "idx": c.idx, "font_size": c.font_size}
                for c in self.subtitles
            ],
            "style": self._get_style_dict(),
            "project_eq": list(self._project_eq),
            "zoom": self._px_per_sec,
            "playback_time": self._playback_time,
        }

    def _apply_project_dict(self, data):
        """Restore project state from a dict."""
        # Stop playback
        self._stop_playback()

        # Clear caches for new project
        self._audio_waveforms.clear()
        self._waveform_images.clear()
        self._processed_audio_cache.clear()

        # ── Media ──
        audio = data.get("audio_path")
        if audio and os.path.isfile(audio):
            self.audio_path = audio
            self.audio_duration = get_wav_duration(audio)
            name = os.path.basename(audio)
            self._lbl_audio.config(
                text=f"Audio: {name} ({self.audio_duration:.1f}s)", fg=C.TEXT)
        else:
            self.audio_path = None
            self.audio_duration = 0.0
            self._lbl_audio.config(text="Audio: (none)", fg=C.TEXT_DIM)

        self.lyrics_path = data.get("lyrics_path")

        vs_path = data.get("video_source_path")
        vs_type = data.get("video_source_type")
        if vs_path and os.path.isfile(vs_path):
            if vs_type == "video":
                self._load_video_file(vs_path)
            else:
                self._load_image_file(vs_path)
        else:
            self.video_source_path = None
            self.video_source_type = None
            self.video_source_duration = 0.0
            self._video_clip = None
            self._bg_image_pil = None
            self._lbl_video.config(text="Video: (none)", fg=C.TEXT_DIM)

        # ── Subtitles ──
        self.subtitles.clear()
        for sd in data.get("subtitles", []):
            self.subtitles.append(
                SubtitleClip(sd["text"], sd["start"], sd["end"],
                             sd.get("idx", 0), sd.get("font_size")))
        self.selected_clip = None
        self.selected_media_clip = None
        self._selected_track = None

        # ── Media clips (version 2+) ──
        video_clips_data = data.get("video_clips", [])
        if video_clips_data:
            self.video_clips = [
                MediaClip(
                    source_path=c["source_path"],
                    source_type=c["source_type"],
                    start=c["start"],
                    end=c["end"],
                    source_offset=c.get("source_offset", 0.0),
                    source_duration=c.get("source_duration", 0.0),
                    track_index=c.get("track_index", 0),
                )
                for c in video_clips_data
            ]
            # Restore track count from data or infer from highest track_index
            saved_vtc = data.get("video_track_count")
            if saved_vtc:
                self._video_track_count = saved_vtc
            else:
                max_idx = max(c.track_index for c in self.video_clips)
                self._video_track_count = max(1, max_idx + 1)
        else:
            self._video_track_count = data.get("video_track_count", 1)
        audio_clips_data = data.get("audio_clips", [])
        if audio_clips_data:
            self.audio_clips = [
                MediaClip(
                    source_path=c["source_path"],
                    source_type=c["source_type"],
                    start=c["start"],
                    end=c["end"],
                    source_offset=c.get("source_offset", 0.0),
                    source_duration=c.get("source_duration", 0.0),
                    track_index=c.get("track_index", 0),
                    audio_effects={**_DEFAULT_AUDIO_FX, **c.get("audio_effects", {})},
                )
                for c in audio_clips_data
            ]
            # Restore track count from highest track_index
            max_idx = max(c.track_index for c in self.audio_clips)
            self._audio_track_count = max(1, max_idx + 1)
        else:
            self._audio_track_count = 1

        # Generate waveforms for all audio clips
        if self.audio_clips:
            self._generate_audio_waveform()
            self._build_audio_cache()

        # ── Style ──
        style = data.get("style")
        if style:
            self._apply_preset(style)

        # ── Project EQ ──
        self._project_eq = list(data.get("project_eq", DEFAULT_EQ_BANDS))

        # ── Misc ──
        self._px_per_sec = data.get("zoom", 60.0)
        self._zoom_var.set(self._px_per_sec)
        self._playback_time = data.get("playback_time", 0.0)

        self._refresh_preview()
        self._update_timeline_height()
        self._redraw_timeline()
        self._update_detail_panel()

    def _new_project(self):
        """Reset to a blank project."""
        if self.subtitles:
            if not messagebox.askyesno(
                    "New Project",
                    "Discard current project? Unsaved changes will be lost."):
                return
        self._stop_playback()
        self._project_path = None
        self.audio_path = None
        self.audio_duration = 0.0
        self.lyrics_path = None
        self.video_source_path = None
        self.video_source_type = None
        self.video_source_duration = 0.0
        if self._video_clip is not None:
            try:
                self._video_clip.reader.close()
            except Exception:
                pass
        self._video_clip = None
        self._bg_image_pil = None
        self._video_thumbnails.clear()
        self._video_thumb_strip = None
        self._audio_waveforms.clear()
        self._waveform_images.clear()
        self._processed_audio_cache.clear()
        self.subtitles.clear()
        self.video_clips.clear()
        self.audio_clips.clear()
        self.selected_clip = None
        self.selected_media_clip = None
        self._selected_track = None
        self._audio_track_count = 1
        self._project_eq = list(DEFAULT_EQ_BANDS)
        self._playback_time = 0.0
        self._lbl_audio.config(text="Audio: (none)", fg=C.TEXT_DIM)
        self._lbl_video.config(text="Video: (none)", fg=C.TEXT_DIM)
        self.title("SubEditor \u2014 Subtitle Video Editor")
        self._status_var.set("New project")
        self._refresh_preview()
        self._update_timeline_height()
        self._redraw_timeline()
        self._update_detail_panel()

    def _open_project(self):
        path = filedialog.askopenfilename(
            title="Open Project",
            filetypes=[("SubEditor Project", "*.subproj"),
                       ("JSON", "*.json"), ("All", "*.*")])
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            messagebox.showerror("Open Error", f"Could not load project:\n{e}")
            return
        self._project_path = path
        self._apply_project_dict(data)
        self.title(f"SubEditor \u2014 {os.path.basename(path)}")
        self._status_var.set(f"Opened project: {os.path.basename(path)}")

    def _save_project(self):
        if self._project_path:
            self._write_project(self._project_path)
        else:
            self._save_project_as()

    def _save_project_as(self):
        path = filedialog.asksaveasfilename(
            title="Save Project As",
            defaultextension=".subproj",
            filetypes=[("SubEditor Project", "*.subproj"),
                       ("JSON", "*.json"), ("All", "*.*")])
        if not path:
            return
        self._project_path = path
        self._write_project(path)

    def _write_project(self, path):
        try:
            data = self._get_project_dict()
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            self.title(f"SubEditor \u2014 {os.path.basename(path)}")
            self._status_var.set(f"Saved: {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("Save Error", f"Could not save project:\n{e}")

    # ────────────────────────────────────────────────────────────
    # File operations
    # ────────────────────────────────────────────────────────────
    def _open_audio(self):
        path = filedialog.askopenfilename(
            title="Open Audio",
            filetypes=[("WAV files", "*.wav"), ("All", "*.*")])
        if not path:
            return
        self.audio_path = path
        self.audio_duration = get_wav_duration(path)
        # Create audio clip spanning the full duration on track 0
        self.audio_clips = [MediaClip(
            source_path=path,
            source_type="audio",
            start=0.0,
            end=self.audio_duration,
            source_offset=0.0,
            source_duration=self.audio_duration,
            track_index=0
        )]
        self._audio_track_count = max(1, self._audio_track_count)
        name = os.path.basename(path)
        self._lbl_audio.config(
            text=f"Audio: {name} ({self.audio_duration:.1f}s)",
            fg=C.TEXT)
        self._status_var.set(
            f"Loaded audio: {name} ({self.audio_duration:.1f}s)")
        self._generate_audio_waveform()
        self._build_audio_cache()
        self._update_timeline_height()
        self._redraw_timeline()

    def _add_audio_layer(self):
        """Add an additional audio file on a new audio track layer."""
        path = filedialog.askopenfilename(
            title="Add Audio Layer",
            filetypes=[("WAV files", "*.wav"), ("All", "*.*")])
        if not path:
            return
        dur = get_wav_duration(path)
        track_idx = self._audio_track_count
        self._audio_track_count += 1
        clip = MediaClip(
            source_path=path, source_type="audio",
            start=0.0, end=dur,
            source_offset=0.0, source_duration=dur,
            track_index=track_idx
        )
        self.audio_clips.append(clip)
        # If no primary audio set, use the first one
        if not self.audio_path:
            self.audio_path = path
            self.audio_duration = dur
            self._lbl_audio.config(
                text=f"Audio: {os.path.basename(path)} ({dur:.1f}s)",
                fg=C.TEXT)
        # Always generate waveform for new audio
        self._generate_audio_waveform()
        self._build_audio_cache()
        self._update_timeline_height()
        self._redraw_timeline()
        self._status_var.set(
            f"Added audio layer {track_idx + 1}: {os.path.basename(path)}")

    def _open_lyrics(self):
        path = filedialog.askopenfilename(
            title="Open Lyrics",
            filetypes=[("Text files", "*.txt"), ("All", "*.*")])
        if not path:
            return
        # Ask to clear existing subtitles
        if self.subtitles:
            if not messagebox.askyesno(
                    "Clear Subtitles",
                    "Clear existing subtitles and generate new ones?"):
                return
        self.lyrics_path = path
        self._lbl_lyrics.config(
            text=f"Lyrics: {os.path.basename(path)}", fg=C.TEXT)
        self._status_var.set(
            f"Loaded lyrics: {os.path.basename(path)}")
        # Auto-generate if audio is loaded
        if self.audio_path:
            self._generate_subtitles()

    def _open_video_source(self):
        path = filedialog.askopenfilename(
            title="Open Video or Background Image",
            filetypes=[
                ("Media files",
                 "*.mp4;*.avi;*.mkv;*.mov;*.wmv;*.webm;*.m4v;"
                 "*.png;*.jpg;*.jpeg;*.bmp;*.tiff;*.webp"),
                ("Video files",
                 "*.mp4;*.avi;*.mkv;*.mov;*.wmv;*.webm;*.m4v"),
                ("Image files",
                 "*.png;*.jpg;*.jpeg;*.bmp;*.tiff;*.webp"),
                ("All", "*.*"),
            ])
        if not path:
            return
        ext = os.path.splitext(path)[1].lower()
        if ext in _VIDEO_EXTS:
            self._load_video_file(path)
        elif ext in _IMAGE_EXTS:
            self._load_image_file(path)
        else:
            messagebox.showwarning(
                "Unsupported", f"Unrecognised file type: {ext}")

    def _load_video_file(self, path):
        try:
            try:
                from moviepy import VideoFileClip
            except ModuleNotFoundError:
                from moviepy.editor import VideoFileClip  # type: ignore
            # Close previous clip
            if self._video_clip is not None:
                try:
                    self._video_clip.reader.close()
                    if self._video_clip.audio:
                        self._video_clip.audio.reader.close_proc()
                except Exception:
                    pass
            clip = VideoFileClip(path)
            self._video_clip = clip
            self._bg_image_pil = None
            duration = clip.duration
        except Exception as e:
            messagebox.showerror(
                "Video Error", f"Could not load video:\n{e}")
            return

        self.video_source_path = path
        self.video_source_type = "video"
        self.video_source_duration = duration
        # Create video clip spanning the full duration
        self.video_clips = [MediaClip(
            source_path=path,
            source_type="video",
            start=0.0,
            end=duration,
            source_offset=0.0,
            source_duration=duration
        )]
        name = os.path.basename(path)
        self._lbl_video.config(
            text=f"Video: {name} ({duration:.1f}s)", fg=C.TEXT)
        self._status_var.set(f"Loaded video: {name} ({duration:.1f}s)")
        self._playback_time = 0.0
        self._generate_video_thumbnails()
        self._refresh_preview()
        self._redraw_timeline()

    def _load_image_file(self, path):
        from PIL import Image
        self.video_source_path = path
        self.video_source_type = "image"
        self.video_source_duration = 0.0
        self._video_clip = None
        self._bg_image_pil = Image.open(path).convert("RGB")
        # For images, create a clip spanning the timeline duration
        img_dur = self.audio_duration if self.audio_duration > 0 else 10.0
        self.video_clips = [MediaClip(
            source_path=path,
            source_type="image",
            start=0.0,
            end=img_dur,
            source_offset=0.0,
            source_duration=img_dur
        )]
        name = os.path.basename(path)
        self._lbl_video.config(text=f"Image: {name}", fg=C.TEXT)
        self._status_var.set(f"Loaded background image: {name}")
        self._playback_time = 0.0
        self._generate_image_thumb_strip()
        self._refresh_preview()
        self._redraw_timeline()

    def _import_srt(self):
        path = filedialog.askopenfilename(
            title="Import SRT",
            filetypes=[("SRT files", "*.srt"), ("All", "*.*")])
        if not path:
            return
        from video_generator import parse_srt
        srt_data = parse_srt(path)
        if not srt_data:
            messagebox.showinfo("Import SRT", "No subtitles found in file.")
            return
        self.subtitles.clear()
        for i, entry in enumerate(srt_data):
            self.subtitles.append(SubtitleClip(
                text=entry["text"], start=entry["start"],
                end=entry["end"], idx=i))
        self.selected_clip = self.subtitles[0] if self.subtitles else None
        self._selected_track = "sub" if self.selected_clip else None
        self._update_detail_panel()
        self._redraw_timeline()
        self._refresh_preview()
        self._status_var.set(
            f"Imported {len(self.subtitles)} subtitles from SRT")

    def _export_srt(self):
        if not self.subtitles:
            messagebox.showinfo("Export SRT", "No subtitles to export.")
            return
        path = filedialog.asksaveasfilename(
            title="Export SRT", defaultextension=".srt",
            filetypes=[("SRT files", "*.srt")])
        if not path:
            return
        sorted_subs = sorted(self.subtitles, key=lambda c: c.start)
        subs = []
        for i, s in enumerate(sorted_subs, 1):
            d = s.as_dict()
            d["index"] = i
            subs.append(d)
        write_srt(subs, path)
        self._status_var.set(
            f"Exported {len(subs)} subtitles to "
            f"{os.path.basename(path)}")

    # ────────────────────────────────────────────────────────────
    # Generate subtitles
    # ────────────────────────────────────────────────────────────
    def _generate_subtitles(self):
        if not self.audio_path:
            messagebox.showwarning("Generate",
                                   "Please open an audio file first.")
            return
        if not self.lyrics_path:
            messagebox.showwarning("Generate",
                                   "Please open a lyrics file first.")
            return

        self._status_var.set("Generating subtitles\u2026")
        self.update_idletasks()

        lyrics = parse_lyrics(self.lyrics_path)
        duration = self.audio_duration
        # Get vocal start time from user input
        try:
            vocal_start = float(self._vocal_start_var.get())
        except (ValueError, AttributeError):
            vocal_start = 0.0

        subtitles = generate_lyrics_timing(lyrics, self.audio_path, duration)
        subtitles = align_subtitles_to_audio(
            subtitles, self.audio_path, duration, vocal_start=vocal_start)

        self.subtitles.clear()
        for i, sub in enumerate(subtitles):
            self.subtitles.append(SubtitleClip(
                text=sub["text"], start=sub["start"],
                end=sub["end"], idx=i))
        self.selected_clip = None
        self._selected_track = None
        self._status_var.set(
            f"Generated {len(self.subtitles)} subtitle clips")
        self._redraw_timeline()

    def _clear_subtitles(self):
        """Clear all subtitle clips."""
        if not self.subtitles:
            self._status_var.set("No subtitles to clear")
            return
        if messagebox.askyesno("Clear Subtitles",
                               f"Clear all {len(self.subtitles)} subtitle clips?"):
            self.subtitles.clear()
            self.selected_clip = None
            self._selected_track = None
            self._update_detail_panel()
            self._redraw_timeline()
            self._status_var.set("All subtitles cleared")

    # ────────────────────────────────────────────────────────────
    # Export video
    # ────────────────────────────────────────────────────────────
    def _export_video(self):
        if not self.audio_path:
            messagebox.showwarning("Export", "No audio loaded.")
            return
        if not self.subtitles:
            messagebox.showwarning("Export", "No subtitles to render.")
            return
        if not self.video_source_path:
            self._open_video_source()
            if not self.video_source_path:
                return

        # Show export settings dialog
        self._show_export_dialog()

    def _show_export_dialog(self):
        """Show a dark-mode export settings dialog."""
        dlg = tk.Toplevel(self)
        dlg.title("Export Video")
        dlg.configure(bg=C.BG)
        dlg.transient(self)
        dlg.grab_set()
        dlg.resizable(False, False)

        # Center on parent
        dlg.update_idletasks()
        pw, ph = self.winfo_width(), self.winfo_height()
        px, py = self.winfo_x(), self.winfo_y()
        dw, dh = 420, 520
        dlg.geometry(f"{dw}x{dh}+{px + (pw - dw) // 2}+{py + (ph - dh) // 2}")

        # Variables
        v_format = tk.StringVar(value="mp4")
        v_fps = tk.IntVar(value=30)
        v_resolution = tk.StringVar(value="1920x1080")
        v_codec = tk.StringVar(value="libx264")
        v_preset = tk.StringVar(value="medium")
        v_crf = tk.IntVar(value=80)
        v_audio_codec = tk.StringVar(value="aac")
        v_audio_bitrate = tk.StringVar(value="320k")
        v_threads = tk.IntVar(value=4)
        v_output_path = tk.StringVar()

        # Format configs
        FORMAT_CONFIGS = {
            "mp4": {"ext": ".mp4", "codecs": ["libx264", "libx265", "mpeg4"], "audio": ["aac", "mp3"]},
            "avi": {"ext": ".avi", "codecs": ["mpeg4", "mjpeg", "rawvideo"], "audio": ["mp3", "pcm_s16le"]},
            "mkv": {"ext": ".mkv", "codecs": ["libx264", "libx265", "vp9"], "audio": ["aac", "opus", "flac"]},
            "webm": {"ext": ".webm", "codecs": ["vp8", "vp9"], "audio": ["opus", "vorbis"]},
            "mov": {"ext": ".mov", "codecs": ["libx264", "prores"], "audio": ["aac", "pcm_s16le"]},
            "gif": {"ext": ".gif", "codecs": ["gif"], "audio": []},
        }

        lbl_kw = dict(bg=C.BG, fg=C.TEXT, font=("Segoe UI", 9))
        lbl_dim_kw = dict(bg=C.BG, fg=C.TEXT_DIM, font=("Segoe UI", 8))
        entry_kw = dict(bg=C.SURFACE, fg=C.TEXT, insertbackground=C.TEXT,
                        font=("Segoe UI", 9), relief="flat", bd=0,
                        highlightthickness=1, highlightbackground=C.BORDER)

        # ── Title ──
        tk.Label(dlg, text="\U0001F4F9 Export Video Settings",
                 bg=C.BG, fg=C.TEXT, font=("Segoe UI", 12, "bold")).pack(
            pady=(15, 10))

        # Main content frame
        content = tk.Frame(dlg, bg=C.BG)
        content.pack(fill=tk.BOTH, expand=True, padx=20, pady=5)

        row = 0

        # ── Video Format ──
        tk.Label(content, text="Video Format", **lbl_kw).grid(
            row=row, column=0, sticky="w", pady=(0, 5))
        fmt_frame = tk.Frame(content, bg=C.BG)
        fmt_frame.grid(row=row, column=1, sticky="ew", pady=(0, 5))
        for fmt in ["mp4", "avi", "mkv", "webm", "mov", "gif"]:
            tk.Radiobutton(fmt_frame, text=fmt.upper(), variable=v_format,
                           value=fmt, bg=C.BG, fg=C.TEXT, selectcolor=C.SURFACE,
                           activebackground=C.BG, activeforeground=C.ACCENT,
                           font=("Segoe UI", 8), indicatoron=True,
                           command=lambda: _on_format_change()).pack(side=tk.LEFT, padx=2)
        row += 1

        # ── Resolution ──
        tk.Label(content, text="Resolution", **lbl_kw).grid(
            row=row, column=0, sticky="w", pady=5)
        res_combo = ttk.Combobox(content, textvariable=v_resolution, width=18,
                                 values=["1920x1080", "1280x720", "854x480",
                                         "640x360", "3840x2160", "2560x1440"])
        res_combo.grid(row=row, column=1, sticky="w", pady=5)
        row += 1

        # ── Frame Rate ──
        tk.Label(content, text="Frame Rate (FPS)", **lbl_kw).grid(
            row=row, column=0, sticky="w", pady=5)
        fps_frame = tk.Frame(content, bg=C.BG)
        fps_frame.grid(row=row, column=1, sticky="ew", pady=5)
        for fps in [24, 25, 30, 60]:
            tk.Radiobutton(fps_frame, text=str(fps), variable=v_fps, value=fps,
                           bg=C.BG, fg=C.TEXT, selectcolor=C.SURFACE,
                           activebackground=C.BG, activeforeground=C.ACCENT,
                           font=("Segoe UI", 8)).pack(side=tk.LEFT, padx=4)
        row += 1

        # ── Video Codec ──
        tk.Label(content, text="Video Codec", **lbl_kw).grid(
            row=row, column=0, sticky="w", pady=5)
        codec_combo = ttk.Combobox(content, textvariable=v_codec, width=18,
                                   values=["libx264", "libx265", "mpeg4"])
        codec_combo.grid(row=row, column=1, sticky="w", pady=5)
        row += 1

        # ── Preset (encoding speed) ──
        tk.Label(content, text="Encoding Preset", **lbl_kw).grid(
            row=row, column=0, sticky="w", pady=5)
        preset_combo = ttk.Combobox(content, textvariable=v_preset, width=18,
                                    values=["ultrafast", "superfast", "veryfast",
                                            "faster", "fast", "medium", "slow",
                                            "slower", "veryslow"])
        preset_combo.grid(row=row, column=1, sticky="w", pady=5)
        row += 1

        # ── Quality (CRF) ──
        tk.Label(content, text="Quality", **lbl_kw).grid(
            row=row, column=0, sticky="w", pady=5)
        crf_frame = tk.Frame(content, bg=C.BG)
        crf_frame.grid(row=row, column=1, sticky="ew", pady=5)
        crf_scale = ttk.Scale(crf_frame, from_=0, to=100, variable=v_crf,
                              orient=tk.HORIZONTAL, length=120)
        crf_scale.pack(side=tk.LEFT)
        crf_lbl = tk.Label(crf_frame, text="80", bg=C.BG, fg=C.TEXT_DIM,
                           font=("Segoe UI", 8), width=3)
        crf_lbl.pack(side=tk.LEFT, padx=5)
        tk.Label(crf_frame, text="(0=worst, 100=lossless)", **lbl_dim_kw).pack(side=tk.LEFT)

        def _update_crf_lbl(*_):
            crf_lbl.config(text=str(v_crf.get()))
        v_crf.trace_add("write", _update_crf_lbl)
        row += 1

        # ── Separator ──
        ttk.Separator(content, orient=tk.HORIZONTAL).grid(
            row=row, column=0, columnspan=2, sticky="ew", pady=10)
        row += 1

        # ── Audio Settings Header ──
        tk.Label(content, text="\U0001F3B5 Audio Settings",
                 bg=C.BG, fg=C.TEXT, font=("Segoe UI", 10, "bold")).grid(
            row=row, column=0, columnspan=2, sticky="w", pady=(0, 5))
        row += 1

        # ── Audio Codec ──
        tk.Label(content, text="Audio Codec", **lbl_kw).grid(
            row=row, column=0, sticky="w", pady=5)
        audio_codec_combo = ttk.Combobox(content, textvariable=v_audio_codec,
                                         width=18, values=["aac", "mp3", "opus", "flac"])
        audio_codec_combo.grid(row=row, column=1, sticky="w", pady=5)
        row += 1

        # ── Audio Bitrate ──
        tk.Label(content, text="Audio Bitrate", **lbl_kw).grid(
            row=row, column=0, sticky="w", pady=5)
        bitrate_combo = ttk.Combobox(content, textvariable=v_audio_bitrate,
                                     width=18, values=["128k", "192k", "256k", "320k"])
        bitrate_combo.grid(row=row, column=1, sticky="w", pady=5)
        row += 1

        # ── Threads ──
        tk.Label(content, text="Threads", **lbl_kw).grid(
            row=row, column=0, sticky="w", pady=5)
        threads_spin = tk.Spinbox(content, from_=1, to=16, textvariable=v_threads,
                                  width=5, bg=C.SURFACE, fg=C.TEXT,
                                  buttonbackground=C.SURFACE_LIGHT,
                                  font=("Segoe UI", 9))
        threads_spin.grid(row=row, column=1, sticky="w", pady=5)
        row += 1

        # Configure grid
        content.columnconfigure(1, weight=1)

        def _on_format_change():
            fmt = v_format.get()
            cfg = FORMAT_CONFIGS.get(fmt, FORMAT_CONFIGS["mp4"])
            codec_combo['values'] = cfg["codecs"]
            if v_codec.get() not in cfg["codecs"] and cfg["codecs"]:
                v_codec.set(cfg["codecs"][0])
            audio_codec_combo['values'] = cfg["audio"] if cfg["audio"] else ["none"]
            if v_audio_codec.get() not in cfg["audio"] and cfg["audio"]:
                v_audio_codec.set(cfg["audio"][0])
            elif not cfg["audio"]:
                v_audio_codec.set("none")

        # ── Progress section (initially hidden) ──
        progress_frame = tk.Frame(dlg, bg=C.BG)
        progress_var = tk.DoubleVar(value=0)
        progress_label = tk.Label(progress_frame, text="Preparing export...",
                                  bg=C.BG, fg=C.TEXT, font=("Segoe UI", 10))
        progress_label.pack(fill=tk.X, pady=(0, 10))
        # Custom green progress bar with dark border
        progress_container = tk.Frame(progress_frame, bg=C.BORDER, highlightthickness=1,
                                      highlightbackground=C.BORDER)
        progress_container.pack(fill=tk.X, pady=(0, 5))
        # Create a custom style for green progress bar
        style = ttk.Style()
        style.configure("Green.Horizontal.TProgressbar",
                        troughcolor="#2a2a2a",
                        background="#22c55e",
                        darkcolor="#16a34a",
                        lightcolor="#4ade80",
                        borderwidth=0,
                        thickness=24)
        progress_bar = ttk.Progressbar(progress_container, variable=progress_var,
                                       maximum=100, length=320, mode='determinate',
                                       style="Green.Horizontal.TProgressbar")
        progress_bar.pack(fill=tk.X, ipady=6)
        # Progress info row: percentage, elapsed, eta
        progress_info = tk.Frame(progress_frame, bg=C.BG)
        progress_info.pack(fill=tk.X, pady=(5, 0))
        progress_pct = tk.Label(progress_info, text="0%",
                                bg=C.BG, fg=C.TEXT, font=("Segoe UI", 9, "bold"))
        progress_pct.pack(side=tk.LEFT)
        progress_elapsed = tk.Label(progress_info, text="Elapsed: 0:00",
                                    bg=C.BG, fg=C.TEXT_DIM, font=("Segoe UI", 8))
        progress_elapsed.pack(side=tk.LEFT, padx=(15, 0))
        progress_eta = tk.Label(progress_info, text="ETA: --:--",
                                bg=C.BG, fg=C.TEXT_DIM, font=("Segoe UI", 8))
        progress_eta.pack(side=tk.RIGHT)
        export_start_time = [None]

        # ── Buttons ──
        btn_frame = tk.Frame(dlg, bg=C.BG)
        btn_frame.pack(fill=tk.X, padx=20, pady=15)

        def _browse_output():
            fmt = v_format.get()
            ext = FORMAT_CONFIGS.get(fmt, {}).get("ext", ".mp4")
            path = filedialog.asksaveasfilename(
                title="Save Video As",
                defaultextension=ext,
                filetypes=[(fmt.upper(), f"*{ext}"), ("All files", "*.*")])
            if path:
                v_output_path.set(path)

        # Track export state
        export_cancelled = [False]
        export_thread = [None]

        def _cancel_export():
            if export_thread[0] and export_thread[0].is_alive():
                export_cancelled[0] = True
                progress_label.config(text="Cancelling...")
            else:
                dlg.destroy()

        def _do_export():
            # Get output path
            out_path = v_output_path.get()
            if not out_path:
                _browse_output()
                out_path = v_output_path.get()
                if not out_path:
                    return

            # Parse resolution
            try:
                res_parts = v_resolution.get().split("x")
                width, height = int(res_parts[0]), int(res_parts[1])
            except (ValueError, IndexError):
                width, height = 1920, 1080

            # Build settings dict
            # Convert quality (0-100, higher=better) to CRF (0-51, lower=better)
            actual_crf = int(51 * (100 - v_crf.get()) / 100)
            settings = {
                "format": v_format.get(),
                "fps": v_fps.get(),
                "width": width,
                "height": height,
                "codec": v_codec.get(),
                "preset": v_preset.get(),
                "crf": actual_crf,
                "audio_codec": v_audio_codec.get(),
                "audio_bitrate": v_audio_bitrate.get(),
                "threads": v_threads.get(),
            }

            # Show progress, hide settings - make dialog smaller
            for widget in content.winfo_children():
                widget.grid_remove()
            content.pack_forget()
            progress_frame.pack(fill=tk.X, padx=20, pady=15, before=btn_frame)
            export_btn.pack_forget()  # Hide export button
            cancel_btn.config(text="Cancel Export", width=15, pady=10)
            cancel_btn.pack_forget()
            cancel_btn.pack(side=tk.TOP, pady=(15, 0))  # Center cancel button
            # Resize dialog to be smaller
            dlg.update_idletasks()
            dlg.geometry("400x250")
            # Re-center on parent
            pw, ph = self.winfo_width(), self.winfo_height()
            px, py = self.winfo_rootx(), self.winfo_rooty()
            dlg.geometry(f"+{px + (pw - 400) // 2}+{py + (ph - 250) // 2}")

            self._sync_style_vars()
            export_start_time[0] = time.time()

            def _format_time(seconds):
                """Format seconds as M:SS or H:MM:SS."""
                if seconds < 0:
                    return "--:--"
                seconds = int(seconds)
                if seconds < 3600:
                    return f"{seconds // 60}:{seconds % 60:02d}"
                return f"{seconds // 3600}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"

            def _update_progress(pct, msg=""):
                if export_cancelled[0]:
                    return
                try:
                    progress_var.set(pct)
                    progress_pct.config(text=f"{int(pct)}%")
                    if msg:
                        progress_label.config(text=msg)
                    # Update elapsed time
                    if export_start_time[0]:
                        elapsed = time.time() - export_start_time[0]
                        progress_elapsed.config(text=f"Elapsed: {_format_time(elapsed)}")
                        # Calculate ETA based on progress
                        if pct > 0:
                            total_estimated = elapsed / (pct / 100)
                            remaining = total_estimated - elapsed
                            progress_eta.config(text=f"ETA: {_format_time(remaining)}")
                        else:
                            progress_eta.config(text="ETA: --:--")
                    dlg.update_idletasks()
                except tk.TclError:
                    pass  # Dialog closed

            def _export_worker():
                try:
                    self._render_video_with_progress(
                        out_path, settings, _update_progress, export_cancelled)
                    if not export_cancelled[0]:
                        dlg.after(0, lambda: _on_export_complete(out_path, None))
                    else:
                        dlg.after(0, lambda: _on_export_complete(None, "Export cancelled"))
                except InterruptedError:
                    # Cancellation - not an error
                    dlg.after(0, lambda: _on_export_complete(None, "Export cancelled"))
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    dlg.after(0, lambda err=str(e): _on_export_complete(None, err))

            def _on_export_complete(path, error):
                if error:
                    progress_label.config(text=f"Error: {error}", fg=C.ERROR)
                    cancel_btn.config(text="Close")
                    self._status_var.set("Export failed")
                else:
                    self._status_var.set(f"Video exported: {os.path.basename(path)}")
                    dlg.destroy()
                    messagebox.showinfo("Export", f"Video saved to:\n{path}")

            export_thread[0] = threading.Thread(target=_export_worker, daemon=True)
            export_thread[0].start()

        cancel_btn = tk.Button(btn_frame, text="Cancel", command=_cancel_export,
                  bg=C.SURFACE_LIGHT, fg=C.TEXT, relief="flat",
                  font=("Segoe UI", 9), padx=15, pady=5,
                  activebackground=C.SURFACE_HOVER,
                  cursor="hand2")
        cancel_btn.pack(side=tk.RIGHT, padx=5)

        export_btn = tk.Button(btn_frame, text="\U0001F4BE Export", command=_do_export,
                  bg=C.SUCCESS, fg="#fff", relief="flat",
                  font=("Segoe UI", 9, "bold"), padx=20, pady=5,
                  activebackground="#2ecc71",
                  cursor="hand2")
        export_btn.pack(side=tk.RIGHT, padx=5)

        # Initialize format-dependent options
        _on_format_change()

    def _render_video(self, out_path, settings=None):
        try:
            from moviepy import (
                ImageClip, AudioFileClip,
                CompositeVideoClip, VideoFileClip,
            )
        except ModuleNotFoundError:
            from moviepy.editor import (  # type: ignore
                ImageClip, AudioFileClip,
                CompositeVideoClip, VideoFileClip,
            )
        from PIL import Image
        import numpy as np

        # Use settings or defaults
        if settings is None:
            settings = {}
        width = settings.get("width", VIDEO_WIDTH)
        height = settings.get("height", VIDEO_HEIGHT)
        fps = settings.get("fps", VIDEO_FPS)
        codec = settings.get("codec", "libx264")
        preset = settings.get("preset", "medium")
        crf = settings.get("crf", 23)
        audio_codec = settings.get("audio_codec", "aac")
        audio_bitrate = settings.get("audio_bitrate", "192k")
        threads = settings.get("threads", 4)
        fmt = settings.get("format", "mp4")

        # Calculate duration as the longest track (using clip end times only, not source durations)
        duration_candidates = []
        if self.subtitles:
            duration_candidates.append(max(c.end for c in self.subtitles))
        if self.video_clips:
            duration_candidates.append(max(c.end for c in self.video_clips))
        if self.audio_clips:
            duration_candidates.append(max(c.end for c in self.audio_clips))
        duration = max(duration_candidates) if duration_candidates else 10.0

        if self.video_source_type == "video":
            bg_clip = VideoFileClip(self.video_source_path)
            if bg_clip.duration < duration:
                bg_clip = bg_clip.loop(duration=duration)
            else:
                bg_clip = bg_clip.subclip(0, duration)
            bg_clip = bg_clip.resize((width, height))
        else:
            bg_image = Image.open(
                self.video_source_path).convert("RGB")
            src_w, src_h = bg_image.size
            scale = min(width / src_w, height / src_h)
            nw, nh = int(src_w * scale), int(src_h * scale)
            bg_image = bg_image.resize((nw, nh), Image.LANCZOS)
            canvas_img = Image.new(
                "RGB", (width, height), (0, 0, 0))
            canvas_img.paste(bg_image,
                             ((width - nw) // 2,
                              (height - nh) // 2))
            bg_array = np.array(canvas_img)
            bg_clip = ImageClip(bg_array).with_duration(duration)

        sub_dicts = [s.as_dict()
                     for s in sorted(self.subtitles,
                                     key=lambda c: c.start)]
        subtitle_clips = self._build_styled_clips(
            sub_dicts, width, height)

        audio_clip = self._build_export_audio(duration)

        final = CompositeVideoClip(
            [bg_clip, *subtitle_clips],
            size=(width, height),
        ).with_duration(duration).with_fps(fps)

        # Handle GIF (no audio)
        if fmt == "gif":
            final.write_gif(out_path, fps=min(fps, 15), logger="bar")
        else:
            final = final.with_audio(audio_clip)
            # Build write_videofile kwargs
            write_kwargs = {
                "fps": fps,
                "codec": codec,
                "preset": preset,
                "threads": threads,
                "logger": "bar",
            }
            # Add CRF for x264/x265
            if codec in ("libx264", "libx265"):
                write_kwargs["ffmpeg_params"] = ["-crf", str(crf)]
            # Add audio settings if not none
            if audio_codec and audio_codec != "none":
                write_kwargs["audio_codec"] = audio_codec
                write_kwargs["audio_bitrate"] = audio_bitrate
            else:
                write_kwargs["audio"] = False

            final.write_videofile(out_path, **write_kwargs)

        audio_clip.close()

    def _render_video_with_progress(self, out_path, settings, progress_callback, cancel_flag):
        """Render video with progress updates."""
        try:
            from moviepy import (
                ImageClip, AudioFileClip,
                CompositeVideoClip, VideoFileClip,
            )
        except ModuleNotFoundError:
            from moviepy.editor import (  # type: ignore
                ImageClip, AudioFileClip,
                CompositeVideoClip, VideoFileClip,
            )
        from PIL import Image
        import numpy as np

        # Custom progress logger
        class ProgressLogger:
            def __init__(self, callback, cancel_flag, stage=""):
                self.callback = callback
                self.cancel_flag = cancel_flag
                self.stage = stage

            def __call__(self, **kw):
                return self

            def iter_bar(self, **kw):
                """Moviepy calls iter_bar(chunk=iterable) or iter_bar(t=iterable)."""
                # Get the iterable from the first kwarg value
                iterable = list(kw.values())[0] if kw else []
                total = len(iterable) if hasattr(iterable, '__len__') else 100
                for i, item in enumerate(iterable):
                    if self.cancel_flag[0]:
                        raise InterruptedError("Export cancelled")
                    pct = ((i + 1) / max(total, 1)) * 100
                    self.callback(pct, self.stage)
                    yield item

            def bars_callback(self, bar, attr, value, old_value=None):
                pass

        # Use settings or defaults
        if settings is None:
            settings = {}
        width = settings.get("width", VIDEO_WIDTH)
        height = settings.get("height", VIDEO_HEIGHT)
        fps = settings.get("fps", VIDEO_FPS)
        codec = settings.get("codec", "libx264")
        preset = settings.get("preset", "medium")
        crf = settings.get("crf", 23)
        audio_codec = settings.get("audio_codec", "aac")
        audio_bitrate = settings.get("audio_bitrate", "192k")
        threads = settings.get("threads", 4)
        fmt = settings.get("format", "mp4")

        # Calculate duration as the longest track (using clip end times only, not source durations)
        duration_candidates = []
        if self.subtitles:
            duration_candidates.append(max(c.end for c in self.subtitles))
        if self.video_clips:
            duration_candidates.append(max(c.end for c in self.video_clips))
        if self.audio_clips:
            duration_candidates.append(max(c.end for c in self.audio_clips))
        duration = max(duration_candidates) if duration_candidates else 10.0

        progress_callback(5, "Loading video clips...")
        if cancel_flag[0]:
            raise InterruptedError("Export cancelled")

        # Build all video clips for compositing
        video_layer_clips = []
        
        # First create a black background for the full duration
        black_bg = ImageClip(np.zeros((height, width, 3), dtype=np.uint8)).with_duration(duration)
        video_layer_clips.append(black_bg)
        
        # Sort video clips by track_index (lower = further back) then by start time
        sorted_video_clips = sorted(self.video_clips, key=lambda c: (getattr(c, 'track_index', 0), c.start))
        
        for vclip in sorted_video_clips:
            try:
                clip_duration = vclip.end - vclip.start
                if clip_duration <= 0:
                    continue
                    
                if vclip.source_type == "video":
                    # Load video file with source offset
                    src_clip = VideoFileClip(vclip.source_path)
                    # Apply source offset and duration
                    src_start = vclip.source_offset
                    src_end = min(src_start + clip_duration, src_clip.duration)
                    if src_end > src_start:
                        src_clip = src_clip.subclip(src_start, src_end)
                    # If clip is shorter than needed, loop it
                    if src_clip.duration < clip_duration:
                        src_clip = src_clip.loop(duration=clip_duration)
                    src_clip = src_clip.resize((width, height))
                else:
                    # Image clip
                    img = Image.open(vclip.source_path).convert("RGB")
                    src_w, src_h = img.size
                    scale = min(width / src_w, height / src_h)
                    nw, nh = int(src_w * scale), int(src_h * scale)
                    img = img.resize((nw, nh), Image.LANCZOS)
                    canvas_img = Image.new("RGB", (width, height), (0, 0, 0))
                    canvas_img.paste(img, ((width - nw) // 2, (height - nh) // 2))
                    src_clip = ImageClip(np.array(canvas_img)).with_duration(clip_duration)
                
                # Set the clip to start at the timeline position
                src_clip = src_clip.with_start(vclip.start).with_duration(clip_duration)
                video_layer_clips.append(src_clip)
            except Exception as e:
                print(f"[WARN] Failed to process video clip {vclip.source_path}: {e}")

        progress_callback(10, "Building subtitle clips...")
        if cancel_flag[0]:
            raise InterruptedError("Export cancelled")

        print(f"[DEBUG] self.subtitles count: {len(self.subtitles)}")
        sub_dicts = [s.as_dict()
                     for s in sorted(self.subtitles,
                                     key=lambda c: c.start)]
        print(f"[DEBUG] sub_dicts: {sub_dicts[:3]}...")  # Show first 3
        subtitle_clips = self._build_styled_clips(
            sub_dicts, width, height)

        progress_callback(15, "Processing audio...")
        if cancel_flag[0]:
            raise InterruptedError("Export cancelled")

        audio_clip = self._build_export_audio(duration)

        progress_callback(20, "Compositing video...")
        if cancel_flag[0]:
            raise InterruptedError("Export cancelled")

        print(f"[DEBUG] Compositing {len(video_layer_clips)} video clips + {len(subtitle_clips)} subtitle clips")
        final = CompositeVideoClip(
            [*video_layer_clips, *subtitle_clips],
            size=(width, height),
        ).with_duration(duration).with_fps(fps)

        # Handle GIF (no audio)
        if fmt == "gif":
            progress_callback(25, "Rendering GIF frames...")
            logger = ProgressLogger(
                lambda pct, msg: progress_callback(25 + pct * 0.75, "Rendering GIF..."),
                cancel_flag, "Rendering GIF...")
            final.write_gif(out_path, fps=min(fps, 15), logger=logger)
        else:
            final = final.with_audio(audio_clip)
            # Build write_videofile kwargs
            write_kwargs = {
                "fps": fps,
                "codec": codec,
                "preset": preset,
                "threads": threads,
                "logger": ProgressLogger(
                    lambda pct, msg: progress_callback(25 + pct * 0.75, "Encoding video..."),
                    cancel_flag, "Encoding video..."),
            }
            # Add CRF for x264/x265
            if codec in ("libx264", "libx265"):
                write_kwargs["ffmpeg_params"] = ["-crf", str(crf)]
            # Add audio settings if not none
            if audio_codec and audio_codec != "none":
                write_kwargs["audio_codec"] = audio_codec
                write_kwargs["audio_bitrate"] = audio_bitrate
            else:
                write_kwargs["audio"] = False

            final.write_videofile(out_path, **write_kwargs)

        audio_clip.close()
        progress_callback(100, "Export complete!")

    def _build_export_audio(self, duration):
        """Build the final mixed audio clip for export, applying all FX."""
        import numpy as np
        try:
            from moviepy import AudioFileClip, CompositeAudioClip, AudioClip
        except ModuleNotFoundError:
            from moviepy.editor import AudioFileClip, CompositeAudioClip, AudioClip  # type: ignore

        def _pad_audio_to_duration(audio_clip, target_duration):
            """Ensure audio clip is exactly the target duration."""
            if audio_clip.duration >= target_duration:
                return audio_clip.subclipped(0, target_duration)
            # Create silent padding and composite
            silence = AudioClip(lambda t: 0, duration=target_duration, fps=44100)
            return CompositeAudioClip([audio_clip, silence]).with_duration(target_duration)

        # If no audio clips with non-default effects and no project EQ, fast path
        has_fx = any(
            clip.audio_effects != _DEFAULT_AUDIO_FX
            for clip in self.audio_clips
        )
        has_project_eq = any(abs(g) > 0.01 for g in self._project_eq)
        if not has_fx and not has_project_eq and len(self.audio_clips) <= 1:
            audio = AudioFileClip(self.audio_path)
            return _pad_audio_to_duration(audio, duration)

        # Process each audio clip with its effects
        clips_data = []
        for clip in self.audio_clips:
            if not os.path.isfile(clip.source_path):
                continue
            try:
                samples, sr, nch = process_audio_clip(
                    clip.source_path,
                    clip.audio_effects,
                    source_offset=clip.source_offset,
                    source_duration=clip.source_duration,
                )
                clips_data.append({
                    'samples': samples,
                    'sr': sr,
                    'nch': nch,
                    'start': clip.start,
                })
            except Exception as e:
                print(f"Warning: Failed to process audio clip "
                      f"{clip.source_path}: {e}")

        if not clips_data:
            audio = AudioFileClip(self.audio_path)
            return _pad_audio_to_duration(audio, duration)

        # Mix all clips together
        mixed, sr, nch = mix_audio_clips(clips_data, duration)

        # Apply project-level EQ to the final mix
        if any(abs(g) > 0.01 for g in self._project_eq):
            mixed = apply_eq(mixed, sr, self._project_eq)
            mixed = np.clip(mixed, -1.0, 1.0)

        # Write to temp WAV and create AudioFileClip
        tmp = tempfile.NamedTemporaryFile(
            suffix=".wav", delete=False, prefix="subeditor_mix_")
        tmp.close()
        write_wav(tmp.name, mixed, sr, nch)
        return AudioFileClip(tmp.name)

    def _build_styled_clips(self, subtitles, width, height):
        from video_generator import _shift_clips
        SINGING_FILL = 0.60
        MIN_PER_WORD = 0.15
        MAX_PER_WORD = 0.50
        all_clips = []
        print(f"[DEBUG] Building styled clips for {len(subtitles)} subtitles")
        for sub in subtitles:
            sub_duration = sub["end"] - sub["start"]
            if sub_duration <= 0:
                print(f"[DEBUG] Skipping subtitle with duration <= 0: {sub}")
                continue
            text = sub["text"]
            if not text.strip():
                print(f"[DEBUG] Skipping empty text subtitle")
                continue
            # Apply text case transformation
            if self._text_case == "uppercase":
                text = text.upper()
            elif self._text_case == "lowercase":
                text = text.lower()
            n_words = len(text.split())
            singing_time = sub_duration * SINGING_FILL
            per_word = singing_time / max(n_words, 1)
            per_word = max(MIN_PER_WORD, min(per_word, MAX_PER_WORD))
            singing_time = n_words * per_word
            highlight_dur = max(0.8, min(singing_time, sub_duration * 0.95))
            stroke_w = self._stroke_width if self._stroke_enabled else 0
            # Use clip-specific font size if set, otherwise global
            clip_font_size = sub.get("font_size") or self._font_size
            print(f"[DEBUG] Processing subtitle: text='{text[:40]}...', duration={sub_duration:.2f}, font_size={clip_font_size}")
            try:
                word_clips = create_word_fancytext_adv(
                    text=text, duration=highlight_dur,
                    width=width, height=height,
                    font_size=clip_font_size, font=self._font_path,
                    text_color=self._text_color,
                    highlight_color=self._highlight_color,
                    stroke_color=self._stroke_color,
                    stroke_width=stroke_w,
                    max_words_per_line=self._max_words_per_line,
                    position_y_ratio=self._position_y_ratio,
                    position_x_ratio=self._position_x_ratio,
                    text_justify=self._text_justify,
                    shadow_enabled=self._shadow_enabled,
                    shadow_offset=self._shadow_offset,
                    shadow_blur=self._shadow_blur,
                    glow_enabled=self._glow_enabled,
                    glow_color=self._glow_color,
                    glow_size=max(1, self._glow_size // 2),
                    glow_opacity=min(1.0, self._glow_opacity / 100.0),
                    highlight_box_enabled=self._highlight_box_enabled,
                    highlight_box_color=self._highlight_box_color,
                    highlight_box_opacity=self._highlight_box_opacity,
                    gradient_overlay_enabled=self._gradient_enabled,
                    gradient_overlay_colors=(
                        self._gradient_color1, self._gradient_color2),
                    gradient_overlay_angle=-90,
                    gradient_overlay_opacity=self._gradient_opacity,
                    bevel_emboss_enabled=self._bevel_enabled,
                    bevel_emboss_style="inner_bevel",
                    bevel_emboss_depth=max(3, self._bevel_depth),
                    bevel_emboss_angle=135,
                    bevel_emboss_highlight_color=(255, 255, 255),
                    bevel_emboss_shadow_color=(0, 0, 0),
                    bevel_emboss_highlight_opacity=0.75,
                    bevel_emboss_shadow_opacity=0.75,
                    bevel_emboss_soften=3,
                    texture_enabled=self._texture_enabled,
                    texture_scale=self._texture_scale,
                    texture_opacity=self._texture_opacity,
                    texture_blend_mode=self._texture_blend_mode,
                    texture_image=self._texture_image,
                    chrome_enabled=self._chrome_enabled,
                    chrome_opacity=self._chrome_opacity,
                )
                # Extend ALL clips so text stays visible for the full subtitle duration
                if word_clips and sub_duration > highlight_dur:
                    extra = sub_duration - highlight_dur
                    for i_clip in range(len(word_clips)):
                        new_dur = word_clips[i_clip].duration + extra
                        word_clips[i_clip] = word_clips[i_clip].with_duration(new_dur)
                        if word_clips[i_clip].mask is not None:
                            word_clips[i_clip].mask = (
                                word_clips[i_clip].mask.with_duration(new_dur))
                if word_clips:
                    lead_in = highlight_dur * 0.05
                    all_clips.extend(
                        _shift_clips(word_clips, sub["start"] - lead_in))
                    print(f"[DEBUG] Created {len(word_clips)} clips for: {text[:30]}...")
                else:
                    print(f"[DEBUG] WARNING: No clips created for: {text[:30]}...")
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"Error building subtitle clip: {e}")
        print(f"[DEBUG] Total clips created: {len(all_clips)}")
        return all_clips

    def _sync_style_vars(self):
        try:
            self._font_size = self._font_size_var.get()
        except Exception:
            pass
        case_val = self._text_case_var.get()
        if case_val == "UPPERCASE":
            self._text_case = "uppercase"
        elif case_val == "lowercase":
            self._text_case = "lowercase"
        else:
            self._text_case = "normal"
        self._stroke_enabled = self._stroke_var.get()
        try:
            self._stroke_width = self._stroke_width_var.get()
        except Exception:
            pass
        self._shadow_enabled = self._shadow_var.get()
        try:
            self._shadow_offset = (self._shadow_ox_var.get(),
                                   self._shadow_oy_var.get())
            self._shadow_blur = self._shadow_blur_var.get()
        except Exception:
            pass
        self._highlight_box_enabled = self._hibox_var.get()
        try:
            self._highlight_box_opacity = self._hibox_opacity_var.get()
        except Exception:
            pass
        self._glow_enabled = self._glow_var.get()
        try:
            self._glow_size = self._glow_size_var.get()
        except Exception:
            pass
        try:
            self._glow_opacity = self._glow_opacity_var.get()
        except Exception:
            pass
        self._gradient_enabled = self._gradient_var.get()
        try:
            self._gradient_opacity = self._grad_opacity_var.get()
        except Exception:
            pass
        self._bevel_enabled = self._bevel_var.get()
        try:
            self._bevel_depth = self._bevel_depth_var.get()
        except Exception:
            pass
        self._chrome_enabled = self._chrome_var.get()
        try:
            self._chrome_opacity = self._chrome_opacity_var.get()
        except Exception:
            pass
        self._texture_enabled = self._texture_var.get()
        try:
            self._texture_scale = self._texture_scale_var.get()
        except Exception:
            pass
        try:
            self._texture_opacity = self._texture_opacity_var.get()
        except Exception:
            pass
        try:
            self._texture_blend_mode = self._texture_blend_var.get()
        except Exception:
            pass
        try:
            self._max_words_per_line = self._max_wpl_var.get()
        except Exception:
            pass
        try:
            self._position_x_ratio = self._x_ratio_var.get()
        except Exception:
            pass
        try:
            self._position_y_ratio = self._y_ratio_var.get()
        except Exception:
            pass
        try:
            justify = self._justify_var.get().lower()
            self._text_justify = justify if justify in ("left", "center", "right") else "center"
        except Exception:
            pass
        try:
            mode = self._highlight_mode_var.get().lower()
            self._highlight_mode = mode if mode in ("word", "character") else "word"
        except Exception:
            pass

    # ────────────────────────────────────────────────────────────
    # Color/font helpers
    # ────────────────────────────────────────────────────────────
    def _pick_color(self, attr_name, button):
        current = getattr(self, attr_name)
        result = colorchooser.askcolor(
            color=_rgb_to_hex(current), title="Choose Colour")
        if result and result[0]:
            rgb = tuple(int(c) for c in result[0])
            setattr(self, attr_name, rgb)
            button.config(bg=_rgb_to_hex(rgb))
            self._refresh_preview()

    def _on_font_change(self):
        """Called when font selection changes - resolve name to path and update."""
        font_name = self._font_family_var.get().strip()
        if not font_name:
            return
        
        # Try to resolve font name to path
        path = _find_font_by_name(font_name)
        if path and os.path.isfile(path):
            self._font_path = path
            self._refresh_preview()
        else:
            # Check if it's already a full path
            if os.path.isfile(font_name):
                self._font_path = font_name
                self._refresh_preview()

    def _load_texture_file(self):
        """Load a custom texture image file."""
        from PIL import Image
        path = filedialog.askopenfilename(
            title="Load Texture Image",
            filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif"),
                       ("All", "*.*")])
        if not path:
            return
        try:
            tex = Image.open(path).convert("RGB")
            self._texture_image = tex
            self._texture_path = path
            self._status_var.set(f"Loaded texture: {os.path.basename(path)}")
            self._trigger_style_refresh()
        except Exception as e:
            messagebox.showerror("Load Error", f"Could not load texture:\n{e}")

    # ────────────────────────────────────────────────────────────
    # Style presets
    # ────────────────────────────────────────────────────────────
    def _get_style_dict(self):
        """Collect current style settings into a dict."""
        self._sync_style_vars()
        return {
            "font_path": self._font_path,
            "font_size": self._font_size,
            "text_case": self._text_case,
            "text_color": list(self._text_color),
            "highlight_color": list(self._highlight_color),
            "stroke_color": list(self._stroke_color),
            "stroke_width": self._stroke_width,
            "stroke_enabled": self._stroke_enabled,
            "shadow_enabled": self._shadow_enabled,
            "shadow_offset": list(self._shadow_offset),
            "shadow_blur": self._shadow_blur,
            "highlight_box_enabled": self._highlight_box_enabled,
            "highlight_box_color": list(self._highlight_box_color),
            "highlight_box_opacity": self._highlight_box_opacity,
            "max_words_per_line": self._max_words_per_line,
            "position_x_ratio": self._position_x_ratio,
            "position_y_ratio": self._position_y_ratio,
            "text_justify": self._text_justify,
            "highlight_mode": self._highlight_mode,
            "glow_enabled": self._glow_enabled,
            "glow_color": list(self._glow_color),
            "glow_size": self._glow_size,
            "glow_opacity": self._glow_opacity,
            "gradient_enabled": self._gradient_enabled,
            "gradient_color1": list(self._gradient_color1),
            "gradient_color2": list(self._gradient_color2),
            "gradient_opacity": self._gradient_opacity,
            "bevel_enabled": self._bevel_enabled,
            "bevel_depth": self._bevel_depth,
            "chrome_enabled": self._chrome_enabled,
            "chrome_opacity": self._chrome_opacity,
            "texture_enabled": self._texture_enabled,
            "texture_scale": self._texture_scale,
            "texture_opacity": self._texture_opacity,
            "texture_blend_mode": self._texture_blend_mode,
        }

    def _apply_preset(self, preset):
        """Apply a preset dict to the style settings and UI widgets."""
        def _t(v): return tuple(v) if isinstance(v, list) else v
        if "font_path" in preset:
            self._font_path = preset["font_path"]
            # Set combobox to font name without extension
            font_name = os.path.splitext(os.path.basename(self._font_path))[0]
            self._font_family_var.set(font_name)
        if "font_size" in preset:
            self._font_size = preset["font_size"]
            self._font_size_var.set(self._font_size)
        if "text_case" in preset:
            self._text_case = preset["text_case"]
            display = {"uppercase": "UPPERCASE", "lowercase": "lowercase"}.get(
                self._text_case, "Normal")
            self._text_case_var.set(display)
        if "text_color" in preset:
            self._text_color = _t(preset["text_color"])
            self._text_color_btn.config(bg=_rgb_to_hex(self._text_color))
        if "highlight_color" in preset:
            self._highlight_color = _t(preset["highlight_color"])
            self._highlight_color_btn.config(bg=_rgb_to_hex(self._highlight_color))
        if "stroke_color" in preset:
            self._stroke_color = _t(preset["stroke_color"])
            self._stroke_color_btn.config(bg=_rgb_to_hex(self._stroke_color))
        if "stroke_width" in preset:
            self._stroke_width = preset["stroke_width"]
            self._stroke_width_var.set(self._stroke_width)
        if "stroke_enabled" in preset:
            self._stroke_enabled = preset["stroke_enabled"]
            self._stroke_var.set(self._stroke_enabled)
        if "shadow_enabled" in preset:
            self._shadow_enabled = preset["shadow_enabled"]
            self._shadow_var.set(self._shadow_enabled)
        if "shadow_offset" in preset:
            self._shadow_offset = _t(preset["shadow_offset"])
            self._shadow_ox_var.set(self._shadow_offset[0])
            self._shadow_oy_var.set(self._shadow_offset[1])
        if "shadow_blur" in preset:
            self._shadow_blur = preset["shadow_blur"]
            self._shadow_blur_var.set(self._shadow_blur)
        if "highlight_box_enabled" in preset:
            self._highlight_box_enabled = preset["highlight_box_enabled"]
            self._hibox_var.set(self._highlight_box_enabled)
        if "highlight_box_color" in preset:
            self._highlight_box_color = _t(preset["highlight_box_color"])
            self._highlight_box_color_btn.config(
                bg=_rgb_to_hex(self._highlight_box_color))
        if "highlight_box_opacity" in preset:
            self._highlight_box_opacity = preset["highlight_box_opacity"]
            self._hibox_opacity_var.set(self._highlight_box_opacity)
        if "max_words_per_line" in preset:
            self._max_words_per_line = preset["max_words_per_line"]
            self._max_wpl_var.set(self._max_words_per_line)
        if "position_x_ratio" in preset:
            self._position_x_ratio = preset["position_x_ratio"]
            self._x_ratio_var.set(self._position_x_ratio)
        if "position_y_ratio" in preset:
            self._position_y_ratio = preset["position_y_ratio"]
            self._y_ratio_var.set(self._position_y_ratio)
        if "text_justify" in preset:
            self._text_justify = preset["text_justify"]
            self._justify_var.set(self._text_justify.capitalize())
        if "highlight_mode" in preset:
            self._highlight_mode = preset["highlight_mode"]
            self._highlight_mode_var.set(self._highlight_mode.capitalize())
        if "glow_enabled" in preset:
            self._glow_enabled = preset["glow_enabled"]
            self._glow_var.set(self._glow_enabled)
        if "glow_color" in preset:
            self._glow_color = _t(preset["glow_color"])
            self._glow_color_btn.config(bg=_rgb_to_hex(self._glow_color))
        if "glow_size" in preset:
            self._glow_size = preset["glow_size"]
            self._glow_size_var.set(self._glow_size)
        if "glow_opacity" in preset:
            self._glow_opacity = preset["glow_opacity"]
            self._glow_opacity_var.set(self._glow_opacity)
        if "gradient_enabled" in preset:
            self._gradient_enabled = preset["gradient_enabled"]
            self._gradient_var.set(self._gradient_enabled)
        if "gradient_color1" in preset:
            self._gradient_color1 = _t(preset["gradient_color1"])
            self._gradient_color1_btn.config(
                bg=_rgb_to_hex(self._gradient_color1))
        if "gradient_color2" in preset:
            self._gradient_color2 = _t(preset["gradient_color2"])
            self._gradient_color2_btn.config(
                bg=_rgb_to_hex(self._gradient_color2))
        if "gradient_opacity" in preset:
            self._gradient_opacity = preset["gradient_opacity"]
            self._grad_opacity_var.set(self._gradient_opacity)
        if "bevel_enabled" in preset:
            self._bevel_enabled = preset["bevel_enabled"]
            self._bevel_var.set(self._bevel_enabled)
        if "bevel_depth" in preset:
            self._bevel_depth = preset["bevel_depth"]
            self._bevel_depth_var.set(self._bevel_depth)
        if "chrome_enabled" in preset:
            self._chrome_enabled = preset["chrome_enabled"]
            self._chrome_var.set(self._chrome_enabled)
        if "chrome_opacity" in preset:
            self._chrome_opacity = preset["chrome_opacity"]
            self._chrome_opacity_var.set(self._chrome_opacity)
        if "texture_enabled" in preset:
            self._texture_enabled = preset["texture_enabled"]
            self._texture_var.set(self._texture_enabled)
        if "texture_scale" in preset:
            self._texture_scale = preset["texture_scale"]
            self._texture_scale_var.set(self._texture_scale)
        if "texture_opacity" in preset:
            self._texture_opacity = preset["texture_opacity"]
            self._texture_opacity_var.set(self._texture_opacity)
        if "texture_blend_mode" in preset:
            self._texture_blend_mode = preset["texture_blend_mode"]
            self._texture_blend_var.set(self._texture_blend_mode)
        self._status_var.set("Preset applied")
        self._refresh_preview()

    def _on_preset_combo(self, _event=None):
        """Handle selection from the preset dropdown."""
        name = self._preset_var.get()
        if name in self._all_presets:
            self._apply_preset(self._all_presets[name])

    def _refresh_preset_combo(self):
        """Re-scan the presets folder and update the dropdown."""
        self._all_presets = _load_all_presets()
        self._preset_combo["values"] = sorted(self._all_presets.keys())

    def _save_preset(self):
        """Prompt for a name with a dark-themed dialog and save preset."""
        dlg = tk.Toplevel(self)
        dlg.title("Save Preset")
        dlg.configure(bg=C.BG)
        dlg.resizable(False, False)
        dlg.transient(self)
        dlg.grab_set()
        # Centre on parent
        dlg.update_idletasks()
        pw, ph = self.winfo_width(), self.winfo_height()
        px, py = self.winfo_rootx(), self.winfo_rooty()
        dw, dh = 320, 120
        dlg.geometry(f"{dw}x{dh}+{px + (pw - dw) // 2}+{py + (ph - dh) // 2}")

        tk.Label(dlg, text="Preset name:", bg=C.BG, fg=C.TEXT,
                 font=("Segoe UI", 10)).pack(padx=16, pady=(14, 4), anchor="w")
        name_var = tk.StringVar()
        entry = tk.Entry(dlg, textvariable=name_var, bg=C.SURFACE,
                         fg=C.TEXT, insertbackground=C.TEXT,
                         relief="flat", font=("Segoe UI", 10),
                         highlightthickness=1,
                         highlightbackground=C.BORDER,
                         highlightcolor=C.ACCENT)
        entry.pack(padx=16, fill=tk.X)
        entry.focus_set()

        result = [None]

        def _ok(_event=None):
            val = name_var.get().strip()
            if val:
                result[0] = val
            dlg.destroy()

        def _cancel(_event=None):
            dlg.destroy()

        btn_frame = tk.Frame(dlg, bg=C.BG)
        btn_frame.pack(pady=(10, 10))
        ok_btn = tk.Button(btn_frame, text="Save", width=8,
                           bg=C.ACCENT, fg="#ffffff", relief="flat",
                           font=("Segoe UI", 9, "bold"),
                           activebackground=C.ACCENT_HOVER,
                           activeforeground="#ffffff", cursor="hand2",
                           command=_ok)
        ok_btn.pack(side=tk.LEFT, padx=(0, 6))
        cancel_btn = tk.Button(btn_frame, text="Cancel", width=8,
                               bg=C.SURFACE_LIGHT, fg=C.TEXT, relief="flat",
                               font=("Segoe UI", 9),
                               activebackground=C.SURFACE_HOVER,
                               activeforeground=C.TEXT, cursor="hand2",
                               command=_cancel)
        cancel_btn.pack(side=tk.LEFT)
        entry.bind("<Return>", _ok)
        dlg.bind("<Escape>", _cancel)
        dlg.wait_window()

        name = result[0]
        if not name:
            return
        data = self._get_style_dict()
        os.makedirs(_PRESETS_DIR, exist_ok=True)
        path = os.path.join(_PRESETS_DIR, f"{name}.json")
        try:
            with open(path, "w", encoding="utf-8") as fp:
                json.dump(data, fp, indent=2)
            # Verify file was written
            if not os.path.isfile(path):
                messagebox.showerror("Save Error", f"File was not created:\n{path}")
                return
        except Exception as e:
            messagebox.showerror("Save Error", f"Could not save preset:\n{e}")
            return
        self._refresh_preset_combo()
        self._preset_var.set(name)
        self._status_var.set(f"Preset saved: {name}")

    # ────────────────────────────────────────────────────────────
    # Detail panel
    # ────────────────────────────────────────────────────────────
    def _update_detail_panel(self):
        if self.selected_clip:
            # Show newlines as \r in the entry for editing
            display_text = self.selected_clip.text.replace("\n", "\\r")
            self._detail_text_var.set(display_text)
            self._detail_start_var.set(f"{self.selected_clip.start:.3f}")
            self._detail_end_var.set(f"{self.selected_clip.end:.3f}")
            self._detail_dur_var.set(f"{self.selected_clip.duration:.3f}s")
            # Font size: show clip-specific or global default
            fs = self.selected_clip.font_size if self.selected_clip.font_size else self._font_size
            self._detail_fontsize_var.set(str(fs))
        elif self._selected_track == "audio" and self.selected_media_clip:
            clip = self.selected_media_clip
            self._detail_text_var.set(os.path.basename(clip.source_path))
            self._detail_start_var.set(f"{clip.start:.3f}")
            self._detail_end_var.set(f"{clip.end:.3f}")
            self._detail_dur_var.set(f"{clip.duration:.3f}s")
            self._detail_fontsize_var.set("")
        elif self._selected_track == "video" and self.selected_media_clip:
            clip = self.selected_media_clip
            self._detail_text_var.set(os.path.basename(clip.source_path))
            self._detail_start_var.set(f"{clip.start:.3f}")
            self._detail_end_var.set(f"{clip.end:.3f}")
            self._detail_dur_var.set(f"{clip.duration:.3f}s")
            self._detail_fontsize_var.set("")
        else:
            self._detail_text_var.set("")
            self._detail_start_var.set("")
            self._detail_end_var.set("")
            self._detail_dur_var.set("")
            self._detail_fontsize_var.set("")

    def _show_audio_fx_dialog(self):
        """Open Audio Effects dialog for the selected audio clip."""
        if self._selected_track != "audio" or not self.selected_media_clip:
            messagebox.showinfo("Audio Effects",
                                "Select an audio clip on the timeline first.")
            return
        clip = self.selected_media_clip
        fx = clip.audio_effects

        # Stop any timeline playback to avoid conflicts
        self._stop_playback()

        dlg = tk.Toplevel(self)
        dlg.title(f"Audio Effects \u2014 {os.path.basename(clip.source_path)}")
        dlg.configure(bg=C.BG)
        dlg.resizable(False, False)
        dlg.transient(self)
        dlg.grab_set()

        SML = ("Segoe UI", 9)
        DIM = {"bg": C.BG, "fg": C.TEXT_DIM, "font": SML}
        LBL_W = 10  # label character width for alignment
        BAR_L = 140  # FillBar length

        # Helper: labelled FillBar in a grid cell
        def _fb(parent, label, var, from_, to, row_num, col_offset=0):
            tk.Label(parent, text=label, width=LBL_W, anchor="e",
                     **DIM).grid(row=row_num, column=col_offset,
                                 sticky="e", padx=(0, 4), pady=2)
            fb = FillBar(parent, variable=var, from_=from_, to=to,
                         length=BAR_L, height=16)
            fb.grid(row=row_num, column=col_offset + 1, sticky="w", pady=2)
            val_lbl = tk.Label(parent, textvariable=var, bg=C.BG,
                               fg=C.TEXT, font=("Segoe UI", 8), width=6, anchor="w")
            val_lbl.grid(row=row_num, column=col_offset + 2, sticky="w",
                         padx=(4, 12), pady=2)
            return fb

        # Two-column body
        body = tk.Frame(dlg, bg=C.BG)
        body.pack(fill=tk.BOTH, padx=6, pady=(6, 0))
        col_left = tk.Frame(body, bg=C.BG)
        col_left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(4, 2))
        col_right = tk.Frame(body, bg=C.BG)
        col_right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(2, 4))

        # ── LEFT: Level & Transform ──
        sec1 = tk.LabelFrame(col_left, text="  Level & Transform  ", bg=C.BG,
                             fg=C.TEXT, font=("Segoe UI", 9, "bold"),
                             padx=8, pady=6)
        sec1.pack(fill=tk.X, pady=(0, 4))

        v_vol = tk.DoubleVar(value=fx.get("volume", 1.0))
        v_amp = tk.DoubleVar(value=fx.get("amplify_db", 0.0))
        v_pitch = tk.DoubleVar(value=fx.get("pitch_semitones", 0.0))
        v_speed = tk.DoubleVar(value=fx.get("speed", 1.0))
        v_bass = tk.DoubleVar(value=fx.get("bass_boost", 0.0))
        v_reverse = tk.BooleanVar(value=fx.get("reverse", False))

        g1 = tk.Frame(sec1, bg=C.BG)
        g1.pack(fill=tk.X)
        _fb(g1, "Volume:", v_vol, 0.0, 2.0, 0)
        _fb(g1, "Amplify dB:", v_amp, -20.0, 20.0, 1)
        _fb(g1, "Pitch (st):", v_pitch, -12.0, 12.0, 2)
        _fb(g1, "Speed:", v_speed, 0.25, 4.0, 3)
        _fb(g1, "Bass Boost:", v_bass, 0.0, 20.0, 4)

        rev_row = tk.Frame(sec1, bg=C.BG)
        rev_row.pack(fill=tk.X, pady=(2, 0))
        ttk.Checkbutton(rev_row, text="Reverse",
                        variable=v_reverse).pack(side=tk.LEFT, padx=(4, 0))

        # ── LEFT: Reverb & DeEsser ──
        sec2 = tk.LabelFrame(col_left, text="  Reverb & DeEsser  ", bg=C.BG,
                             fg=C.TEXT, font=("Segoe UI", 9, "bold"),
                             padx=8, pady=6)
        sec2.pack(fill=tk.X, pady=4)

        v_reverb = tk.StringVar(value=fx.get("reverb_preset", "none").title())
        v_reverb_mix = tk.DoubleVar(value=fx.get("reverb_mix", 0.3))
        v_deesser = tk.DoubleVar(value=fx.get("deesser", 0.0))

        rv_row = tk.Frame(sec2, bg=C.BG)
        rv_row.pack(fill=tk.X, pady=2)
        tk.Label(rv_row, text="Reverb:", width=LBL_W, anchor="e",
                 **DIM).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Combobox(rv_row, textvariable=v_reverb,
                     values=["None", "Room", "Vocal Booth",
                             "Studio Live", "Hall", "Chamber",
                             "Theater", "Cathedral", "Arena",
                             "Tunnel", "Parking Garage", "Plate",
                             "Spring", "Shimmer", "Ambient Wash",
                             "Dark Verb", "Bright Verb", "Lo-Fi",
                             "Dreamy", "Ghost"],
                     width=16, state="readonly", font=SML).pack(
            side=tk.LEFT, padx=(0, 8))

        g2 = tk.Frame(sec2, bg=C.BG)
        g2.pack(fill=tk.X)
        _fb(g2, "Reverb Mix:", v_reverb_mix, 0.0, 1.0, 0)
        _fb(g2, "DeEsser:", v_deesser, 0.0, 1.0, 1)

        # ── LEFT: Fade ──
        sec3 = tk.LabelFrame(col_left, text="  Fade  ", bg=C.BG,
                             fg=C.TEXT, font=("Segoe UI", 9, "bold"),
                             padx=8, pady=6)
        sec3.pack(fill=tk.X, pady=4)

        v_fade_in = tk.DoubleVar(value=fx.get("fade_in", 0.0))
        v_fade_out = tk.DoubleVar(value=fx.get("fade_out", 0.0))

        g3 = tk.Frame(sec3, bg=C.BG)
        g3.pack(fill=tk.X)
        _fb(g3, "Fade In (s):", v_fade_in, 0.0, 30.0, 0)
        _fb(g3, "Fade Out (s):", v_fade_out, 0.0, 30.0, 1)

        # ── RIGHT: Autotune & Vocoder ──
        sec4 = tk.LabelFrame(col_right, text="  Autotune & Vocoder  ", bg=C.BG,
                             fg=C.TEXT, font=("Segoe UI", 9, "bold"),
                             padx=8, pady=6)
        sec4.pack(fill=tk.X, pady=(0, 4))

        v_autotune = tk.DoubleVar(value=fx.get("autotune", 0.0))
        v_key = tk.StringVar(value=fx.get("autotune_key", "C"))
        v_melody = tk.StringVar(value=fx.get("autotune_melody", "none").title())
        v_bpm = tk.IntVar(value=fx.get("autotune_bpm", 140))
        v_humanize = tk.DoubleVar(value=fx.get("autotune_humanize", 0.0))
        v_retune_speed = tk.DoubleVar(value=fx.get("autotune_speed", 0.75))
        v_vibrato = tk.DoubleVar(value=fx.get("autotune_vibrato", 1.0))
        v_vocoder = tk.StringVar(value=fx.get("vocoder_preset", "none").title())
        v_vocoder_mix = tk.DoubleVar(value=fx.get("vocoder_mix", 0.5))

        g4 = tk.Frame(sec4, bg=C.BG)
        g4.pack(fill=tk.X)
        _fb(g4, "Autotune:", v_autotune, 0.0, 1.0, 0)

        at_row = tk.Frame(sec4, bg=C.BG)
        at_row.pack(fill=tk.X, pady=2)
        tk.Label(at_row, text="Key:", width=LBL_W, anchor="e",
                 **DIM).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Combobox(at_row, textvariable=v_key,
                     values=["C", "C#", "D", "D#", "E", "F",
                             "F#", "G", "G#", "A", "A#", "B"],
                     width=4, state="readonly", font=SML).pack(
            side=tk.LEFT, padx=(0, 8))
        tk.Label(at_row, text="BPM:", **DIM).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Spinbox(at_row, from_=60, to=200, textvariable=v_bpm,
                    width=4, font=SML).pack(side=tk.LEFT, padx=(0, 8))

        at_row2 = tk.Frame(sec4, bg=C.BG)
        at_row2.pack(fill=tk.X, pady=2)
        tk.Label(at_row2, text="Melody:", width=LBL_W, anchor="e",
                 **DIM).pack(side=tk.LEFT, padx=(0, 4))
        melody_values = [m.title() for m in MELODY_NAMES]
        ttk.Combobox(at_row2, textvariable=v_melody,
                     values=melody_values,
                     width=16, state="readonly", font=SML).pack(
            side=tk.LEFT)
        
        # Autotune fine controls
        at_fine = tk.Frame(sec4, bg=C.BG)
        at_fine.pack(fill=tk.X)
        _fb(at_fine, "Humanize:", v_humanize, 0.0, 1.0, 0)
        
        at_fine2 = tk.Frame(sec4, bg=C.BG)
        at_fine2.pack(fill=tk.X)
        _fb(at_fine2, "Retune Spd:", v_retune_speed, 0.0, 1.0, 0)
        
        at_fine3 = tk.Frame(sec4, bg=C.BG)
        at_fine3.pack(fill=tk.X)
        _fb(at_fine3, "Vibrato:", v_vibrato, 0.0, 1.0, 0)
        
        voc_row = tk.Frame(sec4, bg=C.BG)
        voc_row.pack(fill=tk.X, pady=2)
        tk.Label(voc_row, text="Vocoder:", width=LBL_W, anchor="e",
                 **DIM).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Combobox(voc_row, textvariable=v_vocoder,
                     values=["None",
                             "Alien", "Cyberpunk Ai Oracle", "Broken Android",
                             "Alien Hive Mind", "Demon Overlord",
                             "Whispering Shadow", "Hollow Skull",
                             "Clockwork Automaton", "Data Stream",
                             "Cartoon Chip Hero", "Giant Titan",
                             "Old Radio Announcer", "Masked Vigilante",
                             "Robot", "Whisper", "Monster", "Chipmunk", "Deep",
                             "Woman", "Heavy Man", "Smoker Male", "Smoker Female"],
                     width=22, state="readonly", font=SML).pack(
            side=tk.LEFT)

        g4b = tk.Frame(sec4, bg=C.BG)
        g4b.pack(fill=tk.X)
        _fb(g4b, "Vocoder Mix:", v_vocoder_mix, 0.0, 1.0, 0)

        # ── RIGHT: Delay & Echo ──
        sec5 = tk.LabelFrame(col_right, text="  Delay & Echo  ", bg=C.BG,
                             fg=C.TEXT, font=("Segoe UI", 9, "bold"),
                             padx=8, pady=6)
        sec5.pack(fill=tk.X, pady=4)

        v_delay_time = tk.DoubleVar(value=fx.get("delay_time", 0.0))
        v_delay_fb = tk.DoubleVar(value=fx.get("delay_feedback", 0.3))
        v_delay_mix = tk.DoubleVar(value=fx.get("delay_mix", 0.5))
        v_echo_time = tk.DoubleVar(value=fx.get("echo_time", 0.0))
        v_echo_decay = tk.DoubleVar(value=fx.get("echo_decay", 0.5))
        v_echo_count = tk.IntVar(value=fx.get("echo_count", 3))
        v_echo_mix = tk.DoubleVar(value=fx.get("echo_mix", 0.5))

        g5 = tk.Frame(sec5, bg=C.BG)
        g5.pack(fill=tk.X)
        _fb(g5, "Delay (s):", v_delay_time, 0.0, 5.0, 0)
        _fb(g5, "Feedback:", v_delay_fb, 0.0, 0.9, 1)
        _fb(g5, "Delay Mix:", v_delay_mix, 0.0, 1.0, 2)
        _fb(g5, "Echo (s):", v_echo_time, 0.0, 5.0, 3)
        _fb(g5, "Echo Decay:", v_echo_decay, 0.0, 0.9, 4)
        _fb(g5, "Echo Mix:", v_echo_mix, 0.0, 1.0, 5)

        echo_ct_row = tk.Frame(sec5, bg=C.BG)
        echo_ct_row.pack(fill=tk.X, pady=2)
        tk.Label(echo_ct_row, text="Repeats:", width=LBL_W, anchor="e",
                 **DIM).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Spinbox(echo_ct_row, from_=1, to=10, increment=1,
                    textvariable=v_echo_count, width=4, font=SML).pack(
            side=tk.LEFT)

        # ── Buttons ──
        btn_row = tk.Frame(dlg, bg=C.BG)
        btn_row.pack(fill=tk.X, padx=10, pady=(8, 10))

        # Collect current UI values into an effects dict
        def _gather_fx():
            return {
                "volume": v_vol.get(),
                "amplify_db": v_amp.get(),
                "pitch_semitones": v_pitch.get(),
                "speed": v_speed.get(),
                "bass_boost": v_bass.get(),
                "reverse": v_reverse.get(),
                "reverb_preset": v_reverb.get().lower(),
                "reverb_mix": v_reverb_mix.get(),
                "deesser": v_deesser.get(),
                "fade_in": v_fade_in.get(),
                "fade_out": v_fade_out.get(),
                "autotune": v_autotune.get(),
                "autotune_key": v_key.get(),
                "autotune_melody": v_melody.get().lower(),
                "autotune_bpm": v_bpm.get(),
                "autotune_humanize": v_humanize.get(),
                "autotune_speed": v_retune_speed.get(),
                "autotune_vibrato": v_vibrato.get(),
                "vocoder_preset": v_vocoder.get().lower(),
                "vocoder_mix": v_vocoder_mix.get(),
                "delay_time": v_delay_time.get(),
                "delay_feedback": v_delay_fb.get(),
                "delay_mix": v_delay_mix.get(),
                "echo_time": v_echo_time.get(),
                "echo_decay": v_echo_decay.get(),
                "echo_count": v_echo_count.get(),
                "echo_mix": v_echo_mix.get(),
            }

        _preview_tmp = [None]  # mutable holder for temp file path
        _preview_btns = []     # buttons to disable during processing

        def _preview():
            """Process audio with current FX settings and play."""
            try:
                import numpy as np
                import time as _time
                import tempfile
                import audio_fx as _afx

                # Stop any current preview / timeline playback
                _stop_preview()
                self._audio_stop()

                # ── Critical check: scipy must be available ──
                if not _afx._HAS_SCIPY:
                    messagebox.showerror(
                        "Audio FX",
                        "scipy is NOT loaded — effects disabled!\n"
                        "Install: pip install scipy")
                    return

                # ── Check CREPE for autotune ──
                cur_fx = _gather_fx()
                if cur_fx.get("autotune", 0) > 0 and not _afx._HAS_CREPE:
                    messagebox.showwarning(
                        "Audio FX",
                        "Autotune requires the 'torchcrepe' package.\n"
                        "Install: pip install torchcrepe\n\n"
                        "Autotune will be skipped.")

                non_default = {k: v for k, v in cur_fx.items()
                               if v != _DEFAULT_AUDIO_FX.get(k)}
                print(f"[AudioFX Preview] Non-default: {non_default}")

                preview_dur = min(15.0, clip.duration)
                print(f"[AudioFX Preview] Processing "
                      f"{clip.source_path}, "
                      f"offset={clip.source_offset:.2f}, "
                      f"dur={preview_dur:.2f}s")

                # Disable buttons during processing to prevent race
                for btn in _preview_btns:
                    try:
                        btn.config(state="disabled")
                    except Exception:
                        pass
                self._status_var.set("Processing effects\u2026")
                dlg.update_idletasks()

                _t0 = _time.time()
                samples, sr, nch = process_audio_clip(
                    clip.source_path, cur_fx,
                    source_offset=clip.source_offset,
                    source_duration=preview_dur)
                _elapsed = _time.time() - _t0
                rms = float(np.sqrt(np.mean(samples ** 2)))
                print(f"[AudioFX Preview] Done: {_elapsed:.2f}s, "
                      f"rms={rms:.4f}")

                # Write to unique temp file
                tmp = tempfile.NamedTemporaryFile(
                    suffix=".wav", delete=False,
                    prefix="subeditor_fx_",
                    dir=tempfile.gettempdir())
                tmp.close()
                preview_path = tmp.name
                write_wav(preview_path, samples, sr, nch)
                print(f"[AudioFX Preview] Written: {preview_path}")

                # Clean up previous temp file
                old = _preview_tmp[0]
                if old and old != preview_path:
                    try:
                        _stop_preview()
                        os.remove(old)
                    except OSError:
                        pass
                _preview_tmp[0] = preview_path

                # Drain queued events that accumulated during processing
                # (prevents queued Stop/Preview clicks from killing audio)
                dlg.update()

                # Re-enable buttons
                for btn in _preview_btns:
                    try:
                        btn.config(state="normal")
                    except Exception:
                        pass

                # Play via sounddevice
                if _HAS_AUDIO:
                    import numpy as np
                    _stop_preview()  # Stop any previous
                    audio_data = samples.astype(np.float32)
                    sd.play(audio_data, sr)
                    print("[AudioFX Preview] Playing via sounddevice")
                self._status_var.set("Preview playing\u2026")
            except Exception as exc:
                import traceback
                traceback.print_exc()
                # Re-enable buttons on error
                for btn in _preview_btns:
                    try:
                        btn.config(state="normal")
                    except Exception:
                        pass
                messagebox.showerror("Preview Error", str(exc))

        def _stop_preview():
            if _HAS_AUDIO:
                try:
                    sd.stop()
                except Exception:
                    pass
            self._status_var.set("Preview stopped")

        def _apply_and_close():
            _stop_preview()
            try:
                # Invalidate old cache entry before updating effects
                self._invalidate_audio_cache(clip)
                cur_fx = _gather_fx()
                for k, v in cur_fx.items():
                    clip.audio_effects[k] = v
                # Rebuild cache with new effects
                self._build_audio_cache()
                self._status_var.set("Audio effects applied")
            except Exception as exc:
                print(f"[AudioFX] Error saving effects: {exc}")
                messagebox.showerror("Audio FX",
                                     f"Failed to save effects:\n{exc}")
            dlg.destroy()

        def _reset_fx():
            v_vol.set(1.0); v_amp.set(0.0); v_pitch.set(0.0)
            v_speed.set(1.0); v_bass.set(0.0); v_reverse.set(False)
            v_reverb.set("None"); v_reverb_mix.set(0.3)
            v_deesser.set(0.0); v_fade_in.set(0.0); v_fade_out.set(0.0)
            v_autotune.set(0.0); v_key.set("C"); v_humanize.set(0.0)
            v_retune_speed.set(0.0); v_vibrato.set(0.5)
            v_vocoder.set("None"); v_vocoder_mix.set(0.5)
            v_delay_time.set(0.0); v_delay_fb.set(0.3); v_delay_mix.set(0.5)
            v_echo_time.set(0.0); v_echo_decay.set(0.5)
            v_echo_count.set(3); v_echo_mix.set(0.5)

        def _on_close():
            _stop_preview()
            dlg.destroy()

        dlg.protocol("WM_DELETE_WINDOW", _on_close)

        ttk.Button(btn_row, text="Reset",
                   command=_reset_fx).pack(side=tk.LEFT)
        _btn_preview = ttk.Button(btn_row, text="\u25B6 Preview",
                   command=_preview)
        _btn_preview.pack(side=tk.LEFT, padx=(8, 0))
        _btn_stop = ttk.Button(btn_row, text="\u25A0 Stop",
                   command=_stop_preview)
        _btn_stop.pack(side=tk.LEFT, padx=(4, 0))
        ttk.Button(btn_row, text="Cancel",
                   command=_on_close).pack(side=tk.RIGHT, padx=(6, 0))
        _btn_apply = ttk.Button(btn_row, text="Apply",
                   command=_apply_and_close)
        _btn_apply.pack(side=tk.RIGHT)
        _preview_btns.extend([_btn_preview, _btn_stop, _btn_apply])

        # Centre dialog on parent
        dlg.update_idletasks()
        dw, dh = dlg.winfo_width(), dlg.winfo_height()
        px, py = self.winfo_x(), self.winfo_y()
        pw, ph = self.winfo_width(), self.winfo_height()
        dlg.geometry(f"+{px + (pw - dw) // 2}+{py + (ph - dh) // 2}")

    def _apply_detail(self):
        if self.selected_clip:
            text = self._detail_text_var.get().strip()
            # Convert \r to actual newlines for multi-line subtitles
            text = text.replace("\\r", "\n")
            try:
                start = float(self._detail_start_var.get())
                end = float(self._detail_end_var.get())
            except ValueError:
                messagebox.showwarning("Invalid", "Start/End must be numbers.")
                return
            if end <= start:
                messagebox.showwarning("Invalid", "End must be > Start.")
                return
            # Parse font size (optional)
            fs_str = self._detail_fontsize_var.get().strip()
            if fs_str:
                try:
                    font_size = int(fs_str)
                    if font_size < 8 or font_size > 500:
                        messagebox.showwarning("Invalid", "Font size must be 8-500.")
                        return
                    self.selected_clip.font_size = font_size
                except ValueError:
                    messagebox.showwarning("Invalid", "Font size must be an integer.")
                    return
            self.selected_clip.text = text
            self.selected_clip.start = start
            self.selected_clip.end = end
            self._redraw_timeline()
            self._refresh_preview()
        elif self._selected_track in ("audio", "video") and self.selected_media_clip:
            try:
                start = float(self._detail_start_var.get())
                end = float(self._detail_end_var.get())
            except ValueError:
                messagebox.showwarning("Invalid", "Start/End must be numbers.")
                return
            if end <= start:
                messagebox.showwarning("Invalid", "End must be > Start.")
                return
            self.selected_media_clip.start = start
            self.selected_media_clip.end = end
            self._redraw_timeline()
            self._refresh_preview()

    def _copy_clip(self):
        """Copy the selected clip to the internal clipboard."""
        if self._selected_track == "subtitle" and self.selected_clip:
            self._clipboard_clip = {
                "type": "subtitle",
                "text": self.selected_clip.text,
                "duration": self.selected_clip.duration,
            }
            self._status_var.set(
                f"Copied: \"{self.selected_clip.text[:30]}\"")
        elif self._selected_track in ("video", "audio") and self.selected_media_clip:
            clip = self.selected_media_clip
            self._clipboard_clip = {
                "type": self._selected_track,
                "source_path": clip.source_path,
                "source_type": clip.source_type,
                "duration": clip.duration,
                "source_offset": clip.source_offset,
                "source_duration": clip.source_duration,
                "track_index": clip.track_index,
                "audio_effects": dict(clip.audio_effects),
            }
            name = os.path.basename(clip.source_path)
            self._status_var.set(f"Copied {self._selected_track} clip: {name}")

    def _paste_clip(self):
        """Paste the clipboard clip at the current playback position."""
        if not self._clipboard_clip:
            self._status_var.set("Nothing to paste")
            return

        t = self._playback_time
        clip_type = self._clipboard_clip.get("type", "subtitle")

        if clip_type == "subtitle":
            dur = self._clipboard_clip["duration"]
            new_clip = SubtitleClip(
                text=self._clipboard_clip.get("text", ""),
                start=t, end=t + dur,
                idx=len(self.subtitles))
            self.subtitles.append(new_clip)
            self.selected_clip = new_clip
            self._selected_track = "subtitle"
            self._update_detail_panel()
            self._status_var.set(f"Pasted subtitle at {t:.2f}s")
        elif clip_type == "video":
            dur = self._clipboard_clip["duration"]
            new_clip = MediaClip(
                source_path=self._clipboard_clip["source_path"],
                source_type=self._clipboard_clip["source_type"],
                start=t, end=t + dur,
                source_offset=self._clipboard_clip["source_offset"],
                source_duration=self._clipboard_clip["source_duration"],
                track_index=self._clipboard_clip.get("track_index", 0),
            )
            self.video_clips.append(new_clip)
            self.selected_media_clip = new_clip
            self._selected_track = "video"
            self._status_var.set(f"Pasted video clip at {t:.2f}s")
        elif clip_type == "audio":
            dur = self._clipboard_clip["duration"]
            new_clip = MediaClip(
                source_path=self._clipboard_clip["source_path"],
                source_type=self._clipboard_clip["source_type"],
                start=t, end=t + dur,
                source_offset=self._clipboard_clip["source_offset"],
                source_duration=self._clipboard_clip["source_duration"],
                track_index=self._clipboard_clip.get("track_index", 0),
                audio_effects=self._clipboard_clip.get("audio_effects"),
            )
            self.audio_clips.append(new_clip)
            self.selected_media_clip = new_clip
            self._selected_track = "audio"
            self._status_var.set(f"Pasted audio clip at {t:.2f}s")

        self._redraw_timeline()

    def _duplicate_clip(self):
        """Duplicate the selected clip right after its end."""
        if self._selected_track == "subtitle" and self.selected_clip:
            src = self.selected_clip
            dur = src.duration
            new_clip = SubtitleClip(
                text=src.text, start=src.end, end=src.end + dur,
                idx=len(self.subtitles))
            self.subtitles.append(new_clip)
            self.selected_clip = new_clip
            self._update_detail_panel()
            self._status_var.set(f"Duplicated subtitle at {src.end:.2f}s")
        elif self._selected_track == "video" and self.selected_media_clip:
            src = self.selected_media_clip
            new_clip = MediaClip(
                source_path=src.source_path,
                source_type=src.source_type,
                start=src.end, end=src.end + src.duration,
                source_offset=src.source_offset,
                source_duration=src.source_duration,
                track_index=getattr(src, 'track_index', 0),
            )
            self.video_clips.append(new_clip)
            self.selected_media_clip = new_clip
            self._status_var.set(f"Duplicated video clip at {src.end:.2f}s")
        elif self._selected_track == "audio" and self.selected_media_clip:
            src = self.selected_media_clip
            new_clip = MediaClip(
                source_path=src.source_path,
                source_type=src.source_type,
                start=src.end, end=src.end + src.duration,
                source_offset=src.source_offset,
                source_duration=src.source_duration,
                track_index=src.track_index,
                audio_effects=dict(src.audio_effects),
            )
            self.audio_clips.append(new_clip)
            self.selected_media_clip = new_clip
            self._status_var.set(f"Duplicated audio clip at {src.end:.2f}s")
        else:
            return
        self._redraw_timeline()

    def _insert_clip(self):
        """Insert a new subtitle clip at the current playback position."""
        t = getattr(self, '_playback_time', 0.0)
        dur = 2.0
        new_clip = SubtitleClip(
            text="New subtitle", start=t, end=t + dur,
            idx=len(self.subtitles))
        self.subtitles.append(new_clip)
        self.selected_clip = new_clip
        self._update_detail_panel()
        self._redraw_timeline()
        self._status_var.set(
            f"Inserted clip at {t:.2f}s")

    def _delete_selected(self):
        """Delete the selected clip(s) (subtitle, video, or audio)."""
        # Handle multi-selection deletion
        if self._multi_selected:
            count = len(self._multi_selected)
            if messagebox.askyesno("Delete", f"Delete {count} selected clip(s)?"):
                for clip in list(self._multi_selected):
                    if clip in self.video_clips:
                        self.video_clips.remove(clip)
                    elif clip in self.audio_clips:
                        self.audio_clips.remove(clip)
                    elif clip in self.subtitles:
                        self.subtitles.remove(clip)
                self._multi_selected.clear()
                self.selected_clip = None
                self.selected_media_clip = None
                self._selected_track = None
                self._status_var.set(f"Deleted {count} clip(s)")
                self._refresh_preview()
                self._redraw_timeline()
            return
        
        if self._selected_track == "video" and self.selected_media_clip:
            if messagebox.askyesno("Delete", "Delete this video/image clip?"):
                if self.selected_media_clip in self.video_clips:
                    self.video_clips.remove(self.selected_media_clip)
                self.selected_media_clip = None
                self._selected_track = None
                self._status_var.set("Video/image clip deleted")
                self._refresh_preview()
                self._redraw_timeline()
        elif self._selected_track == "audio" and self.selected_media_clip:
            if messagebox.askyesno("Delete", "Delete this audio clip?"):
                if self.selected_media_clip in self.audio_clips:
                    self.audio_clips.remove(self.selected_media_clip)
                self.selected_media_clip = None
                self._selected_track = None
                self._status_var.set("Audio clip deleted")
                self._refresh_preview()
                self._redraw_timeline()
        elif self.selected_clip and self.selected_clip in self.subtitles:
            self.subtitles.remove(self.selected_clip)
            self.selected_clip = None
            self._selected_track = None
            self._update_detail_panel()
            self._redraw_timeline()
            self._status_var.set("Subtitle clip deleted")

    def _clear_selection(self):
        """Clear all clip selections."""
        self._multi_selected.clear()
        self.selected_clip = None
        self.selected_media_clip = None
        self._selected_track = None
        self._update_detail_panel()
        self._redraw_timeline()

    def _split_selected(self):
        """Split the selected subtitle clip at the current playhead position."""
        if self._selected_track != "subtitle" or not self.selected_clip:
            self._status_var.set("Select a subtitle clip to split")
            return
        clip = self.selected_clip
        t = self._playback_time
        # Check if playhead is within the clip (with small margin)
        if t <= clip.start + 0.1 or t >= clip.end - 0.1:
            self._status_var.set("Playhead must be inside the clip to split")
            return
        # Save original end before modifying
        orig_end = clip.end
        # Create two clips from the original
        text = clip.text
        # Try to split text roughly proportionally
        words = text.split()
        ratio = (t - clip.start) / (orig_end - clip.start)
        split_idx = max(1, int(len(words) * ratio))
        text1 = " ".join(words[:split_idx]) if split_idx < len(words) else text
        text2 = " ".join(words[split_idx:]) if split_idx < len(words) else text
        if not text1.strip():
            text1 = text
        if not text2.strip():
            text2 = text
        # Modify existing clip to be the first part
        clip.end = t
        clip.text = text1
        # Create new clip for the second part
        new_clip = SubtitleClip(
            text=text2, start=t, end=orig_end,
            idx=len(self.subtitles))
        self.subtitles.append(new_clip)
        # Select the new clip
        self.selected_clip = new_clip
        self._update_detail_panel()
        self._redraw_timeline()
        self._status_var.set(f"Split clip at {t:.2f}s")

    def _on_tl_right_click(self, event):
        """Handle right-click on timeline - show context menu."""
        x = self._tl_canvas.canvasx(event.x)
        y = self._tl_canvas.canvasy(event.y)

        # Don't show menu on ruler
        if y < RULER_HEIGHT:
            return

        clip, mode, track = self._find_clip_at(x, y)
        if not clip or track not in ("video", "audio", "subtitle"):
            return

        # Store the click position and clip for operations
        self._right_click_sec = self._x_to_sec(x)
        self._right_click_clip = clip
        self._right_click_track = track

        # Create dark-themed context menu
        menu = tk.Menu(self, tearoff=0,
                       bg=C.SURFACE, fg=C.TEXT,
                       activebackground=C.ACCENT, activeforeground="white",
                       relief="flat", bd=1,
                       selectcolor=C.ACCENT)
        menu.add_command(label="  Cut", accelerator="Ctrl+X",
                         command=self._cut_right_click_clip)
        menu.add_command(label="  Copy", accelerator="Ctrl+C",
                         command=self._copy_right_click_clip)
        menu.add_command(label="  Paste", accelerator="Ctrl+V",
                         command=self._paste_at_right_click)
        menu.add_separator()
        menu.add_command(label="  Split at this point",
                         command=self._split_at_right_click)
        menu.add_command(label="  Duplicate",
                         command=self._duplicate_right_click_clip)
        menu.add_separator()
        # Audio effects submenu for audio clips
        if track == "audio":
            fx_menu = tk.Menu(menu, tearoff=0,
                              bg=C.SURFACE, fg=C.TEXT,
                              activebackground=C.ACCENT, activeforeground="white")
            fx_menu.add_command(label="  Audio Effects\u2026",
                                command=lambda: self._show_audio_fx_for_clip(clip))
            fx_menu.add_command(label="  Reset Effects",
                                command=lambda: self._reset_audio_fx_for_clip(clip))
            menu.add_cascade(label="  Audio Effects", menu=fx_menu)
            menu.add_separator()
        menu.add_command(label="  Delete", accelerator="Del",
                         command=self._delete_right_click_clip)
        menu.tk_popup(event.x_root, event.y_root)

    def _copy_right_click_clip(self):
        """Copy the right-clicked clip to clipboard."""
        clip = getattr(self, '_right_click_clip', None)
        track = getattr(self, '_right_click_track', None)
        if not clip or not track:
            return

        if track == "subtitle":
            self._clipboard_clip = {
                "type": "subtitle",
                "text": clip.text,
                "duration": clip.duration,
            }
            self._status_var.set(f"Copied subtitle: \"{clip.text[:30]}\"")
        else:
            self._clipboard_clip = {
                "type": track,  # "video" or "audio"
                "source_path": clip.source_path,
                "source_type": clip.source_type,
                "duration": clip.duration,
                "source_offset": clip.source_offset,
                "source_duration": clip.source_duration,
                "track_index": clip.track_index,
                "audio_effects": dict(clip.audio_effects),
            }
            name = os.path.basename(clip.source_path)
            self._status_var.set(f"Copied {track} clip: {name}")

    def _cut_right_click_clip(self):
        """Cut the right-clicked clip (copy + delete)."""
        self._copy_right_click_clip()
        self._delete_right_click_clip()

    def _paste_at_right_click(self):
        """Paste clipboard clip at the right-click position."""
        if not self._clipboard_clip:
            self._status_var.set("Nothing to paste")
            return

        t = getattr(self, '_right_click_sec', self._playback_time)
        clip_type = self._clipboard_clip.get("type", "subtitle")

        if clip_type == "subtitle":
            dur = self._clipboard_clip["duration"]
            new_clip = SubtitleClip(
                text=self._clipboard_clip["text"],
                start=t, end=t + dur,
                idx=len(self.subtitles))
            self.subtitles.append(new_clip)
            self.selected_clip = new_clip
            self._selected_track = "subtitle"
            self._update_detail_panel()
            self._status_var.set(f"Pasted subtitle at {t:.2f}s")
        elif clip_type == "video":
            dur = self._clipboard_clip["duration"]
            new_clip = MediaClip(
                source_path=self._clipboard_clip["source_path"],
                source_type=self._clipboard_clip["source_type"],
                start=t, end=t + dur,
                source_offset=self._clipboard_clip["source_offset"],
                source_duration=self._clipboard_clip["source_duration"],
                track_index=self._clipboard_clip.get("track_index", 0),
            )
            self.video_clips.append(new_clip)
            self.selected_media_clip = new_clip
            self._selected_track = "video"
            self._status_var.set(f"Pasted video clip at {t:.2f}s")
        elif clip_type == "audio":
            dur = self._clipboard_clip["duration"]
            new_clip = MediaClip(
                source_path=self._clipboard_clip["source_path"],
                source_type=self._clipboard_clip["source_type"],
                start=t, end=t + dur,
                source_offset=self._clipboard_clip["source_offset"],
                source_duration=self._clipboard_clip["source_duration"],
                track_index=self._clipboard_clip.get("track_index", 0),
                audio_effects=self._clipboard_clip.get("audio_effects"),
            )
            self.audio_clips.append(new_clip)
            self.selected_media_clip = new_clip
            self._selected_track = "audio"
            self._status_var.set(f"Pasted audio clip at {t:.2f}s")

        self._redraw_timeline()

    def _duplicate_right_click_clip(self):
        """Duplicate the right-clicked clip right after its end."""
        clip = getattr(self, '_right_click_clip', None)
        track = getattr(self, '_right_click_track', None)
        if not clip or not track:
            return

        if track == "subtitle":
            dur = clip.duration
            new_clip = SubtitleClip(
                text=clip.text, start=clip.end, end=clip.end + dur,
                idx=len(self.subtitles))
            self.subtitles.append(new_clip)
            self.selected_clip = new_clip
            self._selected_track = "subtitle"
            self._update_detail_panel()
        elif track == "video":
            new_clip = MediaClip(
                source_path=clip.source_path,
                source_type=clip.source_type,
                start=clip.end, end=clip.end + clip.duration,
                source_offset=clip.source_offset,
                source_duration=clip.source_duration,
                track_index=getattr(clip, 'track_index', 0),
            )
            self.video_clips.append(new_clip)
            self.selected_media_clip = new_clip
            self._selected_track = "video"
        elif track == "audio":
            new_clip = MediaClip(
                source_path=clip.source_path,
                source_type=clip.source_type,
                start=clip.end, end=clip.end + clip.duration,
                source_offset=clip.source_offset,
                source_duration=clip.source_duration,
                track_index=clip.track_index,
                audio_effects=dict(clip.audio_effects),
            )
            self.audio_clips.append(new_clip)
            self.selected_media_clip = new_clip
            self._selected_track = "audio"

        self._redraw_timeline()
        self._status_var.set(f"Duplicated {track} clip")

    def _split_at_right_click(self):
        """Split the clip at the right-click position."""
        clip = getattr(self, '_right_click_clip', None)
        track = getattr(self, '_right_click_track', None)
        t = getattr(self, '_right_click_sec', None)

        if not clip or not track or t is None:
            return

        # Check if click position is within the clip (with small margin)
        if t <= clip.start + 0.1 or t >= clip.end - 0.1:
            self._status_var.set("Cannot split at edge of clip")
            return

        if track == "subtitle":
            # Use existing subtitle split logic
            self.selected_clip = clip
            self._selected_track = "subtitle"
            old_time = self._playback_time
            self._playback_time = t
            self._split_selected()
            self._playback_time = old_time
        elif track == "video":
            self._split_media_clip(clip, t, self.video_clips)
        elif track == "audio":
            self._split_media_clip(clip, t, self.audio_clips)

    def _split_media_clip(self, clip, split_time, clip_list):
        """Split a media clip (video/audio) at the given time."""
        if clip not in clip_list:
            return

        # Calculate source offset for the second clip
        time_into_clip = split_time - clip.start
        new_source_offset = clip.source_offset + time_into_clip

        # Save original values
        orig_end = clip.end

        # Modify the original clip to end at split point
        clip.end = split_time

        # Create new clip for the second part
        new_clip = MediaClip(
            source_path=clip.source_path,
            source_type=clip.source_type,
            start=split_time,
            end=orig_end,
            source_offset=new_source_offset,
            source_duration=clip.source_duration,
            track_index=clip.track_index,
            audio_effects=dict(clip.audio_effects),
        )
        clip_list.append(new_clip)

        # Select the new clip
        self.selected_media_clip = new_clip
        self._redraw_timeline()
        self._status_var.set(f"Split {clip.source_type} clip at {split_time:.2f}s")

    def _delete_right_click_clip(self):
        """Delete the clip that was right-clicked."""
        clip = getattr(self, '_right_click_clip', None)
        track = getattr(self, '_right_click_track', None)

        if not clip or not track:
            return

        if track == "subtitle":
            if clip in self.subtitles:
                self.subtitles.remove(clip)
                if self.selected_clip == clip:
                    self.selected_clip = None
                self._status_var.set("Subtitle clip deleted")
        elif track == "video":
            if clip in self.video_clips:
                self.video_clips.remove(clip)
                if self.selected_media_clip == clip:
                    self.selected_media_clip = None
                self._status_var.set("Video clip deleted")
        elif track == "audio":
            if clip in self.audio_clips:
                self.audio_clips.remove(clip)
                if self.selected_media_clip == clip:
                    self.selected_media_clip = None
                self._status_var.set("Audio clip deleted")

        self._update_detail_panel()
        self._redraw_timeline()

    def _show_audio_fx_for_clip(self, clip):
        """Open Audio Effects dialog for a specific audio clip."""
        if clip not in self.audio_clips:
            return
        # Temporarily select the clip and show the dialog
        self._selected_track = "audio"
        self.selected_media_clip = clip
        self._show_audio_fx_dialog()

    def _reset_audio_fx_for_clip(self, clip):
        """Reset audio effects for a specific clip."""
        if clip not in self.audio_clips:
            return
        if messagebox.askyesno("Reset Effects",
                              f"Reset all audio effects for {os.path.basename(clip.source_path)}?"):
            clip.audio_effects = dict(_DEFAULT_AUDIO_FX)
            self._invalidate_audio_cache(clip)
            self._redraw_timeline()
            self._status_var.set("Audio effects reset")

    def _clear_video_track(self):
        if self.video_clips and messagebox.askyesno(
                "Clear", "Remove all video track clips?"):
            self.video_clips.clear()
            if isinstance(self.selected_clip, MediaClip) and \
               self.selected_clip not in self.audio_clips:
                self.selected_clip = None
                self._selected_track = None
            self._update_detail_panel()
            self._redraw_timeline()

    def _clear_audio_track(self):
        if self.audio_clips and messagebox.askyesno(
                "Clear", "Remove all audio track clips?"):
            self.audio_clips.clear()
            if isinstance(self.selected_clip, MediaClip) and \
               self.selected_clip not in self.video_clips:
                self.selected_clip = None
                self._selected_track = None
            self._audio_track_count = 1
            self._update_timeline_height()
            self._update_detail_panel()
            self._redraw_timeline()

    def _add_video_track(self):
        """Add an additional video track layer."""
        self._video_track_count += 1
        self._update_timeline_height()
        self._redraw_timeline()
        self._status_var.set(f"Added video track {self._video_track_count}")

    def _remove_video_track(self):
        """Remove the last video track layer if empty."""
        if self._video_track_count <= 1:
            messagebox.showinfo("Remove Track", "Cannot remove the last video track.")
            return
        last_idx = self._video_track_count - 1
        clips_on_last = [c for c in self.video_clips if c.track_index == last_idx]
        if clips_on_last:
            if not messagebox.askyesno("Remove Track",
                    f"Video track {last_idx + 1} has {len(clips_on_last)} clip(s).\n"
                    "Remove track and delete these clips?"):
                return
            for c in clips_on_last:
                self.video_clips.remove(c)
        self._video_track_count -= 1
        self._update_timeline_height()
        self._redraw_timeline()
        self._status_var.set(f"Removed video track {last_idx + 1}")

    # ────────────────────────────────────────────────────────────
    # Track preview cache generation
    # ────────────────────────────────────────────────────────────
    def _generate_video_thumbnails(self):
        """Extract evenly-spaced frames from video clip as small PhotoImages."""
        self._video_thumbnails = []
        if self._video_clip is None:
            return
        from PIL import Image, ImageTk
        try:
            dur = self._video_clip.duration
            if dur <= 0:
                return
            thumb_h = TRACK_HEIGHT - 6
            thumb_w = max(1, int(thumb_h * 16 / 9))
            # One thumbnail per thumb_w-sized slot, up to ~120 thumbnails
            n_thumbs = min(120, max(4, int(dur * self._px_per_sec / thumb_w)))
            for i in range(n_thumbs):
                t = dur * i / n_thumbs
                try:
                    frame = self._video_clip.get_frame(t)
                    img = Image.fromarray(frame).resize(
                        (thumb_w, thumb_h), Image.LANCZOS)
                    photo = ImageTk.PhotoImage(img)
                    self._video_thumbnails.append((t, photo))
                except Exception:
                    pass
        except Exception:
            pass

    def _generate_image_thumb_strip(self):
        """Create a single thumbnail tile from background image."""
        self._video_thumb_strip = None
        if self._bg_image_pil is None:
            return
        from PIL import ImageTk
        try:
            thumb_h = TRACK_HEIGHT - 6
            thumb_w = max(1, int(thumb_h * 16 / 9))
            img = self._bg_image_pil.copy().resize(
                (thumb_w, thumb_h))
            self._video_thumb_strip = ImageTk.PhotoImage(img)
        except Exception:
            pass

    def _generate_audio_waveform(self):
        """Generate waveform data for every audio clip in a background thread."""
        # Collect paths that need waveform generation
        paths_to_load = []
        for aclip in self.audio_clips:
            path = aclip.source_path
            if path and path not in self._audio_waveforms and path not in paths_to_load:
                paths_to_load.append(path)
        
        if not paths_to_load:
            return
        
        def _load_waveforms():
            results = {}
            for path in paths_to_load:
                results[path] = self._read_waveform_for_file(path)
            # Schedule UI update on main thread
            self.after(0, lambda: self._on_waveforms_loaded(results))
        
        threading.Thread(target=_load_waveforms, daemon=True).start()
    
    def _on_waveforms_loaded(self, results):
        """Update waveform cache and redraw timeline (called on main thread)."""
        self._audio_waveforms.update(results)
        self._redraw_timeline()

    @staticmethod
    def _read_waveform_for_file(path):
        """Read a WAV file and return a list of (fraction, amplitude) tuples.

        Uses RMS over small windows for a smooth, accurate envelope.
        Returns up to ~1000 sample points normalised to 0-1.
        """
        if not path or not os.path.isfile(path):
            return []
        try:
            import struct as _struct
            wf = wave.open(path, "rb")
            n_channels = wf.getnchannels()
            sampwidth  = wf.getsampwidth()
            n_frames   = wf.getnframes()
            framerate  = wf.getframerate()
            if n_frames == 0:
                wf.close()
                return []

            # Read ALL frames at once (much faster than seeking per sample)
            raw_bytes = wf.readframes(n_frames)
            wf.close()

            fmt_char = {1: 'b', 2: 'h', 4: 'i'}.get(sampwidth, 'h')
            max_val  = max(1, (2 ** (sampwidth * 8 - 1)) - 1)
            total_samples = n_frames * n_channels
            try:
                all_vals = _struct.unpack(f'<{total_samples}{fmt_char}',
                                          raw_bytes)
            except _struct.error:
                return []

            import numpy as _np
            arr = _np.array(all_vals, dtype=_np.float64) / max_val

            # If stereo+, take max of absolute across channels per frame
            if n_channels > 1:
                arr = arr.reshape(-1, n_channels)
                arr = _np.max(_np.abs(arr), axis=1)
            else:
                arr = _np.abs(arr)

            # Downsample to ~1000 RMS windows (faster load, image renderer downsamples further anyway)
            n_points = min(1000, n_frames)
            window   = max(1, n_frames // n_points)
            # Trim to exact multiple
            usable = (len(arr) // window) * window
            if usable == 0:
                return []
            chunks  = arr[:usable].reshape(-1, window)
            rms     = _np.sqrt(_np.mean(chunks ** 2, axis=1))
            # Also compute peak per window for a peak envelope
            peak_env = _np.max(chunks, axis=1)

            peak_val = max(rms.max(), 1e-10)
            rms /= peak_val
            peak_env /= peak_val
            # Clip to 0-1
            rms = _np.clip(rms, 0.0, 1.0)
            peak_env = _np.clip(peak_env, 0.0, 1.0)

            return [
                (i / len(rms), float(rms[i]), float(peak_env[i]))
                for i in range(len(rms))
            ]
        except Exception:
            return []

    def _get_waveform_image(self, aclip, clip_width, track_height):
        """
        Get a cached waveform image for an audio clip.
        Returns (PhotoImage, x_offset) or (None, 0) if no waveform data.
        """
        from PIL import Image, ImageDraw, ImageTk

        wave_data = self._audio_waveforms.get(aclip.source_path, [])
        if not wave_data or clip_width < 10:
            return None, 0

        src_dur = aclip.source_duration or 1.0
        img_w = max(10, int(clip_width))
        img_h = max(10, int(track_height - 4))

        # Cache key includes all rendering parameters
        cache_key = (
            aclip.source_path,
            img_w,
            img_h,
            round(aclip.source_offset, 4),
            round(aclip.duration, 4),
            round(src_dur, 4)
        )

        if cache_key in self._waveform_images:
            return self._waveform_images[cache_key], 0

        # Build points mapping source time to image x position
        clip_start_time = aclip.source_offset
        clip_end_time = aclip.source_offset + aclip.duration
        clip_dur = max(aclip.duration, 0.001)

        # Collect points within clip's time range
        points = []
        for frac, amp_rms, amp_peak in wave_data:
            wave_time = frac * src_dur
            if clip_start_time <= wave_time <= clip_end_time:
                # Map to 0..img_w
                x = (wave_time - clip_start_time) / clip_dur * img_w
                points.append((x, amp_rms))

        if len(points) < 2:
            return None, 0

        # Downsample if we have more points than pixels
        if len(points) > img_w:
            step = len(points) / img_w
            points = [points[int(i * step)] for i in range(img_w)]

        # Render waveform to PIL Image
        img = Image.new("RGBA", (img_w, img_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        mid_y = img_h // 2
        half_h = (img_h - 4) / 2

        # Build polygon points
        top_pts = []
        bot_pts = []
        for x, amp in points:
            h = max(1, amp * half_h)
            top_pts.append((x, mid_y - h))
            bot_pts.append((x, mid_y + h))

        # Ensure we start and end at the edges
        if top_pts and top_pts[0][0] > 1:
            top_pts.insert(0, (0, mid_y - 1))
            bot_pts.insert(0, (0, mid_y + 1))
        if top_pts and top_pts[-1][0] < img_w - 1:
            top_pts.append((img_w - 1, mid_y - 1))
            bot_pts.append((img_w - 1, mid_y + 1))

        # Draw filled polygon and outlines
        if len(top_pts) >= 2:
            poly = top_pts + list(reversed(bot_pts))
            draw.polygon(poly, fill=(61, 122, 184, 255))
            draw.line(top_pts, fill=(108, 176, 232, 255), width=1)
            draw.line(bot_pts, fill=(108, 176, 232, 255), width=1)
            # Center line
            draw.line([(0, mid_y), (img_w - 1, mid_y)],
                      fill=(30, 48, 80, 200), width=1)

        # Convert to PhotoImage and cache
        photo = ImageTk.PhotoImage(img)
        # Limit cache size
        if len(self._waveform_images) > 50:
            keys = list(self._waveform_images.keys())
            for k in keys[:25]:
                del self._waveform_images[k]
        self._waveform_images[cache_key] = photo
        return photo, 0

    # ────────────────────────────────────────────────────────────
    # Timeline drawing
    # ────────────────────────────────────────────────────────────
    def _sec_to_x(self, sec):
        return TRACK_LABEL_W + (sec - self._scroll_x) * self._px_per_sec

    def _x_to_sec(self, x):
        return (x - TRACK_LABEL_W) / self._px_per_sec + self._scroll_x

    def _redraw_timeline(self):
        c = self._tl_canvas
        c.delete("all")
        cw = c.winfo_width() or 800
        ch = c.winfo_height() or 120
        total_dur = self.timeline_duration
        full_w = TRACK_LABEL_W + total_dur * self._px_per_sec + 40
        draw_w = max(full_w, cw)  # tracks fill at least the visible width

        # Update scrollbar thumb to reflect visible fraction & position
        visible_sec = max(0.01, (cw - TRACK_LABEL_W) / self._px_per_sec)
        if total_dur > 0:
            first = self._scroll_x / total_dur
            last = first + visible_sec / total_dur
            first = max(0.0, min(1.0, first))
            last = max(first, min(1.0, last))
            self._tl_xscroll.set(first, last)
        else:
            self._tl_xscroll.set(0.0, 1.0)

        y_off = RULER_HEIGHT

        # Ruler
        c.create_rectangle(0, 0, draw_w, RULER_HEIGHT,
                           fill=C.RULER_BG, outline="")
        c.create_line(0, RULER_HEIGHT - 1, draw_w,
                      RULER_HEIGHT - 1, fill=C.BORDER)
        tick_iv = self._compute_tick_interval()
        t = 0.0
        while t <= total_dur:
            x = self._sec_to_x(t)
            if TRACK_LABEL_W <= x <= full_w:
                c.create_line(x, RULER_HEIGHT - 10, x,
                              RULER_HEIGHT - 1,
                              fill=C.RULER_TICK, width=1)
                m = int(t) // 60
                s = t - m * 60
                lbl = f"{m}:{s:04.1f}" if m else f"{s:.1f}s"
                c.create_text(x, RULER_HEIGHT - 12, text=lbl,
                              fill=C.RULER_FG,
                              font=("Segoe UI", 6), anchor="s")
                hx = self._sec_to_x(t + tick_iv / 2)
                if TRACK_LABEL_W <= hx <= full_w:
                    c.create_line(hx, RULER_HEIGHT - 5, hx,
                                  RULER_HEIGHT - 1,
                                  fill=C.RULER_TICK, width=1)
            t += tick_iv

        # Tracks (dynamic: N video + N audio + 1 subs)
        track_info = []
        for vi in range(self._video_track_count):
            track_info.append((f"Video {vi + 1}" if self._video_track_count > 1 else "Video",
                               C.VIDEO_TRACK))
        for ai in range(self._audio_track_count):
            track_info.append((f"Audio {ai + 1}" if self._audio_track_count > 1 else "Audio",
                               C.AUDIO_TRACK))
        track_info.append(("Subs", C.SUB_TRACK))
        for i, (name, bg) in enumerate(track_info):
            ty = y_off + i * (TRACK_HEIGHT + TRACK_PADDING)
            c.create_rectangle(TRACK_LABEL_W, ty, draw_w,
                               ty + TRACK_HEIGHT,
                               fill=bg, outline="")
            c.create_line(TRACK_LABEL_W, ty + TRACK_HEIGHT,
                          draw_w, ty + TRACK_HEIGHT,
                          fill=C.TRACK_BORDER)
            c.create_rectangle(0, ty, TRACK_LABEL_W,
                               ty + TRACK_HEIGHT,
                               fill=C.LABEL_BG, outline="")
            c.create_line(TRACK_LABEL_W, ty, TRACK_LABEL_W,
                          ty + TRACK_HEIGHT, fill=C.TRACK_BORDER)
            c.create_text(TRACK_LABEL_W // 2,
                          ty + TRACK_HEIGHT // 2,
                          text=name, fill=C.TEXT_DIM,
                          font=("Segoe UI", 8))

        # Video clips with thumbnail previews (multi-track)
        for vclip in self.video_clips:
            track_idx = getattr(vclip, 'track_index', 0)
            vt_y = y_off + track_idx * (TRACK_HEIGHT + TRACK_PADDING)
            x1, x2 = self._sec_to_x(vclip.start), self._sec_to_x(vclip.end)
            # Selection highlight for video track (including multi-select)
            is_sel = (self._selected_track == "video" and self.selected_media_clip == vclip) or vclip in self._multi_selected
            # Draw clip background first (no outline yet)
            c.create_rectangle(x1, vt_y + 2, x2,
                               vt_y + TRACK_HEIGHT - 2,
                               fill=C.VIDEO_CLIP,
                               outline="")
            # Resize handles
            hw = 4
            c.create_rectangle(x1, vt_y + 2, x1 + hw,
                               vt_y + TRACK_HEIGHT - 2,
                               fill="#4a6a4a", outline="")
            c.create_rectangle(x2 - hw, vt_y + 2, x2,
                               vt_y + TRACK_HEIGHT - 2,
                               fill="#4a6a4a", outline="")
            # Draw thumbnails from video
            if vclip.source_type == "video" and self._video_thumbnails:
                thumb_h = TRACK_HEIGHT - 6
                thumb_w = max(1, int(thumb_h * 16 / 9))
                for t_val, photo in self._video_thumbnails:
                    # Only show thumbnails that fall within this clip's source range
                    if t_val < vclip.source_offset or t_val > vclip.source_offset + vclip.duration:
                        continue
                    # Map source time to timeline position
                    tx = self._sec_to_x(vclip.start + (t_val - vclip.source_offset))
                    # Skip if thumbnail would extend outside clip boundaries
                    if tx < x1 or tx + thumb_w > x2:
                        continue
                    if tx + thumb_w < TRACK_LABEL_W or tx > draw_w:
                        continue
                    c.create_image(tx, vt_y + 3, image=photo, anchor="nw")
            # Tile image bg thumbnail across clip
            elif vclip.source_type == "image" and self._video_thumb_strip:
                thumb_h = TRACK_HEIGHT - 6
                thumb_w = max(1, int(thumb_h * 16 / 9))
                tx = x1
                while tx + thumb_w <= x2:  # Only draw if thumbnail fits within clip
                    if tx >= TRACK_LABEL_W:
                        c.create_image(tx, vt_y + 3,
                                       image=self._video_thumb_strip,
                                       anchor="nw")
                    tx += thumb_w
            # Filename overlay
            mid_x = max(x1 + 20, min((x1 + x2) / 2, x2 - 20))
            icon = "\U0001F3AC" if vclip.source_type == "video" else "\U0001F5BC"
            # Shadow text for readability over thumbnails
            c.create_text(mid_x + 1, vt_y + TRACK_HEIGHT // 2 + 1,
                          text=f"{icon} {os.path.basename(vclip.source_path)}",
                          fill="#000000", font=("Segoe UI", 7, "bold"))
            c.create_text(mid_x, vt_y + TRACK_HEIGHT // 2,
                          text=f"{icon} {os.path.basename(vclip.source_path)}",
                          fill="#c0f0c0", font=("Segoe UI", 7, "bold"))
            # Draw white left/right borders
            c.create_line(x1, vt_y + 2, x1, vt_y + TRACK_HEIGHT - 2,
                          fill="#ffffff", width=1)
            c.create_line(x2, vt_y + 2, x2, vt_y + TRACK_HEIGHT - 2,
                          fill="#ffffff", width=1)
            # Draw selection highlight on top if selected
            if is_sel:
                c.create_rectangle(x1, vt_y + 2, x2,
                                   vt_y + TRACK_HEIGHT - 2,
                                   fill="", outline="#ffff00", width=3)

        # Audio clips with waveform preview (multi-track)
        for aclip in self.audio_clips:
            track_idx = getattr(aclip, 'track_index', 0)
            at_y = y_off + (self._video_track_count + track_idx) * (TRACK_HEIGHT + TRACK_PADDING)
            x1, x2 = self._sec_to_x(aclip.start), self._sec_to_x(aclip.end)
            # Selection highlight for audio track (including multi-select)
            is_sel = (self._selected_track == "audio" and self.selected_media_clip == aclip) or aclip in self._multi_selected
            # Draw clip background first (no outline yet)
            c.create_rectangle(x1, at_y + 2, x2,
                               at_y + TRACK_HEIGHT - 2,
                               fill=C.AUDIO_CLIP,
                               outline="")
            # Resize handles
            hw = 4
            c.create_rectangle(x1, at_y + 2, x1 + hw,
                               at_y + TRACK_HEIGHT - 2,
                               fill="#4a5a6a", outline="")
            c.create_rectangle(x2 - hw, at_y + 2, x2,
                               at_y + TRACK_HEIGHT - 2,
                               fill="#4a5a6a", outline="")
            # Draw waveform from cached image
            clip_w = x2 - x1
            wf_img, _ = self._get_waveform_image(aclip, clip_w, TRACK_HEIGHT)
            if wf_img:
                # Position at clip start (canvas handles clipping)
                c.create_image(x1, at_y + 2, image=wf_img, anchor="nw")
            # Filename overlay with shadow
            mid_x = max(x1 + 20, min((x1 + x2) / 2, x2 - 20))
            c.create_text(mid_x + 1, at_y + TRACK_HEIGHT // 2 + 1,
                          text=f"\U0001F3B5 {os.path.basename(aclip.source_path)}",
                          fill="#000000", font=("Segoe UI", 7, "bold"))
            c.create_text(mid_x, at_y + TRACK_HEIGHT // 2,
                          text=f"\U0001F3B5 {os.path.basename(aclip.source_path)}",
                          fill="#b0d0f0", font=("Segoe UI", 7, "bold"))
            # Draw white left/right borders
            c.create_line(x1, at_y + 2, x1, at_y + TRACK_HEIGHT - 2,
                          fill="#ffffff", width=1)
            c.create_line(x2, at_y + 2, x2, at_y + TRACK_HEIGHT - 2,
                          fill="#ffffff", width=1)
            # Draw selection highlight on top if selected
            if is_sel:
                c.create_rectangle(x1, at_y + 2, x2,
                                   at_y + TRACK_HEIGHT - 2,
                                   fill="", outline="#ffff00", width=3)

        # Subtitle clips with enhanced text previews
        st_y = y_off + (self._video_track_count + self._audio_track_count) * (TRACK_HEIGHT + TRACK_PADDING)
        for clip in self.subtitles:
            x1, x2 = self._sec_to_x(clip.start), self._sec_to_x(clip.end)
            if x2 - x1 < 3:
                x2 = x1 + 3
            # Selection highlight (including multi-select)
            is_sel = (clip is self.selected_clip) or clip in self._multi_selected
            fill = C.SUB_CLIP_SEL if is_sel else C.SUB_CLIP
            outline = "#ffffff" if is_sel else C.SUB_CLIP_OUT
            lw = 2 if is_sel else 1
            c.create_rectangle(x1, st_y + 2, x2,
                               st_y + TRACK_HEIGHT - 2,
                               fill=fill, outline=outline, width=lw)
            # Resize handles
            hw = 4
            c.create_rectangle(x1, st_y + 2, x1 + hw,
                               st_y + TRACK_HEIGHT - 2,
                               fill=C.SUB_HANDLE, outline="")
            c.create_rectangle(x2 - hw, st_y + 2, x2,
                               st_y + TRACK_HEIGHT - 2,
                               fill=C.SUB_HANDLE, outline="")
            avail = x2 - x1 - 10
            if avail > 14:
                text = clip.text.replace("\n", " ")  # Show newlines as spaces in timeline
                mc = max(1, int(avail / 6))
                if len(text) > mc:
                    text = text[:mc - 1] + "\u2026"
                # Main subtitle text (larger font with taller tracks)
                c.create_text(x1 + 6, st_y + TRACK_HEIGHT // 2 - 6,
                              text=text, fill=C.SUB_CLIP_TEXT,
                              font=("Segoe UI", 8), anchor="w")
                # Timing info below text
                dur_s = clip.end - clip.start
                time_lbl = f"{clip.start:.1f}s \u2013 {clip.end:.1f}s ({dur_s:.1f}s)"
                c.create_text(x1 + 6, st_y + TRACK_HEIGHT // 2 + 8,
                              text=time_lbl, fill=C.TEXT_DIM,
                              font=("Segoe UI", 6), anchor="w")

        # Playhead
        ph_x = self._sec_to_x(self._playback_time)
        track_bottom = y_off + self._total_track_count * (TRACK_HEIGHT + TRACK_PADDING)
        if TRACK_LABEL_W <= ph_x <= draw_w:
            c.create_line(ph_x, 0, ph_x, track_bottom,
                          fill=C.PLAYHEAD, width=1, tags="playhead")
            # Triangle indicator at ruler
            c.create_polygon(
                ph_x - 4, 0, ph_x + 4, 0, ph_x, 6,
                fill=C.PLAYHEAD, outline="", tags="playhead")

        # Redraw track labels on top so they stay visible when scrolling
        for i, (name, bg) in enumerate(track_info):
            ty = y_off + i * (TRACK_HEIGHT + TRACK_PADDING)
            c.create_rectangle(0, ty, TRACK_LABEL_W,
                               ty + TRACK_HEIGHT,
                               fill=C.LABEL_BG, outline="")
            c.create_line(TRACK_LABEL_W, ty, TRACK_LABEL_W,
                          ty + TRACK_HEIGHT, fill=C.TRACK_BORDER)
            c.create_text(TRACK_LABEL_W // 2,
                          ty + TRACK_HEIGHT // 2,
                          text=name, fill=C.TEXT_DIM,
                          font=("Segoe UI", 8))

    def _compute_tick_interval(self):
        target_px = 80
        raw = target_px / self._px_per_sec
        nices = [0.1, 0.25, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300]
        for n in nices:
            if n >= raw:
                return n
        return 600

    # ────────────────────────────────────────────────────────────
    # Timeline interaction
    # ────────────────────────────────────────────────────────────
    def _get_sub_track_y(self):
        y_off = RULER_HEIGHT
        st_y = y_off + (self._video_track_count + self._audio_track_count) * (TRACK_HEIGHT + TRACK_PADDING)
        return st_y, st_y + TRACK_HEIGHT

    def _get_track_at_y(self, y):
        """Return which track the y coordinate is in.
        Returns 'video', 'audio', 'subtitle', or None.
        """
        y_off = RULER_HEIGHT
        # Video tracks (0..N-1)
        for vi in range(self._video_track_count):
            vt_top = y_off + vi * (TRACK_HEIGHT + TRACK_PADDING)
            vt_bot = vt_top + TRACK_HEIGHT
            if vt_top <= y <= vt_bot:
                return "video"
        # Audio tracks (0..N-1)
        for ai in range(self._audio_track_count):
            at_top = y_off + (self._video_track_count + ai) * (TRACK_HEIGHT + TRACK_PADDING)
            at_bot = at_top + TRACK_HEIGHT
            if at_top <= y <= at_bot:
                return "audio"
        # Subtitle track
        sub_top = y_off + (self._video_track_count + self._audio_track_count) * (TRACK_HEIGHT + TRACK_PADDING)
        sub_bot = sub_top + TRACK_HEIGHT
        if sub_top <= y <= sub_bot:
            return "subtitle"
        return None

    def _get_audio_track_index_at_y(self, y):
        """Return which audio track index (0-based) the y coordinate is in, or -1."""
        y_off = RULER_HEIGHT
        for ai in range(self._audio_track_count):
            at_top = y_off + (self._video_track_count + ai) * (TRACK_HEIGHT + TRACK_PADDING)
            at_bot = at_top + TRACK_HEIGHT
            if at_top <= y <= at_bot:
                return ai
        return -1

    def _get_video_track_index_at_y(self, y):
        """Return which video track index (0-based) the y coordinate is in, or -1."""
        y_off = RULER_HEIGHT
        for vi in range(self._video_track_count):
            vt_top = y_off + vi * (TRACK_HEIGHT + TRACK_PADDING)
            vt_bot = vt_top + TRACK_HEIGHT
            if vt_top <= y <= vt_bot:
                return vi
        return -1

    def _find_clip_at(self, x, y):
        """Find clip at position. Returns (clip, mode, track_type).
        
        For video/audio tracks, clip is a MediaClip.
        For subtitle track, clip is a SubtitleClip.
        """
        track = self._get_track_at_y(y)
        sec = self._x_to_sec(x)

        if track == "video":
            track_idx = self._get_video_track_index_at_y(y)
            for clip in self.video_clips:
                if getattr(clip, 'track_index', 0) != track_idx:
                    continue
                if clip.start <= sec <= clip.end:
                    x1 = self._sec_to_x(clip.start)
                    x2 = self._sec_to_x(clip.end)
                    hz = 6
                    if x <= x1 + hz:
                        return clip, "resize_l", "video"
                    elif x >= x2 - hz:
                        return clip, "resize_r", "video"
                    else:
                        return clip, "select", "video"

        elif track == "audio":
            track_idx = self._get_audio_track_index_at_y(y)
            for clip in self.audio_clips:
                if getattr(clip, 'track_index', 0) != track_idx:
                    continue
                if clip.start <= sec <= clip.end:
                    x1 = self._sec_to_x(clip.start)
                    x2 = self._sec_to_x(clip.end)
                    hz = 6
                    if x <= x1 + hz:
                        return clip, "resize_l", "audio"
                    elif x >= x2 - hz:
                        return clip, "resize_r", "audio"
                    else:
                        return clip, "select", "audio"

        elif track == "subtitle":
            for clip in self.subtitles:
                if clip.start <= sec <= clip.end:
                    x1 = self._sec_to_x(clip.start)
                    x2 = self._sec_to_x(clip.end)
                    hz = 6
                    if x <= x1 + hz:
                        return clip, "resize_l", "subtitle"
                    elif x >= x2 - hz:
                        return clip, "resize_r", "subtitle"
                    else:
                        return clip, "move", "subtitle"

        return None, None, None

    def _on_tl_press(self, event):
        self._tl_canvas.focus_set()  # Defocus any textboxes
        x = self._tl_canvas.canvasx(event.x)
        y = self._tl_canvas.canvasy(event.y)
        ctrl_held = event.state & 0x4  # Ctrl key modifier

        # Click on ruler = seek
        if y < RULER_HEIGHT:
            sec = self._x_to_sec(x)
            sec = max(0, min(sec, self.timeline_duration))
            self._playback_time = sec
            self._refresh_preview()
            self._redraw_timeline()
            return

        clip, mode, track = self._find_clip_at(x, y)

        # Handle multi-selection with Ctrl+click
        if ctrl_held and clip:
            if clip in self._multi_selected:
                self._multi_selected.discard(clip)
            else:
                self._multi_selected.add(clip)
            self._redraw_timeline()
            self._tl_canvas.focus_set()
            return

        # Start box selection if clicking on empty area
        if not clip:
            if not ctrl_held:
                self._multi_selected.clear()
            self._box_select_start = (x, y)
            self._selected_track = None
            self.selected_clip = None
            self.selected_media_clip = None
            self._drag_mode = None
            self._drag_clip = None
            self._update_detail_panel()
            self._redraw_timeline()
            self._tl_canvas.focus_set()
            return

        # Regular click without Ctrl clears multi-selection (unless clicking on already multi-selected clip)
        if clip not in self._multi_selected:
            self._multi_selected.clear()

        if track == "video" and clip:
            self._selected_track = "video"
            self.selected_clip = None
            self.selected_media_clip = clip
            self._drag_mode = mode if mode in ("resize_l", "resize_r") else "move"
            self._drag_clip = clip
            self._drag_start_x = x
            self._drag_orig_start = clip.start
            self._drag_orig_end = clip.end
            self._drag_orig_source_offset = clip.source_offset
            # Store original positions for all multi-selected clips
            self._drag_multi_orig = {c: (c.start, c.end) for c in self._multi_selected}
            self._status_var.set(f"Selected: Video/Image - {os.path.basename(clip.source_path)}")
        elif track == "audio" and clip:
            self._selected_track = "audio"
            self.selected_clip = None
            self.selected_media_clip = clip
            self._drag_mode = mode if mode in ("resize_l", "resize_r") else "move"
            self._drag_clip = clip
            self._drag_start_x = x
            self._drag_orig_start = clip.start
            self._drag_orig_end = clip.end
            self._drag_orig_source_offset = clip.source_offset
            # Store original positions for all multi-selected clips
            self._drag_multi_orig = {c: (c.start, c.end) for c in self._multi_selected}
            self._status_var.set(f"Selected: Audio - {os.path.basename(clip.source_path)}")
        elif track == "subtitle" and clip:
            self._selected_track = "subtitle"
            self.selected_clip = clip
            self.selected_media_clip = None
            self._drag_mode = mode
            self._drag_clip = clip
            self._drag_start_x = x
            self._drag_orig_start = clip.start
            self._drag_orig_end = clip.end
            # Store original positions for all multi-selected clips
            self._drag_multi_orig = {c: (c.start, c.end) for c in self._multi_selected}
        self._update_detail_panel()
        self._redraw_timeline()
        # Keep focus on canvas, not text entry
        self._tl_canvas.focus_set()

    def _on_tl_drag(self, event):
        x = self._tl_canvas.canvasx(event.x)
        y = self._tl_canvas.canvasy(event.y)
        
        # Handle box selection drag
        if self._box_select_start:
            sx, sy = self._box_select_start
            # Delete old rectangle
            if self._box_select_rect:
                self._tl_canvas.delete(self._box_select_rect)
            # Draw new selection rectangle
            self._box_select_rect = self._tl_canvas.create_rectangle(
                sx, sy, x, y, outline="#00AAFF", width=1, dash=(4, 2))
            return
        
        if not self._drag_clip or not self._drag_mode:
            return
        dx = (x - self._drag_start_x) / self._px_per_sec
        clip = self._drag_clip
        md = 0.1
        if self._drag_mode == "move":
            ns = max(0, self._drag_orig_start + dx)
            d = self._drag_orig_end - self._drag_orig_start
            mt = self.timeline_duration
            if mt > d:
                ns = min(ns, mt - d)
            ns = max(0, ns)
            clip.start = ns
            clip.end = ns + d
            # Move all multi-selected clips by the same delta
            for c, (orig_start, orig_end) in self._drag_multi_orig.items():
                if c is clip:
                    continue
                dur = orig_end - orig_start
                new_start = max(0, orig_start + dx)
                if mt > dur:
                    new_start = min(new_start, mt - dur)
                c.start = new_start
                c.end = new_start + dur
        elif self._drag_mode == "resize_l":
            ns = max(0, self._drag_orig_start + dx)
            ns = min(ns, self._drag_orig_end - md)
            # For media clips, update source_offset when resizing left edge
            if isinstance(clip, MediaClip):
                offset_change = ns - self._drag_orig_start
                # Calculate original source_offset from stored values
                orig_source_offset = getattr(self, '_drag_orig_source_offset', clip.source_offset)
                clip.source_offset = max(0, orig_source_offset + offset_change)
            clip.start = ns
        elif self._drag_mode == "resize_r":
            ne = max(clip.start + md, self._drag_orig_end + dx)
            # For media clips, don't let it extend beyond source duration
            if isinstance(clip, MediaClip) and clip.source_duration > 0:
                max_end = clip.start + (clip.source_duration - clip.source_offset)
                ne = min(ne, max_end)
            clip.end = ne
        self._update_detail_panel()
        self._redraw_timeline()

    def _on_tl_release(self, event):
        # Finalize box selection
        if self._box_select_start:
            if self._box_select_rect:
                x = self._tl_canvas.canvasx(event.x)
                y = self._tl_canvas.canvasy(event.y)
                sx, sy = self._box_select_start
                # Get box bounds (normalize order)
                x1, x2 = min(sx, x), max(sx, x)
                y1, y2 = min(sy, y), max(sy, y)
                # Convert x to time
                t1 = self._x_to_sec(x1)
                t2 = self._x_to_sec(x2)
                y_off = RULER_HEIGHT
                # Video track
                video_y1 = y_off
                video_y2 = y_off + TRACK_HEIGHT
                if y1 < video_y2 and y2 > video_y1:
                    for clip in self.video_clips:
                        # Check if clip overlaps with time range
                        if clip.start < t2 and clip.end > t1:
                            self._multi_selected.add(clip)
                # Audio tracks
                for ai in range(self._audio_track_count):
                    track_y1 = y_off + (1 + ai) * (TRACK_HEIGHT + TRACK_PADDING)
                    track_y2 = track_y1 + TRACK_HEIGHT
                    if y1 < track_y2 and y2 > track_y1:
                        for clip in self.audio_clips:
                            if clip.track_index == ai:
                                if clip.start < t2 and clip.end > t1:
                                    self._multi_selected.add(clip)
                # Subtitle track
                sub_y1 = y_off + (1 + self._audio_track_count) * (TRACK_HEIGHT + TRACK_PADDING)
                sub_y2 = sub_y1 + TRACK_HEIGHT
                if y1 < sub_y2 and y2 > sub_y1:
                    for clip in self.subtitles:
                        if clip.start < t2 and clip.end > t1:
                            self._multi_selected.add(clip)
                # Clean up box selection
                self._tl_canvas.delete(self._box_select_rect)
                self._box_select_rect = None
                self._redraw_timeline()
            self._box_select_start = None
        
        self._drag_mode = None
        self._drag_clip = None
        self._drag_multi_orig = {}

    def _on_tl_hover(self, event):
        x = self._tl_canvas.canvasx(event.x)
        y = self._tl_canvas.canvasy(event.y)
        _, mode, track = self._find_clip_at(x, y)
        if mode in ("resize_l", "resize_r"):
            self._tl_canvas.config(cursor="sb_h_double_arrow")
        elif mode == "move":
            self._tl_canvas.config(cursor="fleur")
        elif mode == "select" and track in ("video", "audio"):
            self._tl_canvas.config(cursor="hand2")
        else:
            self._tl_canvas.config(cursor="")

    def _on_tl_mousewheel(self, event):
        if event.state & 0x4:
            factor = 1.15 if event.delta > 0 else 1 / 1.15
            self._px_per_sec = max(5, min(500, self._px_per_sec * factor))
            self._zoom_var.set(self._px_per_sec)
            self._redraw_timeline()
        else:
            delta = -event.delta / 120 * (40 / self._px_per_sec)
            self._scroll_x = max(
                0, min(self.timeline_duration, self._scroll_x + delta))
            self._redraw_timeline()

    def _on_style_change(self, *args):
        """Handle any style control change."""
        # Guard against calls before UI is fully built
        if not hasattr(self, '_preview_canvas'):
            return
        self._sync_style_vars()
        # Force canvas redraw
        self._preview_canvas.delete("all")
        self._refresh_preview()
        self.update_idletasks()

    def _trigger_style_refresh(self):
        """Callback for checkbox commands to refresh preview."""
        if hasattr(self, '_preview_canvas'):
            self._sync_style_vars()
            # Force canvas redraw
            self._preview_canvas.delete("all")
            self._refresh_preview()
            self.update_idletasks()

    def _on_text_case_change(self):
        """Handle text case combobox change."""
        if hasattr(self, '_preview_canvas'):
            self._sync_style_vars()
            self._preview_canvas.delete("all")
            self._refresh_preview()
            self.update_idletasks()

    def _on_zoom_change(self, value):
        self._px_per_sec = float(value)
        self._waveform_images.clear()  # Clear cached images at old zoom level
        self._redraw_timeline()

    def _on_timeline_scroll(self, *args):
        total_dur = self.timeline_duration
        if total_dur <= 0:
            return
        if args[0] == "moveto":
            self._scroll_x = max(0.0, float(args[1]) * total_dur)
        elif args[0] == "scroll":
            amount = int(args[1])
            unit = args[2] if len(args) > 2 else "units"
            if unit == "pages":
                cw = self._tl_canvas.winfo_width() or 800
                page_sec = (cw - TRACK_LABEL_W) / self._px_per_sec * 0.9
                self._scroll_x += amount * page_sec
            else:
                self._scroll_x += amount * (20 / self._px_per_sec)
            self._scroll_x = max(0.0, min(total_dur, self._scroll_x))
        self._redraw_timeline()

    def _on_tl_configure(self, event):
        """Debounced timeline resize - only redraw after dragging stops."""
        if self._timeline_resize_after:
            self.after_cancel(self._timeline_resize_after)
        self._timeline_resize_after = self.after(50, self._do_timeline_resize)

    def _do_timeline_resize(self):
        """Actually redraw the timeline after resize debounce."""
        self._timeline_resize_after = None
        self._redraw_timeline()


# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app = SubtitleEditorApp()
    app.mainloop()
