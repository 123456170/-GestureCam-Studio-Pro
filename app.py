import io
import json
import time
import math
import queue
import threading
import datetime
from pathlib import Path
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import cv2
import streamlit as st
from PIL import Image


# Optional real hand tracking.
# If mediapipe is not installed, the app still runs with simulated landmarks.
try:
    import mediapipe as mp
    HAVE_MEDIAPIPE = True
except Exception:
    mp = None
    HAVE_MEDIAPIPE = False


st.set_page_config(
    page_title="GestureCam Studio Pro",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded",
)

ss = st.session_state


# =============================================================================
# CSS / branding
# =============================================================================
def inject_css() -> None:
    st.markdown(
        """
        <style>
        html, body, [class*="css"] {
            font-family: Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
        }

        .block-container {
            padding-top: 1.1rem;
            padding-bottom: 4rem;
            max-width: 1500px;
        }

        .gcs-hero {
            background: linear-gradient(135deg, rgba(15,23,42,0.96), rgba(30,64,175,0.78), rgba(124,58,237,0.62));
            border: 1px solid rgba(255,255,255,0.10);
            border-radius: 22px;
            padding: 1.25rem 1.4rem;
            color: white;
            box-shadow: 0 12px 40px rgba(0,0,0,0.28);
            margin-bottom: 1rem;
        }

        .gcs-hero-title {
            font-size: 1.85rem;
            font-weight: 850;
            letter-spacing: -0.02em;
        }

        .gcs-hero-sub {
            opacity: 0.92;
            margin-top: 0.25rem;
            font-size: 0.98rem;
        }

        .gcs-card {
            background: linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.03));
            border: 1px solid rgba(255,255,255,0.10);
            border-radius: 18px;
            padding: 1rem;
            box-shadow: 0 8px 24px rgba(0,0,0,0.18);
        }

        .gcs-title {
            font-weight: 800;
            font-size: 1.02rem;
            margin-bottom: 0.65rem;
            color: #e5eefc;
        }

        .gcs-row {
            display: flex;
            justify-content: space-between;
            gap: 0.8rem;
            padding: 0.28rem 0;
            border-bottom: 1px solid rgba(255,255,255,0.06);
            color: #dbe7ff;
        }

        .gcs-row:last-child {
            border-bottom: none;
        }

        .gcs-pill {
            display: inline-block;
            padding: 0.18rem 0.55rem;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 700;
            background: rgba(34,197,94,0.16);
            color: #86efac;
            border: 1px solid rgba(34,197,94,0.24);
        }

        .gcs-pill.warn {
            background: rgba(245,158,11,0.16);
            color: #fcd34d;
            border-color: rgba(245,158,11,0.24);
        }

        .gcs-pill.bad {
            background: rgba(239,68,68,0.16);
            color: #fca5a5;
            border-color: rgba(239,68,68,0.24);
        }

        .gcs-video-shell {
            border: 1px solid rgba(255,255,255,0.10);
            border-radius: 20px;
            overflow: hidden;
            background: #0b1220;
            box-shadow: 0 14px 44px rgba(0,0,0,0.30);
        }

        .gcs-mini {
            color: #94a3b8;
            font-size: 0.88rem;
        }

        div[data-testid="stMetric"] {
            background: linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.03));
            border: 1px solid rgba(255,255,255,0.09);
            border-radius: 16px;
            padding: 0.8rem 0.9rem;
        }

        div[data-testid="stMetricLabel"] {
            color: #cbd5e1 !important;
        }

        div[data-testid="stMetricValue"] {
            color: #f8fafc !important;
            font-weight: 800 !important;
        }

        .stButton > button,
        .stDownloadButton > button,
        .stFileUploader > button {
            border-radius: 12px;
            font-weight: 650;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 12px 12px 0 0;
            padding: 0.55rem 0.9rem;
        }

        .stExpander {
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 14px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero() -> None:
    st.markdown(
        """
        <div class="gcs-hero">
          <div class="gcs-hero-title">🎥 GestureCam Studio Pro</div>
          <div class="gcs-hero-sub">
            Local-first hand-gesture filter studio • webcam support • animated live demo • no API key • no upload
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# Paths
# =============================================================================
BASE_DIR = Path.cwd()
CAPTURES_DIR = BASE_DIR / "captures"
RECORDINGS_DIR = BASE_DIR / "recordings"
LOGS_DIR = BASE_DIR / "logs"
DEMO_DIR = BASE_DIR / "demo"
SETTINGS_DIR = BASE_DIR / "settings"


def ensure_dirs() -> None:
    for p in [CAPTURES_DIR, RECORDINGS_DIR, LOGS_DIR, DEMO_DIR, SETTINGS_DIR]:
        p.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Config
# Every threshold has a one-line comment. These are tunable starting points.
# =============================================================================
CONFIG: Dict[str, Any] = {
    "capture": {
        "width": 640,           # default work width
        "height": 360,          # default work height
        "fps": 30,              # target simulation FPS
        "mirror": True,         # mirror the displayed camera view
    },
    "webcam": {
        "default_index": 0,     # default webcam index
        "max_index": 5,         # max selectable webcam index
        "buffer_size": 1,       # keep only newest frame
    },
    "tracking": {
        "downscale_width": 640, # max width for landmark processing
        "min_confidence": 0.45, # min confidence to accept a label
    },
    "gestures": {
        "swipe_dx_min": 0.22,             # min horizontal palm movement for swipe
        "swipe_dy_max_ratio": 0.5,        # max vertical drift as ratio of horizontal movement
        "swipe_window_ms": 300,           # swipe evaluation window
        "swipe_refractory_ms": 700,       # cooldown after swipe
        "swipe_reset_dx": 0.18,           # movement needed to release swipe latch
        "pinch_start": 0.28,              # pinch starts below this thumb-index ratio
        "pinch_end": 0.38,                # pinch ends above this thumb-index ratio
        "intensity_min_ratio": 0.05,      # pinch ratio mapped to intensity 0
        "intensity_max_ratio": 0.35,      # pinch ratio mapped to intensity 1
        "hold_ms": 1000,                  # open-palm hold duration for arming/photo
        "pose_stable_ms": 220,            # stable time for other static poses
        "hold_refractory_ms": 1200,       # cooldown after hold-based action
        "refractory_ms": 500,             # generic cooldown
        "arming_ttl_ms": 8000,            # armed state lifetime
        "circle_angle_deg": 270,          # cumulative turn for circle gesture
        "circle_window_ms": 1500,         # max time to complete circle
        "finger_extend_ratio": 1.10,      # tip distance vs PIP distance for extension
        "confidence_min": 0.45,           # min confidence for emitted event
        "two_hand_window_ms": 400,        # two-hand distance change window
        "two_hand_pending_delta": 0.05,   # marks two-hand interaction active
        "two_hand_spread_delta": 0.10,    # required distance increase for SPREAD
        "two_hand_refractory_ms": 900,    # cooldown after SPREAD
        "clap_distance": 0.12,            # palm distance for CLAP
        "clap_refractory_ms": 900,        # cooldown after CLAP
    },
    "filters": {
        "work_width": 640,      # preferred processing width
        "work_height": 360,     # preferred processing height
        "max_chain": 3,         # max chained filters
        "fps_floor": 20,        # FPS floor for quality warnings
    },
    "ui": {
        "autoplay_seconds": 4,  # auto-run demo seconds on first launch
        "preview_width": 160,   # filter preview thumbnail width
        "preview_height": 90,   # filter preview thumbnail height
    },
    "custom": {
        "target_samples": 40,       # suggested samples per custom gesture
        "augment_count": 6,         # synthetic augmentations per recorded sample
        "max_per_label": 120,       # max stored samples per label
        "reject_distance": 0.55,    # kNN rejection threshold
    },
    "performance": {
        "auto_quality": True,       # reduce resolution if FPS drops
        "latency_samples": 240,     # rolling latency sample count
    },
    "privacy": {
        "upload": False,                  # never upload video or landmarks
        "store_landmarks_only": True,     # replay stores landmarks only
        "record_only_on_demand": True,    # video recording only on explicit action
    },
}


# =============================================================================
# Source constants
# =============================================================================
SOURCE_SIM = "Simulated studio scene"
SOURCE_WEB = "Real webcam"
SOURCE_FILE = "Video file"
SOURCE_OPTIONS = [SOURCE_SIM, SOURCE_WEB, SOURCE_FILE]


# =============================================================================
# Utilities
# =============================================================================
def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _jsonable(v: Any) -> Any:
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    return str(v)


def _stamp() -> str:
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    arr = sorted(values)
    k = (len(arr) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(arr) - 1)
    if f == c:
        return float(arr[f])
    return float(arr[f] + (arr[c] - arr[f]) * (k - f))


def first_available(mapping: Dict[str, Any], *preferred_keys: str) -> Optional[Any]:
    """
    Safely return the first non-None value from a dict.

    This avoids bugs like:
        value = dict.get("right") or dict.get("left")
    when the values are NumPy arrays.
    """
    for key in preferred_keys:
        value = mapping.get(key)
        if value is not None:
            return value

    for value in mapping.values():
        if value is not None:
            return value

    return None


# =============================================================================
# Data types
# =============================================================================
@dataclass
class GestureEvent:
    type: str
    hand: str
    t_ms: int
    confidence: float
    params: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "hand": self.hand,
            "t_ms": int(self.t_ms),
            "confidence": float(self.confidence),
            "params": _jsonable(self.params),
        }


# =============================================================================
# One Euro filter and landmark smoothing
# =============================================================================
class OneEuroFilter:
    """Low-latency smoothing filter for noisy landmark coordinates."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.02, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev: Optional[float] = None
        self.dx_prev = 0.0
        self.t_prev: Optional[float] = None

    def _alpha(self, cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * max(cutoff, 1e-6))
        return 1.0 / (1.0 + tau / max(dt, 1e-6))

    def __call__(self, x: float, t: float) -> float:
        if self.t_prev is None or self.x_prev is None:
            self.x_prev = float(x)
            self.dx_prev = 0.0
            self.t_prev = float(t)
            return self.x_prev

        dt = max(float(t) - self.t_prev, 1e-6)
        dx = (float(x) - self.x_prev) / dt

        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * float(x) + (1.0 - a) * self.x_prev

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = float(t)
        return x_hat


class LandmarkSmoother:
    """Applies a One Euro filter independently to each landmark coordinate."""

    def __init__(self, n_landmarks: int = 21, min_cutoff: float = 1.0, beta: float = 0.02):
        self.filters = [
            [OneEuroFilter(min_cutoff=min_cutoff, beta=beta) for _ in range(2)]
            for _ in range(n_landmarks)
        ]

    def filter(self, pts: np.ndarray, t_s: float) -> np.ndarray:
        pts = np.asarray(pts, dtype=np.float32)
        out = np.empty_like(pts, dtype=np.float32)
        for i in range(pts.shape[0]):
            out[i, 0] = self.filters[i][0](float(pts[i, 0]), t_s)
            out[i, 1] = self.filters[i][1](float(pts[i, 1]), t_s)
        return out


# =============================================================================
# Hand geometry
# Landmark index map:
# 0 wrist, 4 thumb tip, 8 index tip, 12 middle tip, 16 ring tip, 20 pinky tip
# 5 index MCP, 9 middle MCP, 13 ring MCP, 17 pinky MCP
# =============================================================================
def dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)))


def palm_width(lm: np.ndarray) -> float:
    return max(dist(lm[5], lm[17]), 1e-6)


def palm_center(lm: np.ndarray) -> np.ndarray:
    idx = np.array([0, 5, 9, 13, 17], dtype=int)
    return np.mean(lm[idx], axis=0).astype(np.float32)


def pinch_ratio(lm: np.ndarray) -> float:
    return dist(lm[4], lm[8]) / palm_width(lm)


def finger_extended(lm: np.ndarray, tip: int, pip: int) -> bool:
    ratio = CONFIG["gestures"]["finger_extend_ratio"]
    return dist(lm[tip], lm[0]) > ratio * dist(lm[pip], lm[0])


def extended_finger_flags(lm: np.ndarray) -> Dict[str, bool]:
    return {
        "thumb": finger_extended(lm, 4, 3),
        "index": finger_extended(lm, 8, 6),
        "middle": finger_extended(lm, 12, 10),
        "ring": finger_extended(lm, 16, 14),
        "pinky": finger_extended(lm, 20, 18),
    }


def classify_static(lm: np.ndarray) -> Tuple[str, float]:
    flags = extended_finger_flags(lm)
    count = sum(bool(flags[k]) for k in ("index", "middle", "ring", "pinky"))
    thumb = bool(flags["thumb"])

    if count == 0 and not thumb:
        return "FIST", 0.88

    if count == 4 and thumb:
        return "OPEN_PALM", 0.90

    if count == 1 and flags["index"]:
        if lm[8, 1] < lm[5, 1] - 0.03:
            return "POINTING_UP", 0.82
        return "POINT", 0.76

    if count == 2 and flags["index"] and flags["middle"]:
        return "VICTORY", 0.84

    if thumb and count == 0:
        if lm[4, 1] < lm[0, 1] - 0.05:
            return "THUMB_UP", 0.80
        if lm[4, 1] > lm[0, 1] + 0.05:
            return "THUMB_DOWN", 0.80

    if count in (3, 4):
        return f"FINGERS_{count}", 0.70

    return "UNKNOWN", 0.0


# =============================================================================
# Gesture engine
# =============================================================================
class GestureEngine:
    """Turns smoothed landmark streams into debounced GestureEvents."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.smoothers: Dict[str, LandmarkSmoother] = {}
        self.filtered: Dict[str, np.ndarray] = {}
        self.history: Dict[str, List[Tuple[int, np.ndarray]]] = {}
        self.circle_history: Dict[str, List[Tuple[int, np.ndarray]]] = {}
        self.last_label: Dict[str, str] = {}
        self.label_start: Dict[str, int] = {}
        self.hold_progress: Dict[str, float] = {}
        self.last_action: Dict[str, Dict[str, int]] = {}
        self.last_pinch_event: Dict[str, int] = {}
        self.pinch_active: Dict[str, bool] = {}
        self.swipe_lock: Dict[str, Dict[str, Any]] = {}
        self.circle_lock_until: Dict[str, int] = {}
        self.two_history: List[Tuple[int, float]] = []
        self.two_hand_until: int = 0

    def can_emit(self, hand: str, typ: str, t_ms: int, cooldown_ms: Optional[int] = None) -> bool:
        g = self.config["gestures"]
        cd = int(cooldown_ms if cooldown_ms is not None else g["refractory_ms"])
        last = self.last_action.setdefault(hand, {})
        if t_ms - last.get(typ, -10**9) < cd:
            return False
        last[typ] = t_ms
        return True

    def detect_swipe(self, hand: str, t_ms: int) -> Optional[GestureEvent]:
        g = self.config["gestures"]
        hist = self.history.get(hand, [])
        if len(hist) < 2:
            return None

        _, p0 = hist[0]
        _, p1 = hist[-1]
        dx = float(p1[0] - p0[0])
        dy = float(p1[1] - p0[1])

        if abs(dx) < g["swipe_dx_min"]:
            return None
        if abs(dy) >= g["swipe_dy_max_ratio"] * abs(dx):
            return None

        typ = "SWIPE_RIGHT" if dx > 0 else "SWIPE_LEFT"
        lock = self.swipe_lock.get(hand)
        if lock and lock["dir"] == typ and abs(float(p1[0]) - float(lock["x"])) < g["swipe_reset_dx"]:
            return None

        if not self.can_emit(hand, typ, t_ms, g["swipe_refractory_ms"]):
            return None

        self.swipe_lock[hand] = {"dir": typ, "x": float(p1[0])}
        self.history[hand] = []
        return GestureEvent(typ, hand, t_ms, 0.80, {"dx": dx, "dy": dy})

    def detect_circle(self, hand: str, t_ms: int) -> Optional[GestureEvent]:
        g = self.config["gestures"]
        hist = self.circle_history.get(hand, [])
        if len(hist) < 8 or t_ms < self.circle_lock_until.get(hand, 0):
            return None

        pts = np.array([p for _, p in hist], dtype=np.float32)
        angle_sum = 0.0
        for i in range(1, len(pts) - 1):
            v1 = pts[i] - pts[i - 1]
            v2 = pts[i + 1] - pts[i]
            cross = float(v1[0] * v2[1] - v1[1] * v2[0])
            dot = float(np.dot(v1, v2)) + 1e-6
            angle_sum += math.degrees(math.atan2(cross, dot))

        if abs(angle_sum) < g["circle_angle_deg"]:
            return None

        typ = "CIRCLE_CW" if angle_sum > 0 else "CIRCLE_CCW"
        if not self.can_emit(hand, typ, t_ms, g["refractory_ms"]):
            return None

        self.circle_history[hand] = []
        self.circle_lock_until[hand] = t_ms + 800
        return GestureEvent(typ, hand, t_ms, 0.76, {"angle_deg": angle_sum})

    def detect_two_hand(self, t_ms: int) -> List[GestureEvent]:
        g = self.config["gestures"]
        events: List[GestureEvent] = []

        if "right" not in self.filtered or "left" not in self.filtered:
            return events

        # While both hands are visible, suppress single-hand gestures briefly.
        self.two_hand_until = max(self.two_hand_until, t_ms + 200)

        cr = palm_center(self.filtered["right"])
        cl = palm_center(self.filtered["left"])
        d = dist(cr, cl)

        self.two_history.append((t_ms, d))
        min_t = t_ms - g["two_hand_window_ms"]
        while self.two_history and self.two_history[0][0] < min_t:
            self.two_history.pop(0)

        if len(self.two_history) < 2:
            return events

        d0 = self.two_history[0][1]
        delta = d - d0

        if delta > g["two_hand_pending_delta"]:
            self.two_hand_until = max(self.two_hand_until, t_ms + 300)

        if delta > g["two_hand_spread_delta"]:
            if self.can_emit("both", "SPREAD", t_ms, g["two_hand_refractory_ms"]):
                zoom = clamp(d / max(d0, 1e-6), 1.0, 2.0)
                events.append(GestureEvent("SPREAD", "both", t_ms, 0.72, {"zoom": zoom, "distance": d}))

        if d < g["clap_distance"]:
            if self.can_emit("both", "CLAP", t_ms, g["clap_refractory_ms"]):
                events.append(GestureEvent("CLAP", "both", t_ms, 0.75, {"distance": d}))

        return events

    def process(self, hands: Dict[str, np.ndarray], t_ms: int) -> List[GestureEvent]:
        g = self.config["gestures"]
        t_s = t_ms / 1000.0

        # Remove state for hands that disappeared.
        active = set(hands.keys())
        for hand in list(self.filtered.keys()):
            if hand not in active:
                self.filtered.pop(hand, None)
                self.history.pop(hand, None)
                self.circle_history.pop(hand, None)
                self.last_label.pop(hand, None)
                self.label_start.pop(hand, None)
                self.hold_progress.pop(hand, None)
                self.pinch_active.pop(hand, None)

        # First pass: smooth and update per-hand histories.
        for hand, raw in hands.items():
            if hand not in self.smoothers:
                self.smoothers[hand] = LandmarkSmoother(n_landmarks=21, min_cutoff=1.0, beta=0.02)
            lm = self.smoothers[hand].filter(raw, t_s)
            self.filtered[hand] = lm

            center = palm_center(lm)

            hist = self.history.setdefault(hand, [])
            hist.append((t_ms, center))
            min_t = t_ms - g["swipe_window_ms"]
            while hist and hist[0][0] < min_t:
                hist.pop(0)

            chist = self.circle_history.setdefault(hand, [])
            chist.append((t_ms, center))
            min_tc = t_ms - g["circle_window_ms"]
            while chist and chist[0][0] < min_tc:
                chist.pop(0)

            if len(hist) >= 2 and abs(float(hist[-1][1][0] - hist[-2][1][0])) < 0.003:
                self.swipe_lock.pop(hand, None)

        if not ("right" in self.filtered and "left" in self.filtered):
            self.two_history.clear()

        events: List[GestureEvent] = []

        # Two-hand gestures have priority over per-hand static poses.
        two_events = self.detect_two_hand(t_ms)
        if two_events:
            events.extend(two_events)

        if t_ms < self.two_hand_until:
            return events

        # Per-hand dynamic and static gestures.
        for hand, lm in list(self.filtered.items()):
            if hand not in hands:
                continue

            circle_ev = self.detect_circle(hand, t_ms)
            if circle_ev is not None:
                self.hold_progress[hand] = 0.0
                events.append(circle_ev)
                continue

            swipe_ev = self.detect_swipe(hand, t_ms)
            if swipe_ev is not None:
                self.hold_progress[hand] = 0.0
                events.append(swipe_ev)
                continue

            pr = pinch_ratio(lm)
            if pr < g["pinch_start"]:
                self.pinch_active[hand] = True
            elif pr > g["pinch_end"]:
                self.pinch_active[hand] = False

            if self.pinch_active.get(hand, False):
                self.hold_progress[hand] = 0.0
                denom = max(g["intensity_max_ratio"] - g["intensity_min_ratio"], 1e-6)
                intensity = clamp((pr - g["intensity_min_ratio"]) / denom, 0.0, 1.0)
                if t_ms - self.last_pinch_event.get(hand, 0) > 120:
                    self.last_pinch_event[hand] = t_ms
                    events.append(
                        GestureEvent(
                            "PINCH",
                            hand,
                            t_ms,
                            0.75,
                            {"intensity": float(intensity), "pinch_ratio": float(pr)},
                        )
                    )
                continue

            label, conf = classify_static(lm)
            if self.last_label.get(hand) != label:
                self.last_label[hand] = label
                self.label_start[hand] = t_ms

            hold = t_ms - self.label_start.get(hand, t_ms)
            needed = g["hold_ms"] if label == "OPEN_PALM" else g["pose_stable_ms"]
            self.hold_progress[hand] = clamp(hold / max(needed, 1), 0.0, 1.0)

            if label != "UNKNOWN" and conf >= g["confidence_min"] and hold >= needed:
                cooldown = g["hold_refractory_ms"] if label == "OPEN_PALM" else g["refractory_ms"]
                if self.can_emit(hand, label, t_ms, cooldown):
                    events.append(GestureEvent(label, hand, t_ms, conf, {"hold_ms": hold}))

        return events

    def feedback(self) -> Dict[str, Any]:
        progress = max(self.hold_progress.values(), default=0.0)
        if self.last_label:
            last_label = max(self.last_label.items(), key=lambda kv: self.label_start.get(kv[0], 0))[0]
        else:
            last_label = "NONE"
        return {"hold_progress": float(progress), "last_label": last_label}


# =============================================================================
# Filter helpers and caches
# =============================================================================
_GRID_CACHE: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}
_MASK_CACHE: Dict[Tuple[int, int], np.ndarray] = {}
_GRAIN_CACHE: Dict[Tuple[int, int], np.ndarray] = {}
_BASE_CACHE: Dict[Tuple[int, int], np.ndarray] = {}


def _grid(h: int, w: int) -> Tuple[np.ndarray, np.ndarray]:
    key = (h, w)
    if key not in _GRID_CACHE:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        _GRID_CACHE[key] = (xx, yy)
    return _GRID_CACHE[key]


def _uint(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr, 0, 255).astype(np.uint8)


def _blend(orig: np.ndarray, filt: np.ndarray, a: float) -> np.ndarray:
    if a <= 0:
        return orig.copy()
    return _uint((1.0 - a) * orig.astype(np.float32) + a * filt.astype(np.float32))


def _subject_mask(h: int, w: int) -> np.ndarray:
    key = (h, w)
    if key not in _MASK_CACHE:
        mask = np.zeros((h, w), dtype=np.float32)
        center = (int(w * 0.50), int(h * 0.62))
        axes = (max(1, int(w * 0.22)), max(1, int(h * 0.36)))
        cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1)
        mask = cv2.GaussianBlur(mask, (0, 0), 3.0)
        _MASK_CACHE[key] = mask
    return _MASK_CACHE[key]


def _grain_noise(h: int, w: int) -> np.ndarray:
    key = (h, w)
    if key not in _GRAIN_CACHE:
        rng = np.random.default_rng(7)
        _GRAIN_CACHE[key] = rng.normal(0.0, 1.0, (h, w, 3)).astype(np.float32)
    return _GRAIN_CACHE[key]


def _get_hands(ctx: Optional[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    if not ctx:
        return {}
    hands = ctx.get("hands", {})
    if hands is None:
        return {}
    return hands


# =============================================================================
# Filters: color
# =============================================================================
def f_original(frame: np.ndarray, intensity: float, ctx: Optional[Dict[str, Any]] = None) -> np.ndarray:
    return frame.copy()


def f_warm(frame: np.ndarray, intensity: float, ctx: Optional[Dict[str, Any]] = None) -> np.ndarray:
    if intensity <= 0:
        return frame.copy()
    f = frame.astype(np.float32)
    f[..., 0] *= 1.0 + 0.35 * intensity
    f[..., 2] *= 1.0 - 0.20 * intensity
    return _uint(f)


def f_cool(frame: np.ndarray, intensity: float, ctx: Optional[Dict[str, Any]] = None) -> np.ndarray:
    if intensity <= 0:
        return frame.copy()
    f = frame.astype(np.float32)
    f[..., 0] *= 1.0 - 0.20 * intensity
    f[..., 2] *= 1.0 + 0.35 * intensity
    return _uint(f)


def f_blackwhite(frame: np.ndarray, intensity: float, ctx: Optional[Dict[str, Any]] = None) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    bw = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    return _blend(frame, bw, intensity)


def f_sepia(frame: np.ndarray, intensity: float, ctx: Optional[Dict[str, Any]] = None) -> np.ndarray:
    mat = np.array(
        [
            [0.393, 0.769, 0.189],
            [0.349, 0.686, 0.168],
            [0.272, 0.534, 0.131],
        ],
        dtype=np.float32,
    )
    sepia = np.dot(frame.astype(np.float32), mat.T)
    return _blend(frame, _uint(sepia), intensity)


def f_vintage(frame: np.ndarray, intensity: float, ctx: Optional[Dict[str, Any]] = None) -> np.ndarray:
    if intensity <= 0:
        return frame.copy()
    f = frame.astype(np.float32)
    f = f * (1.0 - 0.12 * intensity) + 20.0 * intensity
    f[..., 0] *= 1.0 + 0.10 * intensity
    f[..., 2] *= 1.0 - 0.08 * intensity
    return _uint(f)


def f_teal_orange(frame: np.ndarray, intensity: float, ctx: Optional[Dict[str, Any]] = None) -> np.ndarray:
    if intensity <= 0:
        return frame.copy()

    f = frame.astype(np.float32)

    # Fixed broadcasting: lum is 2D (H, W), not (H, W, 1).
    lum = f.mean(axis=2) / 255.0
    shadow = 1.0 - lum
    highlight = lum

    f[..., 0] = f[..., 0] * (1.0 + 0.25 * highlight * intensity)
    f[..., 2] = f[..., 2] * (1.0 + 0.25 * shadow * intensity)
    f[..., 1] = f[..., 1] * (1.0 - 0.05 * intensity)

    return _uint(f)


def f_hue_rotate(frame: np.ndarray, intensity: float, ctx: Optional[Dict[str, Any]] = None) -> np.ndarray:
    if intensity <= 0:
        return frame.copy()
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + intensity * 179.0) % 180.0
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    return _blend(frame, out, intensity)


def f_saturation(frame: np.ndarray, intensity: float, ctx: Optional[Dict[str, Any]] = None) -> np.ndarray:
    if intensity <= 0:
        return frame.copy()
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * (1.0 + intensity), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def f_contrast_curve(frame: np.ndarray, intensity: float, ctx: Optional[Dict[str, Any]] = None) -> np.ndarray:
    if intensity <= 0:
        return frame.copy()
    f = frame.astype(np.float32)
    factor = 1.0 + intensity
    out = (f - 128.0) * factor + 128.0
    return _uint(out)


# =============================================================================
# Filters: art
# =============================================================================
def f_cartoon(frame: np.ndarray, intensity: float, ctx: Optional[Dict[str, Any]] = None) -> np.ndarray:
    if intensity <= 0:
        return frame.copy()
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    smooth = cv2.bilateralFilter(bgr, 9, 50.0 + 100.0 * intensity, 50.0 + 100.0 * intensity)
    gray = cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 7)
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    combined = cv2.bitwise_and(smooth, edges_bgr)
    out = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
    return _blend(frame, out, intensity)


def f_pencil(frame: np.ndarray, intensity: float, ctx: Optional[Dict[str, Any]] = None) -> np.ndarray:
    if intensity <= 0:
        return frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    inv = 255 - gray
    blur = cv2.GaussianBlur(inv, (21, 21), 0)
    denom = 255.0 - blur.astype(np.float32) + 1.0
    sketch = np.clip(gray.astype(np.float32) / denom * 256.0, 0, 255).astype(np.uint8)
    out = cv2.cvtColor(sketch, cv2.COLOR_GRAY2RGB)
    return _blend(frame, out, intensity)


def f_neon(frame: np.ndarray, intensity: float, ctx: Optional[Dict[str, Any]] = None) -> np.ndarray:
    if intensity <= 0:
        return frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    neon = np.zeros_like(frame)
    neon[..., 0] = edges
    neon[..., 1] = edges // 2
    neon[..., 2] = 255
    return _blend(frame, neon, intensity)


def f_pixelate(frame: np.ndarray, intensity: float, ctx: Optional[Dict[str, Any]] = None) -> np.ndarray:
    if intensity <= 0:
        return frame.copy()
    h, w = frame.shape[:2]
    block = max(2, int(2 + 40 * intensity))
    nw = max(1, w // block)
    nh = max(1, h // block)
    down = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_NEAREST)
    up = cv2.resize(down, (w, h), interpolation=cv2.INTER_NEAREST)
    return up


def f_halftone(frame: np.ndarray, intensity: float, ctx: Optional[Dict[str, Any]] = None) -> np.ndarray:
    if intensity <= 0:
        return frame.copy()
    h, w = frame.shape[:2]
    step = max(4, int(10 - 6 * intensity))
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    xx, yy = _grid(h, w)
    cx = (xx % step) - step / 2.0
    cy = (yy % step) - step / 2.0
    radius = (gray.astype(np.float32) / 255.0) * (step * 0.45)
    mask = (cx * cx + cy * cy) <= (radius * radius)
    out = np.full_like(frame, 255)
    out[mask] = 0
    return _blend(frame, out, intensity)


def f_posterize(frame: np.ndarray, intensity: float, ctx: Optional[Dict[str, Any]] = None) -> np.ndarray:
    if intensity <= 0:
        return frame.copy()
    bits = max(1, int(round(8 - 6 * intensity)))
    shift = 8 - bits
    out = (frame >> shift) << shift
    return out


def f_dream_glow(frame: np.ndarray, intensity: float, ctx: Optional[Dict[str, Any]] = None) -> np.ndarray:
    if intensity <= 0:
        return frame.copy()
    blur = cv2.GaussianBlur(frame, (0, 0), 2.0 + 8.0 * intensity)
    glow = _uint(frame.astype(np.float32) + blur.astype(np.float32) * (0.45 * intensity))
    return _blend(frame, glow, intensity)


# =============================================================================
# Filters: distortion
# =============================================================================
def f_bulge(frame: np.ndarray, intensity: float, ctx: Optional[Dict[str, Any]] = None) -> np.ndarray:
    if intensity <= 0:
        return frame.copy()
    h, w = frame.shape[:2]
    xx, yy = _grid(h, w)
    cx, cy = w / 2.0, h / 2.0
    R = min(h, w) * 0.45
    dx = xx - cx
    dy = yy - cy
    r = np.sqrt(dx * dx + dy * dy) + 1e-6
    factor = np.ones_like(r, dtype=np.float32)
    inner = r < R
    factor[inner] = 1.0 - 0.45 * intensity * (1.0 - r[inner] / R) ** 2
    mapx = (cx + dx * factor).astype(np.float32)
    mapy = (cy + dy * factor).astype(np.float32)
    return cv2.remap(frame, mapx, mapy, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def f_swirl(frame: np.ndarray, intensity: float, ctx: Optional[Dict[str, Any]] = None) -> np.ndarray:
    if intensity <= 0:
        return frame.copy()
    h, w = frame.shape[:2]
    xx, yy = _grid(h, w)
    cx, cy = w / 2.0, h / 2.0
    R = min(h, w) * 0.45
    dx = xx - cx
    dy = yy - cy
    r = np.sqrt(dx * dx + dy * dy) + 1e-6
    angle = np.zeros_like(r, dtype=np.float32)
    inner = r < R
    angle[inner] = intensity * 2.0 * math.pi * (1.0 - r[inner] / R) ** 2
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    mapx = (cx + dx * cos_a - dy * sin_a).astype(np.float32)
    mapy = (cy + dx * sin_a + dy * cos_a).astype(np.float32)
    return cv2.remap(frame, mapx, mapy, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def f_mirror(frame: np.ndarray, intensity: float, ctx: Optional[Dict[str, Any]] = None) -> np.ndarray:
    if intensity <= 0:
        return frame.copy()
    h, w = frame.shape[:2]
    xx, yy = _grid(h, w)
    mw = int(intensity * w * 0.5)
    if mw <= 1:
        return frame.copy()
    mapx = xx.copy()
    left = xx < mw
    mapx[left] = mw - 1 - xx[left]
    mapx = np.clip(mapx, 0, w - 1).astype(np.float32)
    mapy = yy.astype(np.float32)
    return cv2.remap(frame, mapx, mapy, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def f_kaleidoscope(frame: np.ndarray, intensity: float, ctx: Optional[Dict[str, Any]] = None) -> np.ndarray:
    if intensity <= 0:
        return frame.copy()
    h, w = frame.shape[:2]
    xx, yy = _grid(h, w)
    cx, cy = w / 2.0, h / 2.0
    dx = xx - cx
    dy = yy - cy
    r = np.sqrt(dx * dx + dy * dy)
    theta = np.arctan2(dy, dx)
    sectors = max(2, int(2 + 10 * intensity))
    sector = 2.0 * math.pi / sectors
    folded = np.abs(np.mod(theta, sector) - sector / 2.0)
    mapx = (cx + r * np.cos(folded)).astype(np.float32)
    mapy = (cy + r * np.sin(folded)).astype(np.float32)
    return cv2.remap(frame, mapx, mapy, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def f_ripple(frame: np.ndarray, intensity: float, ctx: Optional[Dict[str, Any]] = None) -> np.ndarray:
    if intensity <= 0:
        return frame.copy()
    h, w = frame.shape[:2]
    xx, yy = _grid(h, w)
    cx, cy = w / 2.0, h / 2.0
    dx = xx - cx
    dy = yy - cy
    r = np.sqrt(dx * dx + dy * dy)
    amp = 10.0 * intensity
    mapx = (xx + amp * np.sin(r / 12.0)).astype(np.float32)
    mapy = (yy + amp * np.cos(r / 12.0)).astype(np.float32)
    return cv2.remap(frame, mapx, mapy, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def f_tilt_shift(frame: np.ndarray, intensity: float, ctx: Optional[Dict[str, Any]] = None) -> np.ndarray:
    if intensity <= 0:
        return frame.copy()
    h, w = frame.shape[:2]
    blur = cv2.GaussianBlur(frame, (0, 0), 2.0 + 10.0 * intensity)
    out = frame.copy()
    top = int(h * 0.22)
    bottom = int(h * 0.78)
    out[:top, :, :] = blur[:top, :, :]
    out[bottom:, :, :] = blur[bottom:, :, :]
    return _blend(frame, out, intensity)


# =============================================================================
# Filters: overlay
# =============================================================================
def f_vignette(frame: np.ndarray, intensity: float, ctx: Optional[Dict[str, Any]] = None) -> np.ndarray:
    if intensity <= 0:
        return frame.copy()
    h, w = frame.shape[:2]
    xx, yy = _grid(h, w)
    cx, cy = w / 2.0, h / 2.0
    nx = (xx - cx) / (w * 0.5)
    ny = (yy - cy) / (h * 0.5)
    r = np.sqrt(nx * nx + ny * ny)
    mask = 1.0 - intensity * np.clip(r - 0.45, 0.0, 1.0) * 1.35
    mask = np.clip(mask, 0.0, 1.0)
    out = frame.astype(np.float32) * mask[..., None]
    return _uint(out)


def f_film_grain(frame: np.ndarray, intensity: float, ctx: Optional[Dict[str, Any]] = None) -> np.ndarray:
    if intensity <= 0:
        return frame.copy()
    h, w = frame.shape[:2]
    noise = _grain_noise(h, w)
    out = frame.astype(np.float32) + noise * (18.0 * intensity)
    return _uint(out)


def f_rgb_glitch(frame: np.ndarray, intensity: float, ctx: Optional[Dict[str, Any]] = None) -> np.ndarray:
    if intensity <= 0:
        return frame.copy()
    shift = int(1 + 15 * intensity)
    out = frame.copy()
    out[..., 0] = np.roll(out[..., 0], shift, axis=1)
    out[..., 2] = np.roll(out[..., 2], -shift, axis=1)
    return out


def f_scanlines(frame: np.ndarray, intensity: float, ctx: Optional[Dict[str, Any]] = None) -> np.ndarray:
    if intensity <= 0:
        return frame.copy()
    out = frame.astype(np.float32)
    out[::2, :, :] *= (1.0 - 0.45 * intensity)
    return _uint(out)


def f_light_leak(frame: np.ndarray, intensity: float, ctx: Optional[Dict[str, Any]] = None) -> np.ndarray:
    if intensity <= 0:
        return frame.copy()
    h, w = frame.shape[:2]
    xx, yy = _grid(h, w)
    dist2 = ((xx - w * 0.85) ** 2) / (w * w * 0.06) + ((yy - h * 0.12) ** 2) / (h * h * 0.06)
    leak = np.exp(-dist2).astype(np.float32)
    tint = np.array([90.0, 50.0, 20.0], dtype=np.float32) * intensity
    out = frame.astype(np.float32) + leak[..., None] * tint
    return _uint(out)


def f_frame_border(frame: np.ndarray, intensity: float, ctx: Optional[Dict[str, Any]] = None) -> np.ndarray:
    if intensity <= 0:
        return frame.copy()
    h, w = frame.shape[:2]
    out = frame.copy()
    thick = max(1, int(min(h, w) * 0.08 * intensity))
    cv2.rectangle(out, (thick // 2, thick // 2), (w - thick // 2, h - thick // 2), (255, 255, 255), thick)
    return out


def f_old_tv(frame: np.ndarray, intensity: float, ctx: Optional[Dict[str, Any]] = None) -> np.ndarray:
    if intensity <= 0:
        return frame.copy()
    out = f_scanlines(frame, intensity, ctx)
    out = f_vignette(out, intensity, ctx)
    out = f_film_grain(out, intensity * 0.7, ctx)
    return out


# =============================================================================
# Filters: background effects
# =============================================================================
def f_background_blur(frame: np.ndarray, intensity: float, ctx: Optional[Dict[str, Any]] = None) -> np.ndarray:
    if intensity <= 0:
        return frame.copy()
    h, w = frame.shape[:2]
    mask = _subject_mask(h, w)[..., None]
    blur = cv2.GaussianBlur(frame, (0, 0), 3.0 + 15.0 * intensity)
    bg_out = frame.astype(np.float32) * mask + blur.astype(np.float32) * (1.0 - mask)
    return _blend(frame, _uint(bg_out), intensity)


def f_background_color(frame: np.ndarray, intensity: float, ctx: Optional[Dict[str, Any]] = None) -> np.ndarray:
    if intensity <= 0:
        return frame.copy()
    h, w = frame.shape[:2]
    mask = _subject_mask(h, w)[..., None]
    color = np.array([25.0, 90.0, 130.0], dtype=np.float32)
    bg_out = frame.astype(np.float32) * mask + color * (1.0 - mask)
    return _blend(frame, _uint(bg_out), intensity)


def f_background_replace(frame: np.ndarray, intensity: float, ctx: Optional[Dict[str, Any]] = None) -> np.ndarray:
    if intensity <= 0:
        return frame.copy()
    h, w = frame.shape[:2]
    mask = _subject_mask(h, w)[..., None]
    xx, yy = _grid(h, w)
    checker = (((xx // 20) + (yy // 20)) % 2).astype(np.uint8) * 255
    bg = np.stack([checker, 255 - checker, np.full_like(checker, 128)], axis=-1).astype(np.uint8)
    bg_out = frame.astype(np.float32) * mask + bg.astype(np.float32) * (1.0 - mask)
    return _blend(frame, _uint(bg_out), intensity)


def f_spotlight(frame: np.ndarray, intensity: float, ctx: Optional[Dict[str, Any]] = None) -> np.ndarray:
    if intensity <= 0:
        return frame.copy()
    h, w = frame.shape[:2]
    mask = _subject_mask(h, w)
    light = 0.18 + 0.82 * mask
    out = frame.astype(np.float32) * light[..., None]
    return _blend(frame, _uint(out), intensity)


# =============================================================================
# Filters: hand-aware effects
# =============================================================================
def f_hand_spotlight(frame: np.ndarray, intensity: float, ctx: Optional[Dict[str, Any]] = None) -> np.ndarray:
    if intensity <= 0:
        return frame.copy()
    hands = _get_hands(ctx)
    if not hands:
        return frame.copy()
    h, w = frame.shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)
    for lm in hands.values():
        c = palm_center(lm)
        px, py = int(c[0] * w), int(c[1] * h)
        cv2.ellipse(mask, (px, py), (max(1, int(w * 0.18)), max(1, int(h * 0.28))), 0, 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), 5.0)
    light = 0.15 + 0.85 * mask
    out = frame.astype(np.float32) * light[..., None]
    return _blend(frame, _uint(out), intensity)


def f_magnifier_bubble(frame: np.ndarray, intensity: float, ctx: Optional[Dict[str, Any]] = None) -> np.ndarray:
    if intensity <= 0:
        return frame.copy()
    hands = _get_hands(ctx)
    if not hands:
        return frame.copy()
    h, w = frame.shape[:2]
    out = frame.copy()
    for lm in hands.values():
        p = (lm[4] + lm[8]) * 0.5
        px, py = int(p[0] * w), int(p[1] * h)
        R = int(min(h, w) * (0.07 + 0.13 * intensity))
        if R < 8:
            continue
        rx0, rx1 = max(0, px - R), min(w, px + R)
        ry0, ry1 = max(0, py - R), min(h, py + R)
        rw, rh = rx1 - rx0, ry1 - ry0
        if rw < 8 or rh < 8:
            continue
        sr = max(4, int(R * 0.55))
        sx0, sx1 = max(0, px - sr), min(w, px + sr)
        sy0, sy1 = max(0, py - sr), min(h, py + sr)
        crop = frame[sy0:sy1, sx0:sx1]
        if crop.size == 0:
            continue
        mag = cv2.resize(crop, (rw, rh), interpolation=cv2.INTER_LINEAR)
        circle = np.zeros((rh, rw), dtype=np.float32)
        cv2.ellipse(circle, (rw // 2, rh // 2), (rw // 2, rh // 2), 0, 0, 360, 1.0, -1)
        circle = cv2.GaussianBlur(circle, (0, 0), 2.0)
        alpha = (circle * intensity)[..., None]
        roi = out[ry0:ry1, rx0:rx1].astype(np.float32)
        out[ry0:ry1, rx0:rx1] = _uint(roi * (1.0 - alpha) + mag.astype(np.float32) * alpha)
    return out


def f_hand_ripple(frame: np.ndarray, intensity: float, ctx: Optional[Dict[str, Any]] = None) -> np.ndarray:
    if intensity <= 0:
        return frame.copy()
    hands = _get_hands(ctx)
    if not hands:
        return frame.copy()
    h, w = frame.shape[:2]
    overlay = frame.copy()
    max_r = int(min(h, w) * 0.25 * intensity)
    for lm in hands.values():
        tip = lm[8]
        px, py = int(tip[0] * w), int(tip[1] * h)
        for rr in range(6, max(7, max_r), 7):
            cv2.circle(overlay, (px, py), rr, (255, 255, 255), 1, cv2.LINE_AA)
    return _blend(frame, overlay, intensity * 0.75)


# =============================================================================
# Filter registry
# =============================================================================
FILTER_REGISTRY: List[Dict[str, Any]] = [
    {"name": "Original", "group": 1, "fn": f_original},
    {"name": "Warm", "group": 1, "fn": f_warm},
    {"name": "Cool", "group": 1, "fn": f_cool},
    {"name": "Black & White", "group": 1, "fn": f_blackwhite},
    {"name": "Sepia", "group": 1, "fn": f_sepia},
    {"name": "Vintage Fade", "group": 2, "fn": f_vintage},
    {"name": "Teal-Orange", "group": 2, "fn": f_teal_orange},
    {"name": "Hue Rotate", "group": 2, "fn": f_hue_rotate},
    {"name": "Saturation", "group": 2, "fn": f_saturation},
    {"name": "Contrast Curve", "group": 2, "fn": f_contrast_curve},
    {"name": "Cartoon", "group": 2, "fn": f_cartoon},
    {"name": "Pencil Sketch", "group": 2, "fn": f_pencil},
    {"name": "Neon Edges", "group": 2, "fn": f_neon},
    {"name": "Pixelate", "group": 2, "fn": f_pixelate},
    {"name": "Halftone", "group": 2, "fn": f_halftone},
    {"name": "Posterize", "group": 2, "fn": f_posterize},
    {"name": "Dream Glow", "group": 2, "fn": f_dream_glow},
    {"name": "Bulge", "group": 3, "fn": f_bulge},
    {"name": "Swirl", "group": 3, "fn": f_swirl},
    {"name": "Mirror", "group": 3, "fn": f_mirror},
    {"name": "Kaleidoscope", "group": 3, "fn": f_kaleidoscope},
    {"name": "Ripple", "group": 3, "fn": f_ripple},
    {"name": "Tilt-Shift", "group": 3, "fn": f_tilt_shift},
    {"name": "Vignette", "group": 4, "fn": f_vignette},
    {"name": "Film Grain", "group": 4, "fn": f_film_grain},
    {"name": "RGB Glitch", "group": 4, "fn": f_rgb_glitch},
    {"name": "Scanlines", "group": 4, "fn": f_scanlines},
    {"name": "Light Leak", "group": 4, "fn": f_light_leak},
    {"name": "Frame Border", "group": 4, "fn": f_frame_border},
    {"name": "Old TV", "group": 4, "fn": f_old_tv},
    {"name": "Background Blur", "group": 5, "fn": f_background_blur},
    {"name": "Background Color", "group": 5, "fn": f_background_color},
    {"name": "Background Replace", "group": 5, "fn": f_background_replace},
    {"name": "Spotlight", "group": 5, "fn": f_spotlight},
    {"name": "Hand Spotlight", "group": 5, "fn": f_hand_spotlight},
    {"name": "Magnifier Bubble", "group": 5, "fn": f_magnifier_bubble},
    {"name": "Hand Ripple", "group": 5, "fn": f_hand_ripple},
]

FILTER_BY_NAME = {item["name"]: item for item in FILTER_REGISTRY}
FILTER_NAMES = [item["name"] for item in FILTER_REGISTRY]


def apply_zoom(frame: np.ndarray, z: float) -> np.ndarray:
    if z <= 1.001:
        return frame
    h, w = frame.shape[:2]
    nh = max(1, int(h / z))
    nw = max(1, int(w / z))
    y0 = max(0, (h - nh) // 2)
    x0 = max(0, (w - nw) // 2)
    crop = frame[y0:y0 + nh, x0:x0 + nw]
    return cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)


def apply_chain(frame: np.ndarray, ctx: Dict[str, Any]) -> np.ndarray:
    names = [ss.filter] + [n for n in ss.chain if n != ss.filter]
    out = frame
    for name in names[: CONFIG["filters"]["max_chain"]]:
        item = FILTER_BY_NAME.get(name)
        if item is None:
            continue
        t0 = time.perf_counter()
        out = item["fn"](out, ss.intensity, ctx)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        prev = ss.profiler.get(name, {"ms": 0.0, "n": 0})
        n = prev["n"] + 1
        ss.profiler[name] = {"ms": (prev["ms"] * prev["n"] + dt_ms) / n, "n": n}
    if ss.zoom > 1.0:
        out = apply_zoom(out, ss.zoom)
    return out


# =============================================================================
# Synthetic video and landmarks
# =============================================================================
_CANONICAL_PALM: Optional[np.ndarray] = None


def canonical_palm() -> np.ndarray:
    """Returns a copy of a normalized 21-point open-palm template."""
    global _CANONICAL_PALM
    if _CANONICAL_PALM is None:
        _CANONICAL_PALM = np.array(
            [
                [0.00, 0.18],   # 0 wrist
                [-0.04, 0.14],  # 1
                [-0.08, 0.10],  # 2
                [-0.12, 0.06],  # 3
                [-0.16, 0.02],  # 4 thumb tip
                [-0.08, 0.02],  # 5 index MCP
                [-0.09, -0.04], # 6
                [-0.10, -0.09], # 7
                [-0.11, -0.14], # 8 index tip
                [-0.01, 0.00],  # 9 middle MCP
                [-0.01, -0.06], # 10
                [-0.01, -0.12], # 11
                [-0.01, -0.18], # 12 middle tip
                [0.06, 0.02],   # 13 ring MCP
                [0.07, -0.04],  # 14
                [0.08, -0.09],  # 15
                [0.09, -0.14],  # 16 ring tip
                [0.12, 0.06],   # 17 pinky MCP
                [0.14, 0.02],   # 18
                [0.16, -0.02],  # 19
                [0.18, -0.06],  # 20 pinky tip
            ],
            dtype=np.float32,
        )
    return _CANONICAL_PALM.copy()


_FINGER_JOINTS = [
    ("index", 5, 6, 7, 8),
    ("middle", 9, 10, 11, 12),
    ("ring", 13, 14, 15, 16),
    ("pinky", 17, 18, 19, 20),
]


def _curl_finger(lm: np.ndarray, mcp: int, pip: int, dip: int, tip: int) -> None:
    base = lm[mcp].copy()
    lm[pip] = base + np.array([0.0, 0.025], dtype=np.float32)
    lm[dip] = base + np.array([0.0, 0.040], dtype=np.float32)
    lm[tip] = base + np.array([0.0, 0.055], dtype=np.float32)


def _apply_pose(lm: np.ndarray, gesture: str) -> np.ndarray:
    out = lm.copy()

    if gesture in {"OPEN_PALM", "IDLE", "SWIPE_LEFT", "SWIPE_RIGHT", "CIRCLE_CW", "CIRCLE_CCW", "SPREAD", "CLAP"}:
        keep = ["index", "middle", "ring", "pinky"]
    elif gesture == "FIST":
        keep = []
    elif gesture == "POINTING_UP":
        keep = ["index"]
    elif gesture == "VICTORY":
        keep = ["index", "middle"]
    elif gesture == "PINCH":
        keep = ["index", "middle", "ring", "pinky"]
    elif gesture.startswith("FINGERS_"):
        try:
            n = int(gesture.split("_")[1])
        except Exception:
            n = 0
        keep = ["index", "middle", "ring", "pinky"][: max(0, min(4, n))]
    else:
        keep = ["index", "middle", "ring", "pinky"]

    for name, mcp, pip, dip, tip in _FINGER_JOINTS:
        if name not in keep:
            _curl_finger(out, mcp, pip, dip, tip)

    if gesture in {"FIST", "POINTING_UP", "VICTORY"} or gesture.startswith("FINGERS_"):
        out[3] = out[1] + np.array([0.0, 0.020], dtype=np.float32)
        out[4] = out[1] + np.array([0.0, 0.040], dtype=np.float32)
    elif gesture == "THUMB_UP":
        for name, mcp, pip, dip, tip in _FINGER_JOINTS:
            _curl_finger(out, mcp, pip, dip, tip)
        out[3] = out[0] + np.array([-0.02, -0.08], dtype=np.float32)
        out[4] = out[0] + np.array([-0.03, -0.18], dtype=np.float32)
    elif gesture == "THUMB_DOWN":
        for name, mcp, pip, dip, tip in _FINGER_JOINTS:
            _curl_finger(out, mcp, pip, dip, tip)
        out[3] = out[0] + np.array([-0.02, 0.08], dtype=np.float32)
        out[4] = out[0] + np.array([-0.03, 0.18], dtype=np.float32)

    return out


def _center_for(gesture: str, progress: float, t_s: float, hand: str) -> Tuple[float, float]:
    base_x = 0.58 if hand == "right" else 0.42
    base_y = 0.62

    if gesture == "SWIPE_RIGHT":
        return base_x - 0.30 + 0.60 * progress, base_y
    if gesture == "SWIPE_LEFT":
        return base_x + 0.30 - 0.60 * progress, base_y
    if gesture == "CIRCLE_CW":
        ang = 2.0 * math.pi * progress
        return 0.50 + 0.18 * math.cos(ang), 0.62 + 0.12 * math.sin(ang)
    if gesture == "CIRCLE_CCW":
        ang = -2.0 * math.pi * progress
        return 0.50 + 0.18 * math.cos(ang), 0.62 + 0.12 * math.sin(ang)
    if gesture == "SPREAD":
        return base_x + (0.14 * progress if hand == "right" else -0.14 * progress), base_y
    if gesture == "CLAP":
        offset = 0.25 * (1.0 - progress)
        return 0.50 + (offset if hand == "right" else -offset), base_y
    if gesture == "PINCH":
        return base_x + 0.015 * math.sin(t_s * 2.0), base_y + 0.010 * math.cos(t_s * 1.7)

    return base_x + 0.012 * math.sin(t_s * 1.4), base_y + 0.008 * math.cos(t_s * 1.1)


def make_one(gesture: str, progress: float, t_s: float, hand: str = "right") -> np.ndarray:
    lm = canonical_palm()
    lm = _apply_pose(lm, gesture)

    if gesture == "PINCH":
        sep = 0.012 + 0.058 * clamp(progress, 0.0, 1.0)
        lm[4] = lm[8] + np.array([sep, sep * 0.25], dtype=np.float32)
        lm[3] = lm[7] + np.array([sep * 0.7, sep * 0.2], dtype=np.float32)

    if hand == "left":
        lm[:, 0] *= -1.0

    cx, cy = _center_for(gesture, progress, t_s, hand)
    lm += np.array([cx, cy], dtype=np.float32)

    jitter = np.array(
        [
            [0.0015 * math.sin(t_s * 7.0 + i), 0.0015 * math.cos(t_s * 6.0 + i * 1.3)]
            for i in range(21)
        ],
        dtype=np.float32,
    )
    lm += jitter
    return np.clip(lm, 0.01, 0.99).astype(np.float32)


SCRIPT: List[Tuple[int, str]] = [
    (2000, "OPEN_PALM"),
    (700, "SWIPE_RIGHT"),
    (500, "IDLE"),
    (700, "SWIPE_LEFT"),
    (500, "IDLE"),
    (2000, "PINCH"),
    (1800, "OPEN_PALM"),
    (1200, "FIST"),
    (1200, "VICTORY"),
    (1200, "THUMB_UP"),
    (1200, "THUMB_DOWN"),
    (1800, "CIRCLE_CW"),
    (700, "SPREAD"),
    (1500, "CLAP"),
    (1500, "POINTING_UP"),
    (1500, "FINGERS_3"),
    (1800, "OPEN_PALM"),
    (1200, "VICTORY"),
]
SCRIPT_TOTAL = sum(d for d, _ in SCRIPT)


def scripted_hands(t_ms: int) -> Tuple[Dict[str, np.ndarray], str]:
    tt = int(t_ms) % SCRIPT_TOTAL
    t_s = t_ms / 1000.0
    for dur, gesture in SCRIPT:
        if tt < dur:
            p = tt / max(1, dur)
            if gesture in {"SPREAD", "CLAP"}:
                return {
                    "right": make_one(gesture, p, t_s, "right"),
                    "left": make_one(gesture, p, t_s, "left"),
                }, gesture
            return {"right": make_one(gesture, p, t_s, "right")}, gesture
        tt -= dur
    return {"right": make_one("IDLE", 0.0, t_s, "right")}, "IDLE"


MANUAL_GESTURES = [
    "Auto script",
    "IDLE",
    "OPEN_PALM",
    "FIST",
    "POINTING_UP",
    "VICTORY",
    "THUMB_UP",
    "THUMB_DOWN",
    "PINCH",
    "SWIPE_RIGHT",
    "SWIPE_LEFT",
    "CIRCLE_CW",
    "CIRCLE_CCW",
    "SPREAD",
    "CLAP",
    "FINGERS_1",
    "FINGERS_2",
    "FINGERS_3",
    "FINGERS_4",
]


def current_hands(t_ms: int) -> Dict[str, np.ndarray]:
    if ss.manual_gesture == "Auto script":
        return scripted_hands(t_ms)[0]

    gesture = ss.manual_gesture
    t_s = t_ms / 1000.0
    p = (t_ms % 2000) / 2000.0

    if gesture in {"SPREAD", "CLAP"}:
        return {
            "right": make_one(gesture, p, t_s, "right"),
            "left": make_one(gesture, p, t_s, "left"),
        }
    return {"right": make_one(gesture, p, t_s, "right")}


def _base_frame(w: int, h: int) -> np.ndarray:
    key = (w, h)
    if key not in _BASE_CACHE:
        ys = np.linspace(0, 1, h, dtype=np.float32).reshape(h, 1)
        r = 100.0 + 70.0 * ys
        g = 150.0 + 55.0 * ys
        b = 220.0 - 55.0 * ys
        sky = np.stack([np.repeat(r, w, axis=1), np.repeat(g, w, axis=1), np.repeat(b, w, axis=1)], axis=2)
        frame = sky.astype(np.uint8)

        ground_y = int(h * 0.62)
        gh = max(1, h - ground_y)
        yg = np.linspace(0, 1, gh, dtype=np.float32).reshape(gh, 1)
        gr = 40.0 + 30.0 * yg
        gg = 90.0 + 50.0 * yg
        gb = 45.0 + 20.0 * yg
        ground = np.stack([np.repeat(gr, w, axis=1), np.repeat(gg, w, axis=1), np.repeat(gb, w, axis=1)], axis=2)
        frame[ground_y:, :] = ground.astype(np.uint8)

        _BASE_CACHE[key] = frame
    return _BASE_CACHE[key].copy()


def make_synthetic_frame(t_s: float, w: int = 640, h: int = 360) -> np.ndarray:
    """Creates a realistic-looking procedural studio scene without external assets."""
    frame = _base_frame(w, h)

    # Sun / key light
    sx = int(w * (0.18 + 0.08 * math.sin(t_s * 0.35)))
    sy = int(h * 0.22)
    cv2.circle(frame, (sx, sy), max(4, int(h * 0.08)), (255, 230, 150), -1, lineType=cv2.LINE_AA)

    # Clouds
    for i in range(3):
        cx = int((w * (0.25 + 0.25 * i) + t_s * 20.0 * (i + 1)) % (w + 100)) - 50
        cy = int(h * (0.18 + 0.05 * i))
        cv2.ellipse(
            frame,
            (cx, cy),
            (max(6, int(w * 0.08)), max(3, int(h * 0.035))),
            0,
            0,
            360,
            (245, 245, 245),
            -1,
            lineType=cv2.LINE_AA,
        )

    # Perspective floor lines
    vp = (w // 2, int(h * 0.58))
    for x in range(0, w, 40):
        cv2.line(frame, vp, (x, h), (255, 255, 255), 1, cv2.LINE_AA)
    ground_y = int(h * 0.62)
    for y in range(ground_y, h, 20):
        cv2.line(frame, (0, y), (w, y), (255, 255, 255), 1, cv2.LINE_AA)

    # Simple person silhouette
    px = int(w * 0.50 + w * 0.025 * math.sin(t_s * 0.7))
    py = int(h * 0.58)
    head_r = max(4, int(h * 0.07))
    torso_w = max(8, int(w * 0.14))
    torso_h = max(12, int(h * 0.30))

    cv2.ellipse(frame, (px, py - int(h * 0.22)), (head_r, head_r), 0, 0, 360, (25, 30, 40), -1, cv2.LINE_AA)
    cv2.rectangle(
        frame,
        (px - torso_w // 2, py - int(h * 0.12)),
        (px + torso_w // 2, py + torso_h // 2),
        (35, 42, 55),
        -1,
        cv2.LINE_AA,
    )

    arm_swing = int(h * 0.05 * math.sin(t_s * 1.5))
    cv2.line(
        frame,
        (px - torso_w // 2, py - int(h * 0.05)),
        (px - torso_w, py + arm_swing),
        (35, 42, 55),
        max(2, int(h * 0.025)),
        cv2.LINE_AA,
    )
    cv2.line(
        frame,
        (px + torso_w // 2, py - int(h * 0.05)),
        (px + torso_w, py - arm_swing),
        (35, 42, 55),
        max(2, int(h * 0.025)),
        cv2.LINE_AA,
    )

    return frame


def draw_waiting_overlay(frame: np.ndarray, text: str) -> None:
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h // 2 - 34), (w, h // 2 + 34), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
    cv2.putText(
        frame,
        text,
        (12, h // 2 + 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def draw_landmarks(frame: np.ndarray, hands: Dict[str, np.ndarray]) -> None:
    h, w = frame.shape[:2]
    for lm in hands.values():
        pts = (lm * np.array([w, h], dtype=np.float32)).astype(int)
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, tuple(pts[a]), tuple(pts[b]), (0, 255, 0), 2, cv2.LINE_AA)
        for p in pts:
            cv2.circle(frame, tuple(p), 3, (255, 0, 0), -1, cv2.LINE_AA)


CHEAT_SHEET = [
    "Swipe L/R: previous/next filter",
    "Pinch: intensity",
    "Open palm 1s: arm + photo",
    "Fist: bypass original/filtered",
    "Victory: start/stop record when armed",
    "Point up: show cheat sheet",
    "Thumb up/down: save/undo",
    "Circle: hue rotate",
    "Fingers 3-4: filter group",
    "Two hands spread: zoom",
    "Clap: reset bypass/zoom",
]


def draw_hud(frame: np.ndarray, info: Dict[str, Any]) -> None:
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = 0.45 if h >= 240 else 0.35
    thickness = 1

    panel = frame.copy()
    pw = min(245, w - 12)
    ph = min(132, h - 12)
    cv2.rectangle(panel, (8, 8), (8 + pw, 8 + ph), (0, 0, 0), -1)
    cv2.addWeighted(panel, 0.42, frame, 0.58, 0, frame)

    cv2.putText(frame, f"Filter: {info.get('filter', 'Original')}", (16, 30), font, fs, (255, 255, 255), thickness, cv2.LINE_AA)

    bar_x, bar_y, bar_w, bar_h = 16, 42, max(40, pw - 30), 12
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (180, 180, 180), 1)
    fill = int(clamp(float(info.get("intensity", 0.0)), 0.0, 1.0) * bar_w)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill, bar_y + bar_h), (0, 220, 120), -1)

    cv2.putText(frame, f"Intensity: {float(info.get('intensity', 0.0)):.2f}", (16, 70), font, fs, (255, 255, 255), thickness, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {float(info.get('fps', 0.0)):.1f}", (16, 88), font, fs, (255, 255, 255), thickness, cv2.LINE_AA)
    cv2.putText(frame, f"Gesture: {info.get('last_gesture', 'none')}", (16, 106), font, fs, (255, 255, 255), thickness, cv2.LINE_AA)
    cv2.putText(frame, f"Armed: {'YES' if info.get('armed') else 'no'}", (16, 124), font, fs, (0, 255, 255), thickness, cv2.LINE_AA)

    if info.get("recording"):
        cv2.circle(frame, (w - 24, 24), 8, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(frame, "REC", (w - 62, 29), font, fs, (0, 0, 255), thickness, cv2.LINE_AA)

    if info.get("bypass"):
        cv2.putText(frame, "BYPASS ORIGINAL", (max(8, w - 190), 52), font, fs, (255, 255, 0), thickness, cv2.LINE_AA)

    progress = clamp(float(info.get("hold_progress", 0.0)), 0.0, 1.0)
    if progress > 0.01 and w > 110 and h > 110:
        center = (w - 54, 82)
        cv2.ellipse(frame, center, (28, 28), 0, -90, -90 + int(360 * progress), (0, 255, 200), 4, cv2.LINE_AA)
        cv2.putText(frame, f"{int(progress * 100)}%", (center[0] - 13, center[1] + 5), font, fs, (255, 255, 255), thickness, cv2.LINE_AA)

    if info.get("show_cheat"):
        line_h = 16 if h < 240 else 18
        max_lines = max(3, (h - 24) // line_h)
        lines = CHEAT_SHEET[:max_lines]
        box_h = len(lines) * line_h + 16
        panel2 = frame.copy()
        cv2.rectangle(panel2, (8, h - box_h - 8), (min(360, w - 8), h - 8), (0, 0, 0), -1)
        cv2.addWeighted(panel2, 0.48, frame, 0.52, 0, frame)
        for i, line in enumerate(lines):
            cv2.putText(frame, line, (16, h - box_h + 8 + i * line_h), font, max(0.28, fs - 0.08), (255, 255, 255), 1, cv2.LINE_AA)


def mirror_landmarks(lm: np.ndarray) -> np.ndarray:
    out = lm.copy()
    out[:, 0] = 1.0 - out[:, 0]
    return out


def swap_hands(hands: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for k, v in hands.items():
        if k == "right":
            nk = "left"
        elif k == "left":
            nk = "right"
        else:
            nk = k
        out[nk] = v
    return out


# =============================================================================
# Webcam / video source thread
# =============================================================================
class FrameSourceThread(threading.Thread):
    """
    Reads webcam or video file frames in a background thread.
    Keeps only the newest frame so lag does not build up.
    """

    def __init__(self, source: Any, kind: str, fps: int = 30, size: Tuple[int, int] = (640, 360)):
        super().__init__(daemon=True)
        self.source = source
        self.kind = kind
        self.fps = fps
        self.size = size
        self.stop_event = threading.Event()
        self.q: queue.Queue = queue.Queue(maxsize=1)
        self.capture_fps = 0.0
        self.error: Optional[str] = None
        self.running = False
        self.frames = 0

    def run(self) -> None:
        cap = None
        try:
            if self.kind == "webcam":
                cap = cv2.VideoCapture(int(self.source))
            elif self.kind == "file":
                cap = cv2.VideoCapture(str(self.source))
            else:
                self.error = "Unknown source kind"
                return

            if not cap.isOpened():
                self.error = f"Could not open {self.kind} source: {self.source}"
                return

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.size[0]))
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.size[1]))
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, int(CONFIG["webcam"]["buffer_size"]))
            except Exception:
                pass

            self.running = True
            last_fps_time = time.time()
            frames_in_window = 0

            while not self.stop_event.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    if self.kind == "file":
                        if self.frames == 0:
                            self.error = "Video file opened but returned no frames."
                            break
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    self.error = "Frame read failed. Camera may be busy or unavailable."
                    break

                # Keep only newest frame.
                if not self.q.empty():
                    try:
                        self.q.get_nowait()
                    except queue.Empty:
                        pass
                try:
                    self.q.put_nowait(frame)
                except queue.Full:
                    pass

                self.frames += 1
                frames_in_window += 1
                now = time.time()
                if now - last_fps_time >= 1.0:
                    self.capture_fps = frames_in_window / max(now - last_fps_time, 1e-6)
                    last_fps_time = now
                    frames_in_window = 0

                time.sleep(0.001)

        except Exception as exc:
            self.error = str(exc)
        finally:
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
            self.running = False

    def stop(self) -> None:
        self.stop_event.set()
        self.join(timeout=1.0)

    def get_frame(self) -> Optional[np.ndarray]:
        try:
            return self.q.get_nowait()
        except queue.Empty:
            return None


class Recorder:
    """Writes filtered frames to MP4 only when recording is active."""

    def __init__(self) -> None:
        self.writer: Optional[cv2.VideoWriter] = None
        self.path: Optional[Path] = None
        self.active: bool = False
        self.frames: int = 0
        self.fps: int = 30
        self.started: float = 0.0

    def start(self, fps: int = 30, size: Tuple[int, int] = (640, 360)) -> bool:
        self.path = RECORDINGS_DIR / f"rec_{_stamp()}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(str(self.path), fourcc, int(fps), (int(size[0]), int(size[1])))
        if not self.writer.isOpened():
            self.writer = None
            self.path = None
            return False
        self.active = True
        self.frames = 0
        self.fps = int(fps)
        self.started = time.time()
        return True

    def write(self, frame_rgb: np.ndarray) -> None:
        if self.active and self.writer is not None:
            self.writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
            self.frames += 1

    def stop(self) -> Optional[Path]:
        path = self.path
        if self.writer is not None:
            self.writer.release()
        self.writer = None
        self.active = False
        self.path = None
        return path


def stop_current_source() -> None:
    thread = ss.get("webcam_thread")
    if thread is not None:
        try:
            thread.stop()
        except Exception:
            pass
    ss.webcam_thread = None
    ss.webcam_fps = 0.0

    recorder = ss.get("recorder")
    if recorder is not None and getattr(recorder, "active", False):
        path = recorder.stop()
        if path is not None:
            recordings = ss.get("recordings")
            if recordings is not None:
                recordings.append(str(path))


def start_source_thread(kind: str, source: Any) -> bool:
    stop_current_source()
    try:
        thread = FrameSourceThread(
            source=source,
            kind=kind,
            fps=int(CONFIG["capture"]["fps"]),
            size=(int(ss.work_size[0]), int(ss.work_size[1])),
        )
        thread.start()
        time.sleep(0.25)

        if thread.error is not None:
            thread.stop()
            ss.webcam_thread = None
            ss.webcam_error = thread.error
            ss.webcam_fps = 0.0
            return False

        ss.webcam_thread = thread
        ss.webcam_error = None
        return True
    except Exception as exc:
        ss.webcam_thread = None
        ss.webcam_error = str(exc)
        ss.webcam_fps = 0.0
        return False


def get_source_frame(size: Tuple[int, int]) -> Optional[np.ndarray]:
    thread = ss.get("webcam_thread")
    if thread is None:
        return None

    if thread.error is not None:
        ss.webcam_error = thread.error

    if not thread.is_alive():
        return None

    frame = thread.get_frame()
    if frame is None:
        return None

    ss.webcam_fps = float(thread.capture_fps)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return cv2.resize(rgb, (int(size[0]), int(size[1])), interpolation=cv2.INTER_AREA)


# =============================================================================
# Optional MediaPipe hand tracking
# =============================================================================
def get_mp_hands():
    if not HAVE_MEDIAPIPE:
        return None

    if ss.get("mp_hands", None) is None:
        try:
            ss.mp_hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                model_complexity=0,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        except Exception:
            ss.mp_hands = False

    if ss.mp_hands is False:
        return None
    return ss.mp_hands


def detect_mediapipe_hands(rgb_frame: np.ndarray) -> Optional[Dict[str, np.ndarray]]:
    hands_obj = get_mp_hands()
    if hands_obj is None:
        return None

    try:
        results = hands_obj.process(rgb_frame)
        if not hasattr(results, "multi_hand_landmarks"):
            return None
    except Exception:
        return None

    out: Dict[str, np.ndarray] = {}
    if results.multi_hand_landmarks is None:
        return out

    handedness = getattr(results, "multi_handedness", None)
    if handedness is None:
        handedness = []

    for i, hand_lms in enumerate(results.multi_hand_landmarks):
        lm = np.array([[p.x, p.y] for p in hand_lms.landmark], dtype=np.float32)
        label = "right"
        if i < len(handedness):
            try:
                label = str(handedness[i].classification[0].label).lower()
            except Exception:
                label = "right"

        # Avoid duplicate keys if MediaPipe reports two same labels.
        if label in out:
            label = f"{label}_{i}"
        out[label] = lm

    return out


# =============================================================================
# Persistence helpers
# =============================================================================
def log_event(ev: GestureEvent) -> None:
    path = LOGS_DIR / "events.jsonl"
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(ev.to_json()) + "\n")


def save_capture(frame_rgb: np.ndarray, tag: str = "cap") -> Path:
    path = CAPTURES_DIR / f"{tag}_{_stamp()}.png"
    cv2.imwrite(str(path), cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
    return path


# =============================================================================
# Custom gesture trainer
# =============================================================================
def landmark_vector(lm: Optional[np.ndarray]) -> np.ndarray:
    if lm is None:
        return np.zeros(63, dtype=np.float32)
    lm = np.asarray(lm, dtype=np.float32)
    c = palm_center(lm)
    w = palm_width(lm)
    norm = (lm - c) / max(w, 1e-6)
    vec = np.zeros((21, 3), dtype=np.float32)
    vec[:, :2] = norm
    return vec.reshape(-1)


def augment_vector(vec: np.ndarray, n: int = 6) -> np.ndarray:
    arr = vec.reshape(21, 3).copy()
    out = [vec.copy()]
    rng = np.random.default_rng()
    for _ in range(max(0, n)):
        angle = math.radians(float(rng.uniform(-15, 15)))
        scale = float(rng.uniform(0.90, 1.10))
        x = arr[:, 0] * scale
        y = arr[:, 1] * scale
        rx = x * math.cos(angle) - y * math.sin(angle) + rng.normal(0.0, 0.01, 21)
        ry = x * math.sin(angle) + y * math.cos(angle) + rng.normal(0.0, 0.01, 21)
        aug = arr.copy()
        aug[:, 0] = rx
        aug[:, 1] = ry
        out.append(aug.reshape(-1).astype(np.float32))
    return np.array(out, dtype=np.float32)


class CustomGestureTrainer:
    """Tiny local kNN-style trainer for custom static hand poses."""

    def __init__(self) -> None:
        self.samples: Dict[str, np.ndarray] = {}

    def add(self, label: str, vec: np.ndarray, augment_count: int = 6, max_per_label: int = 120) -> None:
        label = (label or "CUSTOM").strip() or "CUSTOM"
        new = augment_vector(vec, augment_count)
        if label in self.samples:
            merged = np.vstack([self.samples[label], new])
            self.samples[label] = merged[-max_per_label:]
        else:
            self.samples[label] = new[-max_per_label:]

    def remove_label(self, label: str) -> None:
        self.samples.pop(label, None)

    def clear(self) -> None:
        self.samples.clear()

    def predict(self, vec: np.ndarray, reject_distance: float = 0.55) -> Tuple[str, float]:
        if not self.samples:
            return "NONE", 0.0
        best_label = "NONE"
        best_dist = 1e9
        for label, mat in self.samples.items():
            d = float(np.linalg.norm(mat - vec[None, :], axis=1).min())
            if d < best_dist:
                best_dist = d
                best_label = label
        if best_dist > reject_distance:
            return "NONE", clamp(1.0 - best_dist / max(reject_distance, 1e-6), 0.0, 1.0)
        return best_label, clamp(1.0 - best_dist / max(reject_distance, 1e-6), 0.0, 1.0)

    def to_dict(self) -> Dict[str, Any]:
        return {k: v.tolist() for k, v in self.samples.items()}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CustomGestureTrainer":
        obj = cls()
        for k, v in data.items():
            obj.samples[k] = np.array(v, dtype=np.float32)
        return obj

    def summary(self) -> Dict[str, int]:
        return {k: int(v.shape[0]) for k, v in self.samples.items()}


def save_custom_model(trainer: CustomGestureTrainer) -> None:
    try:
        path = SETTINGS_DIR / "custom_gestures.json"
        path.write_text(json.dumps(trainer.to_dict(), indent=2), encoding="utf-8")
    except Exception:
        pass


def load_custom_model() -> CustomGestureTrainer:
    path = SETTINGS_DIR / "custom_gestures.json"
    if path.exists():
        try:
            return CustomGestureTrainer.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            return CustomGestureTrainer()
    return CustomGestureTrainer()


# =============================================================================
# Replay helpers
# =============================================================================
def save_replay(records: List[Dict[str, Any]]) -> Path:
    path = LOGS_DIR / f"replay_{_stamp()}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(_jsonable(r)) + "\n")
    return path


def load_latest_replay() -> List[Dict[str, Any]]:
    files = sorted(LOGS_DIR.glob("replay_*.jsonl"))
    if not files:
        return []
    path = files[-1]
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def replay_records(records: List[Dict[str, Any]]) -> List[GestureEvent]:
    eng = GestureEngine(CONFIG)
    events: List[GestureEvent] = []
    for rec in records:
        hands_data = rec.get("hands") or {}
        hands = {k: np.array(v, dtype=np.float32) for k, v in hands_data.items()}
        t_ms = int(rec.get("t_ms", 0))
        events.extend(eng.process(hands, t_ms))
    return events


# =============================================================================
# Presets
# =============================================================================
def save_presets() -> None:
    try:
        (SETTINGS_DIR / "presets.json").write_text(json.dumps(ss.presets, indent=2), encoding="utf-8")
    except Exception:
        pass


def save_preset(name: str) -> None:
    name = (name or "preset").strip() or "preset"
    ss.presets[name] = {
        "filter": ss.filter,
        "intensity": float(ss.intensity),
        "chain": list(ss.chain),
        "zoom": float(ss.zoom),
        "bypass": bool(ss.bypass),
    }
    save_presets()
    ss.last_gesture = f"Preset saved: {name}"


def load_preset(name: str) -> None:
    p = ss.presets.get(name)
    if p is None:
        ss.last_gesture = "Preset not found"
        return
    ss.filter = p.get("filter", ss.filter) if p.get("filter") in FILTER_BY_NAME else ss.filter
    ss.intensity = clamp(float(p.get("intensity", ss.intensity)), 0.0, 1.0)
    ss.chain = [n for n in (p.get("chain") or []) if n in FILTER_BY_NAME][:3]
    ss.zoom = clamp(float(p.get("zoom", ss.zoom)), 1.0, 2.0)
    ss.bypass = bool(p.get("bypass", ss.bypass))
    ss.last_gesture = f"Preset loaded: {name}"


# =============================================================================
# App state helpers
# =============================================================================
def init_session() -> None:
    ensure_dirs()

    ss.setdefault("t0", time.time())
    ss.setdefault("time_ms", 0)
    ss.setdefault("booted", False)
    ss.setdefault("run_frames", 0)
    ss.setdefault("last_render_time", 0.0)
    ss.setdefault("fps", 0.0)
    ss.setdefault("frame_ms", 0.0)
    ss.setdefault("latency_samples", [])

    ss.setdefault("work_size", (CONFIG["capture"]["width"], CONFIG["capture"]["height"]))
    ss.setdefault("left_handed", False)
    ss.setdefault("show_cheat", False)
    ss.setdefault("manual_gesture", "Auto script")
    ss.setdefault("auto_quality", bool(CONFIG["performance"]["auto_quality"]))

    ss.setdefault("input_source", SOURCE_SIM)
    ss.setdefault("previous_input_source", SOURCE_SIM)
    ss.setdefault("camera_index", int(CONFIG["webcam"]["default_index"]))
    ss.setdefault("video_path", "")
    ss.setdefault("webcam_thread", None)
    ss.setdefault("webcam_error", None)
    ss.setdefault("webcam_fps", 0.0)
    ss.setdefault("use_mediapipe_tracking", True)
    ss.setdefault("mp_hands", None)

    ss.setdefault("filter", "Warm")
    ss.setdefault("intensity", 0.65)
    ss.setdefault("chain", [])
    ss.setdefault("zoom", 1.0)
    ss.setdefault("bypass", False)

    ss.setdefault("armed_until", 0)
    ss.setdefault("last_gesture", "none")
    ss.setdefault("events", [])
    ss.setdefault("captures", [])
    ss.setdefault("recordings", [])
    ss.setdefault("undo_stack", [])
    ss.setdefault("profiler", {})

    ss.setdefault("engine", GestureEngine(CONFIG))
    ss.setdefault("recorder", Recorder())

    ss.setdefault("current_frame", None)
    ss.setdefault("current_clean", None)
    ss.setdefault("current_info", {})

    ss.setdefault("filter_previews", {})
    ss.setdefault("preview_key", None)

    ss.setdefault("demo_gif", None)
    ss.setdefault("demo_mp4", None)
    ss.setdefault("test_results", [])
    ss.setdefault("script_eval", None)

    ss.setdefault("custom_trainer", load_custom_model())

    ss.setdefault("replay_recording", False)
    ss.setdefault("replay_buffer", [])
    ss.setdefault("replay_result", None)

    preset_path = SETTINGS_DIR / "presets.json"
    if "presets" not in ss:
        try:
            if preset_path.exists():
                ss.presets = json.loads(preset_path.read_text(encoding="utf-8"))
            else:
                ss.presets = {}
        except Exception:
            ss.presets = {}

    # Widget state defaults. These keys are synchronized with app state in main().
    ss.setdefault("input_source_select", SOURCE_SIM)
    ss.setdefault("camera_index_input", int(CONFIG["webcam"]["default_index"]))
    ss.setdefault("video_path_input", "")
    ss.setdefault("use_mp_webcam_checkbox", True)
    ss.setdefault("use_mp_file_checkbox", True)
    ss.setdefault("resolution_select", "640x360")
    ss.setdefault("mirror_checkbox", True)
    ss.setdefault("lefthanded_checkbox", False)
    ss.setdefault("cheat_checkbox", False)
    ss.setdefault("autoquality_checkbox", True)
    ss.setdefault("autoplay_slider", int(CONFIG["ui"]["autoplay_seconds"]))
    ss.setdefault("gesture_source_select", "Auto script")
    ss.setdefault("filter_select", "Warm")
    ss.setdefault("intensity_slider", 0.65)
    ss.setdefault("chain_multiselect", [])
    ss.setdefault("zoom_slider", 1.0)
    ss.setdefault("bypass_checkbox", False)
    ss.setdefault("swipe_dx_slider", float(CONFIG["gestures"]["swipe_dx_min"]))
    ss.setdefault("pinch_start_slider", float(CONFIG["gestures"]["pinch_start"]))
    ss.setdefault("pinch_end_slider", float(CONFIG["gestures"]["pinch_end"]))
    ss.setdefault("hold_ms_slider", int(CONFIG["gestures"]["hold_ms"]))
    ss.setdefault("arming_ttl_slider", int(CONFIG["gestures"]["arming_ttl_ms"]))
    ss.setdefault("preset_name_input", "default")
    ss.setdefault("custom_label_name", "MyGesture")
    ss.setdefault("replay_record_checkbox", False)
    ss.setdefault("delete_confirm_checkbox", False)


def push_undo() -> None:
    ss.undo_stack.append(
        {
            "filter": ss.filter,
            "intensity": ss.intensity,
            "bypass": ss.bypass,
            "zoom": ss.zoom,
        }
    )
    if len(ss.undo_stack) > 30:
        ss.undo_stack.pop(0)


def undo() -> None:
    if not ss.undo_stack:
        ss.last_gesture = "Undo: nothing to undo"
        return
    state = ss.undo_stack.pop()
    ss.filter = state["filter"]
    ss.intensity = state["intensity"]
    ss.bypass = state["bypass"]
    ss.zoom = state["zoom"]
    ss.last_gesture = "Undo applied"


def set_filter_offset(delta: int) -> None:
    try:
        idx = FILTER_NAMES.index(ss.filter)
    except ValueError:
        idx = 0
    ss.filter = FILTER_NAMES[(idx + delta) % len(FILTER_NAMES)]


def set_group(group: int) -> None:
    names = [item["name"] for item in FILTER_REGISTRY if item["group"] == group]
    if names:
        ss.filter = names[0]


def toggle_recording(frame: Optional[np.ndarray] = None) -> None:
    rec: Recorder = ss.recorder
    if rec.active:
        path = rec.stop()
        if path is not None:
            ss.recordings.append(str(path))
        ss.last_gesture = "Recording stopped"
    else:
        ok = rec.start(fps=CONFIG["capture"]["fps"], size=ss.work_size)
        if ok:
            if frame is not None:
                rec.write(frame)
            ss.last_gesture = "Recording started"
        else:
            ss.last_gesture = "Recorder could not open MP4 writer"


def handle_events(events: List[GestureEvent], clean_frame: Optional[np.ndarray] = None, t_ms: Optional[int] = None) -> None:
    if t_ms is None:
        t_ms = ss.time_ms

    for ev in events:
        ss.events.append(ev.to_json())
        log_event(ev)
        typ = ev.type

        if typ == "SWIPE_LEFT":
            push_undo()
            set_filter_offset(-1)
            ss.last_gesture = "Swipe left: previous filter"

        elif typ == "SWIPE_RIGHT":
            push_undo()
            set_filter_offset(1)
            ss.last_gesture = "Swipe right: next filter"

        elif typ == "PINCH":
            ss.intensity = clamp(float(ev.params.get("intensity", ss.intensity)), 0.0, 1.0)
            ss.last_gesture = f"Pinch intensity {ss.intensity:.2f}"

        elif typ == "OPEN_PALM":
            ss.armed_until = int(ev.t_ms) + int(CONFIG["gestures"]["arming_ttl_ms"])
            if clean_frame is not None:
                path = save_capture(clean_frame, "photo")
                ss.captures.append(str(path))
            ss.last_gesture = "Open palm: armed + photo"

        elif typ == "FIST":
            push_undo()
            ss.bypass = not ss.bypass
            ss.last_gesture = "Fist: bypass toggled"

        elif typ == "VICTORY":
            if int(ev.t_ms) <= int(ss.armed_until):
                toggle_recording(clean_frame)
            else:
                ss.last_gesture = "Victory ignored: open-palm arm first"

        elif typ == "THUMB_UP":
            if clean_frame is not None:
                path = save_capture(clean_frame, "confirmed")
                ss.captures.append(str(path))
            save_preset(f"thumb_up_{_stamp()}")
            ss.last_gesture = "Thumb up: confirmed/saved"

        elif typ == "THUMB_DOWN":
            undo()

        elif typ == "POINTING_UP":
            ss.show_cheat = True
            ss.last_gesture = "Pointing up: cheat sheet opened"

        elif typ in {"CIRCLE_CW", "CIRCLE_CCW"}:
            push_undo()
            ss.filter = "Hue Rotate"
            ss.intensity = 0.70
            ss.last_gesture = f"{typ}: hue rotate"

        elif typ.startswith("FINGERS_"):
            try:
                n = int(typ.split("_")[1])
            except Exception:
                n = 0
            if 1 <= n <= 5:
                push_undo()
                set_group(n)
                ss.last_gesture = f"Fingers {n}: group {n}"
            else:
                ss.last_gesture = "Unknown finger count"

        elif typ == "SPREAD":
            push_undo()
            ss.zoom = clamp(float(ev.params.get("zoom", 1.2)), 1.0, 2.0)
            ss.last_gesture = f"Two-hand spread: zoom {ss.zoom:.2f}"

        elif typ == "CLAP":
            push_undo()
            ss.bypass = False
            ss.zoom = 1.0
            ss.intensity = 0.50
            ss.last_gesture = "Clap: reset"


def compose_frame(t_ms: int) -> Tuple[np.ndarray, Dict[str, Any]]:
    t0 = time.perf_counter()

    # Auto quality reduction if FPS is too low.
    if ss.auto_quality and ss.fps > 0 and ss.fps < 18 and ss.work_size != (320, 180):
        ss.work_size = (320, 180)
        stop_current_source()
        ss.last_gesture = "Auto quality reduced; source/recording stopped"

    w, h = ss.work_size

    source_frame = None
    if ss.input_source in (SOURCE_WEB, SOURCE_FILE):
        source_frame = get_source_frame((w, h))

    if source_frame is not None:
        clean = source_frame
    else:
        clean = make_synthetic_frame(t_ms / 1000.0, w, h)
        if ss.input_source != SOURCE_SIM:
            draw_waiting_overlay(clean, "Waiting for webcam/video... simulated fallback")

    # Choose landmarks.
    hands = current_hands(t_ms)
    mp_active = False

    if source_frame is not None and HAVE_MEDIAPIPE and bool(ss.use_mediapipe_tracking):
        detected = detect_mediapipe_hands(source_frame)
        if detected is not None:
            hands = detected
            mp_active = True

    if ss.left_handed:
        hands = swap_hands(hands)

    events = ss.engine.process(hands, t_ms)

    if ss.replay_recording and len(ss.replay_buffer) < 900:
        ss.replay_buffer.append(
            {
                "t_ms": int(t_ms),
                "hands": {k: v.tolist() for k, v in hands.items()},
            }
        )

    ctx = {"hands": hands, "t_ms": t_ms}

    if ss.bypass:
        filtered = clean.copy()
    else:
        filtered = apply_chain(clean, ctx)

    if CONFIG["capture"]["mirror"]:
        filtered = cv2.flip(filtered, 1)
        clean = cv2.flip(clean, 1)
        hands = {k: mirror_landmarks(v) for k, v in hands.items()}

    handle_events(events, filtered, t_ms)

    if ss.recorder.active:
        ss.recorder.write(filtered)

    annotated = filtered.copy()
    draw_landmarks(annotated, hands)

    fb = ss.engine.feedback()
    now = time.time()
    dt = now - ss.last_render_time if ss.last_render_time else 0.0
    ss.last_render_time = now
    if 0.001 < dt < 1.0:
        inst_fps = 1.0 / dt
        ss.fps = inst_fps if ss.fps == 0 else 0.8 * ss.fps + 0.2 * inst_fps

    info = {
        "filter": ss.filter,
        "intensity": ss.intensity,
        "fps": ss.fps,
        "last_gesture": ss.last_gesture,
        "armed": t_ms <= ss.armed_until,
        "recording": ss.recorder.active,
        "bypass": ss.bypass,
        "show_cheat": ss.show_cheat,
        "hold_progress": fb.get("hold_progress", 0.0),
        "last_label": fb.get("last_label", "NONE"),
        "source": ss.input_source,
        "tracking": "MediaPipe" if mp_active else "Simulated",
        "capture_fps": float(ss.webcam_fps),
    }

    draw_hud(annotated, info)

    ss.current_clean = filtered
    ss.current_frame = annotated
    ss.current_info = info

    frame_ms = (time.perf_counter() - t0) * 1000.0
    ss.frame_ms = frame_ms if ss.frame_ms == 0 else 0.8 * ss.frame_ms + 0.2 * frame_ms
    ss.latency_samples.append(frame_ms)
    if len(ss.latency_samples) > int(CONFIG["performance"]["latency_samples"]):
        ss.latency_samples.pop(0)

    return annotated, info


def status_html(info: Dict[str, Any]) -> str:
    armed = '<span class="gcs-pill">ARMED</span>' if info.get("armed") else '<span class="gcs-pill warn">SAFE</span>'
    rec = '<span class="gcs-pill bad">REC</span>' if info.get("recording") else '<span class="gcs-pill">IDLE</span>'
    bypass = '<span class="gcs-pill warn">BYPASS</span>' if info.get("bypass") else '<span class="gcs-pill">FILTERED</span>'

    return f"""
    <div class="gcs-card">
      <div class="gcs-title">Live status</div>
      <div class="gcs-row"><span>Source</span><strong>{info.get('source', SOURCE_SIM)}</strong></div>
      <div class="gcs-row"><span>Tracking</span><strong>{info.get('tracking', 'Simulated')}</strong></div>
      <div class="gcs-row"><span>Filter</span><strong>{info.get('filter', 'Original')}</strong></div>
      <div class="gcs-row"><span>Intensity</span><strong>{float(info.get('intensity', 0.0)):.2f}</strong></div>
      <div class="gcs-row"><span>Display FPS</span><strong>{float(info.get('fps', 0.0)):.1f}</strong></div>
      <div class="gcs-row"><span>Capture FPS</span><strong>{float(info.get('capture_fps', 0.0)):.1f}</strong></div>
      <div class="gcs-row"><span>Frame time</span><strong>{float(ss.frame_ms):.1f} ms</strong></div>
      <div class="gcs-row"><span>Last gesture</span><strong>{info.get('last_gesture', 'none')}</strong></div>
      <div class="gcs-row"><span>Hold progress</span><strong>{float(info.get('hold_progress', 0.0)) * 100:.0f}%</strong></div>
      <div class="gcs-row"><span>State</span><strong>{armed} {rec} {bypass}</strong></div>
      <div class="gcs-mini" style="margin-top:0.65rem;">
        Local-only processing. No upload. Delete generated data anytime from the Gallery tab.
      </div>
    </div>
    """


# =============================================================================
# Filter previews
# =============================================================================
def build_filter_previews(intensity: float) -> Dict[str, np.ndarray]:
    pw = int(CONFIG["ui"]["preview_width"])
    ph = int(CONFIG["ui"]["preview_height"])
    base = make_synthetic_frame(0.0, pw, ph)
    hands = scripted_hands(1000)[0]
    ctx = {"hands": hands, "t_ms": 1000}
    previews: Dict[str, np.ndarray] = {}
    for item in FILTER_REGISTRY:
        out = item["fn"](base, intensity, ctx)
        cv2.putText(
            out,
            item["name"][:18],
            (3, ph - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        previews[item["name"]] = out
    return previews


# =============================================================================
# Demo reel generation
# =============================================================================
def generate_demo_reel(force: bool = False) -> None:
    gif_path = DEMO_DIR / "gesturecam_demo.gif"
    mp4_path = DEMO_DIR / "gesturecam_demo.mp4"

    if not force and "demo_gif" in ss and ss.demo_gif is not None and Path(ss.demo_gif).exists():
        return

    fps = 12
    frames = 42
    size = (320, 180)
    w, h = size

    pil_frames: List[Image.Image] = []
    bgr_frames: List[np.ndarray] = []

    cycle = [
        "Warm",
        "Cartoon",
        "Pixelate",
        "Vignette",
        "Hue Rotate",
        "Background Blur",
        "Magnifier Bubble",
        "Dream Glow",
        "Old TV",
    ]

    for i in range(frames):
        t_ms = i * int(1000 / fps)
        hands, gesture = scripted_hands(t_ms)
        base = make_synthetic_frame(t_ms / 1000.0, w, h)
        fname = cycle[(i // 5) % len(cycle)]
        fn = FILTER_BY_NAME[fname]["fn"]
        out = fn(base, 0.75, {"hands": hands, "t_ms": t_ms})
        draw_landmarks(out, hands)
        draw_hud(
            out,
            {
                "filter": fname,
                "intensity": 0.75,
                "fps": fps,
                "last_gesture": gesture,
                "armed": i > 8,
                "hold_progress": (i % 12) / 12.0,
                "recording": 18 <= i <= 28,
                "bypass": False,
                "show_cheat": False,
                "source": "Demo reel",
                "tracking": "Simulated",
                "capture_fps": fps,
            },
        )
        pil_frames.append(Image.fromarray(out))
        bgr_frames.append(cv2.cvtColor(out, cv2.COLOR_RGB2BGR))

    pil_frames[0].save(
        gif_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=int(1000 / fps),
        loop=0,
        optimize=False,
    )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(mp4_path), fourcc, fps, size)
    if writer.isOpened():
        for bgr in bgr_frames:
            writer.write(bgr)
        writer.release()
        ss.demo_mp4 = str(mp4_path)
    else:
        ss.demo_mp4 = None

    ss.demo_gif = str(gif_path)


# =============================================================================
# Self-tests and evaluation
# =============================================================================
def run_self_tests() -> List[Tuple[str, bool, str]]:
    results: List[Tuple[str, bool, str]] = []

    def check(name: str, fn: Any) -> None:
        try:
            fn()
            results.append((name, True, "PASS"))
        except Exception as exc:
            results.append((name, False, str(exc)))

    def test_geometry() -> None:
        lm = make_one("OPEN_PALM", 0.0, 0.0, "right")
        assert pinch_ratio(lm) > 0.0
        flags = extended_finger_flags(lm)
        assert sum(flags[k] for k in ("index", "middle", "ring", "pinky")) >= 3
        fist = make_one("FIST", 0.0, 0.0, "right")
        assert classify_static(fist)[0] == "FIST"

    def test_swipe_right() -> None:
        eng = GestureEngine(CONFIG)
        for i, t in enumerate(range(0, 400, 33)):
            eng.history.setdefault("right", []).append(
                (t, np.array([0.20 + i * 0.025, 0.50], dtype=np.float32))
            )
        ev = eng.detect_swipe("right", 399)
        assert ev is not None and ev.type == "SWIPE_RIGHT"

    def test_diagonal_rejected() -> None:
        eng = GestureEngine(CONFIG)
        for i, t in enumerate(range(0, 400, 33)):
            eng.history.setdefault("right", []).append(
                (t, np.array([0.20 + i * 0.025, 0.30 + i * 0.018], dtype=np.float32))
            )
        ev = eng.detect_swipe("right", 399)
        assert ev is None

    def test_hold_open_palm() -> None:
        eng = GestureEngine(CONFIG)
        lm = make_one("OPEN_PALM", 0.0, 0.0, "right")
        got = []
        for t in range(0, 1400, 50):
            got.extend(eng.process({"right": lm}, t))
        assert any(e.type == "OPEN_PALM" for e in got)

    def test_one_euro_smooths_jitter() -> None:
        rng = np.random.default_rng(1)
        noisy = 1.0 + rng.normal(0.0, 0.05, 120)
        f = OneEuroFilter(min_cutoff=1.0, beta=0.02)
        smoothed = np.array([f(float(v), i * 0.033) for i, v in enumerate(noisy)])
        assert np.var(smoothed[30:]) < 0.5 * np.var(noisy[30:])

    def test_filters_identity_and_finite() -> None:
        frame = np.random.default_rng(0).integers(0, 256, (64, 64, 3), dtype=np.uint8)
        hands = {"right": make_one("OPEN_PALM", 0.0, 0.0, "right")}
        ctx = {"hands": hands, "t_ms": 0}
        for item in FILTER_REGISTRY:
            out0 = item["fn"](frame, 0.0, ctx)
            assert out0.shape == frame.shape
            assert out0.dtype == np.uint8
            assert np.allclose(out0, frame)
            out1 = item["fn"](frame, 0.5, ctx)
            assert np.isfinite(out1.astype(np.float32)).all()

    def test_teal_orange_broadcast() -> None:
        frame = np.random.default_rng(2).integers(0, 256, (90, 160, 3), dtype=np.uint8)
        out = f_teal_orange(frame, 0.5)
        assert out.shape == frame.shape
        assert out.dtype == np.uint8

    def test_replay_determinism() -> None:
        records = [
            {"t_ms": i * 33, "hands": {"right": make_one("OPEN_PALM", 0.0, i * 0.033, "right").tolist()}}
            for i in range(40)
        ]
        a = replay_records(records)
        b = replay_records(records)
        assert [x.type for x in a] == [x.type for x in b]

    def test_custom_trainer() -> None:
        trainer = CustomGestureTrainer()
        v1 = landmark_vector(make_one("FIST", 0.0, 0.0, "right"))
        v2 = landmark_vector(make_one("OPEN_PALM", 0.0, 0.0, "right"))
        trainer.add("FIST", v1, augment_count=4)
        trainer.add("PALM", v2, augment_count=4)
        pred, _ = trainer.predict(v1, reject_distance=0.90)
        assert pred == "FIST"

    def test_event_json() -> None:
        ev = GestureEvent("TEST", "right", 10, 0.5, {"x": np.float32(1.2), "n": np.int64(3)})
        j = ev.to_json()
        json.dumps(j)
        assert j["params"]["x"] == 1.2
        assert j["params"]["n"] == 3

    check("Geometry: pinch and finger flags", test_geometry)
    check("Swipe detector: right swipe", test_swipe_right)
    check("Swipe detector: diagonal rejected", test_diagonal_rejected)
    check("Hold timer: open palm", test_hold_open_palm)
    check("One Euro filter: jitter reduction", test_one_euro_smooths_jitter)
    check("Filters: identity at 0 and finite at 0.5", test_filters_identity_and_finite)
    check("Teal-Orange broadcast fix", test_teal_orange_broadcast)
    check("Replay determinism", test_replay_determinism)
    check("Custom trainer basic prediction", test_custom_trainer)
    check("GestureEvent JSON serialization", test_event_json)

    return results


def run_scripted_evaluation() -> Dict[str, Any]:
    eng = GestureEngine(CONFIG)
    found = set()
    expected = {g for _, g in SCRIPT if g not in {"IDLE"}}
    for t in range(0, SCRIPT_TOTAL, 33):
        hands = scripted_hands(t)[0]
        for ev in eng.process(hands, t):
            found.add(ev.type)
    hit = found & expected
    recall = len(hit) / max(1, len(expected))
    return {
        "expected": sorted(expected),
        "found": sorted(found),
        "missing": sorted(expected - found),
        "extra": sorted(found - expected),
        "recall": float(recall),
    }


# =============================================================================
# Callbacks
# =============================================================================
def cb_start_webcam() -> None:
    ok = start_source_thread("webcam", int(ss.camera_index))
    if ok:
        ss.last_gesture = "Webcam started"
    else:
        ss.last_gesture = "Webcam failed to start"


def cb_start_video() -> None:
    raw = str(ss.video_path).strip()
    if not raw:
        ss.webcam_error = "Video file path is empty"
        ss.last_gesture = "Video file path is empty"
        return

    path = Path(raw).expanduser()
    if not path.exists():
        ss.webcam_error = f"Video file not found: {path}"
        ss.last_gesture = "Video file not found"
        return

    ok = start_source_thread("file", str(path))
    if ok:
        ss.last_gesture = "Video source started"
    else:
        ss.last_gesture = "Video source failed to start"


def cb_stop_source() -> None:
    stop_current_source()
    ss.last_gesture = "Source stopped"


def ensure_source_running() -> None:
    if ss.input_source == SOURCE_WEB:
        thread = ss.get("webcam_thread")
        if thread is None or not thread.is_alive():
            cb_start_webcam()
    elif ss.input_source == SOURCE_FILE:
        thread = ss.get("webcam_thread")
        if thread is None or not thread.is_alive():
            cb_start_video()


def cb_play(seconds: float = 2.0) -> None:
    ensure_source_running()
    ss.run_frames = int(seconds * CONFIG["capture"]["fps"])


def cb_step() -> None:
    ss.time_ms += 33


def cb_reset_timeline() -> None:
    ss.time_ms = 0
    ss.engine = GestureEngine(CONFIG)
    ss.replay_buffer = []
    ss.last_gesture = "Timeline reset"


def cb_capture() -> None:
    if ss.current_clean is not None:
        path = save_capture(ss.current_clean, "manual")
        ss.captures.append(str(path))
        ss.last_gesture = "Manual capture saved"


def cb_toggle_record() -> None:
    toggle_recording(ss.current_clean)


def cb_delete_all() -> None:
    stop_current_source()

    for d in [CAPTURES_DIR, RECORDINGS_DIR, LOGS_DIR, DEMO_DIR, SETTINGS_DIR]:
        for f in d.glob("*"):
            try:
                if f.is_file():
                    f.unlink(missing_ok=True)
            except Exception:
                pass

    ss.events.clear()
    ss.captures.clear()
    ss.recordings.clear()
    ss.presets = {}
    ss.engine = GestureEngine(CONFIG)
    ss.custom_trainer = CustomGestureTrainer()
    ss.demo_gif = None
    ss.demo_mp4 = None
    ss.filter_previews = {}
    ss.preview_key = None
    ss.test_results = []
    ss.script_eval = None
    ss.replay_buffer = []
    ss.replay_result = None
    ss.replay_recording = False
    ss.last_gesture = "All user data deleted"

    ss["replay_record_checkbox"] = False
    ss["delete_confirm_checkbox"] = False


def cb_force(typ: str, params: Optional[Dict[str, Any]] = None) -> None:
    ev = GestureEvent(typ, "right", ss.time_ms, 0.90, params or {})
    handle_events([ev], ss.current_clean, ss.time_ms)


def cb_set_filter(name: str) -> None:
    if name in FILTER_BY_NAME:
        push_undo()
        ss.filter = name
        ss.last_gesture = f"Filter set: {name}"


def cb_arm_now() -> None:
    ss.armed_until = int(ss.time_ms) + int(CONFIG["gestures"]["arming_ttl_ms"])
    ss.last_gesture = "Armed manually"


def cb_save_preset(name: str) -> None:
    save_preset(name)


def cb_load_preset(name: str) -> None:
    load_preset(name)


def cb_delete_preset(name: str) -> None:
    if name in ss.presets:
        del ss.presets[name]
        save_presets()
        ss.last_gesture = f"Preset deleted: {name}"


def cb_run_tests() -> None:
    ss.test_results = run_self_tests()


def cb_run_script_eval() -> None:
    ss.script_eval = run_scripted_evaluation()


def cb_record_custom_sample() -> None:
    label = ss.get("custom_label_name", "MyGesture")
    lm = first_available(ss.engine.filtered, "right", "left")
    if lm is None:
        hands = current_hands(ss.time_ms)
        lm = first_available(hands, "right", "left")
    if lm is None:
        ss.last_gesture = "No hand landmarks available"
        return
    vec = landmark_vector(lm)
    ss.custom_trainer.add(
        label,
        vec,
        augment_count=int(CONFIG["custom"]["augment_count"]),
        max_per_label=int(CONFIG["custom"]["max_per_label"]),
    )
    save_custom_model(ss.custom_trainer)
    ss.last_gesture = f"Recorded custom sample: {label}"


def cb_clear_custom_label() -> None:
    label = ss.get("custom_label_name", "MyGesture")
    ss.custom_trainer.remove_label(label)
    save_custom_model(ss.custom_trainer)
    ss.last_gesture = f"Deleted custom label: {label}"


def cb_clear_all_custom() -> None:
    ss.custom_trainer.clear()
    save_custom_model(ss.custom_trainer)
    ss.last_gesture = "All custom gestures cleared"


def cb_save_replay() -> None:
    if not ss.replay_buffer:
        ss.last_gesture = "No replay buffer to save"
        return
    path = save_replay(ss.replay_buffer)
    ss.last_gesture = f"Replay saved: {path.name}"


def cb_replay_latest() -> None:
    records = load_latest_replay()
    if not records:
        ss.last_gesture = "No replay file found"
        return
    events = replay_records(records)
    ss.replay_result = {
        "frames": len(records),
        "events": [e.to_json() for e in events],
    }
    ss.last_gesture = f"Replayed {len(records)} frames"


def cb_generate_demo() -> None:
    generate_demo_reel(force=True)
    ss.last_gesture = "Demo reel generated"


# =============================================================================
# Main UI
# =============================================================================
def main() -> None:
    inject_css()
    init_session()
    hero()

    with st.sidebar:
        st.header("⚙️ Studio settings")

        # -------------------------------------------------------------------------
        # Sync widget keys from app state so gestures/presets update the UI correctly.
        # -------------------------------------------------------------------------
        if ss.input_source not in SOURCE_OPTIONS:
            ss.input_source = SOURCE_SIM
        ss["input_source_select"] = ss.input_source

        if ss.camera_index < 0:
            ss.camera_index = 0
        if ss.camera_index > int(CONFIG["webcam"]["max_index"]):
            ss.camera_index = int(CONFIG["webcam"]["max_index"])
        ss["camera_index_input"] = int(ss.camera_index)

        ss["video_path_input"] = str(ss.video_path)
        ss["use_mp_webcam_checkbox"] = bool(ss.use_mediapipe_tracking)
        ss["use_mp_file_checkbox"] = bool(ss.use_mediapipe_tracking)

        size_options = ["640x360", "480x270", "320x180"]
        res_label = f"{int(ss.work_size[0])}x{int(ss.work_size[1])}"
        if res_label not in size_options:
            res_label = "640x360"
            ss.work_size = (640, 360)
        ss["resolution_select"] = res_label

        ss["mirror_checkbox"] = bool(CONFIG["capture"]["mirror"])
        ss["lefthanded_checkbox"] = bool(ss.left_handed)
        ss["cheat_checkbox"] = bool(ss.show_cheat)
        ss["autoquality_checkbox"] = bool(ss.auto_quality)
        ss["autoplay_slider"] = int(clamp(float(CONFIG["ui"]["autoplay_seconds"]), 1, 12))

        if ss.manual_gesture not in MANUAL_GESTURES:
            ss.manual_gesture = "Auto script"
        ss["gesture_source_select"] = ss.manual_gesture

        if ss.filter not in FILTER_BY_NAME:
            ss.filter = "Warm"
        ss["filter_select"] = ss.filter

        ss.intensity = clamp(float(ss.intensity), 0.0, 1.0)
        ss["intensity_slider"] = float(ss.intensity)

        ss.chain = [n for n in ss.chain if n in FILTER_BY_NAME][:3]
        ss["chain_multiselect"] = list(ss.chain)

        ss.zoom = clamp(float(ss.zoom), 1.0, 2.0)
        ss["zoom_slider"] = float(ss.zoom)

        ss["bypass_checkbox"] = bool(ss.bypass)

        g = CONFIG["gestures"]
        ss["swipe_dx_slider"] = float(g["swipe_dx_min"])
        ss["pinch_start_slider"] = float(g["pinch_start"])
        ss["pinch_end_slider"] = float(g["pinch_end"])
        ss["hold_ms_slider"] = int(g["hold_ms"])
        ss["arming_ttl_slider"] = int(g["arming_ttl_ms"])

        ss["replay_record_checkbox"] = bool(ss.replay_recording)

        # -------------------------------------------------------------------------
        # Input source
        # -------------------------------------------------------------------------
        ss.input_source = st.selectbox(
            "Input source",
            SOURCE_OPTIONS,
            key="input_source_select",
            help="Choose simulated scene, real webcam, or a local video file.",
        )

        if ss.input_source != ss.previous_input_source:
            stop_current_source()
            ss.previous_input_source = ss.input_source
            ss.last_gesture = f"Source changed: {ss.input_source}"

        if ss.input_source == SOURCE_WEB:
            ss.camera_index = int(
                st.number_input(
                    "Camera index",
                    min_value=0,
                    max_value=int(CONFIG["webcam"]["max_index"]),
                    step=1,
                    key="camera_index_input",
                )
            )
            st.caption("Camera stays local. Tell people in view that the camera is on.")
            wc1, wc2 = st.columns(2)
            wc1.button("Start webcam", key="start_webcam_btn", on_click=cb_start_webcam)
            wc2.button("Stop source", key="stop_webcam_btn", on_click=cb_stop_source)

            if HAVE_MEDIAPIPE:
                ss.use_mediapipe_tracking = bool(
                    st.checkbox(
                        "Use MediaPipe hand tracking",
                        key="use_mp_webcam_checkbox",
                    )
                )
            else:
                st.info("MediaPipe is not installed. Webcam video works, but hand landmarks stay simulated.")

            if ss.webcam_thread is not None and ss.webcam_thread.is_alive():
                st.success("Webcam thread running")
                st.metric("Capture FPS", f"{float(ss.webcam_fps):.1f}")
            if ss.webcam_error is not None:
                st.error(ss.webcam_error)

        elif ss.input_source == SOURCE_FILE:
            ss.video_path = st.text_input(
                "Video file path",
                key="video_path_input",
                help="Absolute or relative path to an mp4/avi file.",
            )
            vc1, vc2 = st.columns(2)
            vc1.button("Start video", key="start_video_btn", on_click=cb_start_video)
            vc2.button("Stop source", key="stop_video_btn", on_click=cb_stop_source)

            if HAVE_MEDIAPIPE:
                ss.use_mediapipe_tracking = bool(
                    st.checkbox(
                        "Use MediaPipe hand tracking",
                        key="use_mp_file_checkbox",
                    )
                )
            else:
                st.info("MediaPipe is not installed. Video playback works, but hand landmarks stay simulated.")

            if ss.webcam_thread is not None and ss.webcam_thread.is_alive():
                st.success("Video thread running")
                st.metric("Capture FPS", f"{float(ss.webcam_fps):.1f}")
            if ss.webcam_error is not None:
                st.error(ss.webcam_error)

        else:
            st.caption("Simulated scene starts instantly and privately.")

        selected_size = st.selectbox(
            "Work resolution",
            size_options,
            key="resolution_select",
        )
        new_size = tuple(int(v) for v in selected_size.split("x"))
        if new_size != ss.work_size:
            stop_current_source()
            ss.work_size = new_size

        CONFIG["capture"]["mirror"] = bool(st.checkbox("Mirror view", key="mirror_checkbox"))
        ss.left_handed = bool(st.checkbox("Left-handed mode", key="lefthanded_checkbox"))
        ss.show_cheat = bool(st.checkbox("Show cheat sheet", key="cheat_checkbox"))
        ss.auto_quality = bool(st.checkbox("Auto reduce quality if FPS drops", key="autoquality_checkbox"))

        st.divider()
        st.subheader("Demo playback")
        CONFIG["ui"]["autoplay_seconds"] = int(
            st.slider(
                "Autoplay seconds",
                min_value=1,
                max_value=12,
                value=int(CONFIG["ui"]["autoplay_seconds"]),
                step=1,
                key="autoplay_slider",
            )
        )

        st.divider()
        st.subheader("Gesture source")
        ss.manual_gesture = st.selectbox(
            "Gesture simulation",
            MANUAL_GESTURES,
            key="gesture_source_select",
        )

        st.divider()
        st.subheader("Filter")
        ss.filter = st.selectbox(
            "Current filter",
            FILTER_NAMES,
            key="filter_select",
        )
        ss.intensity = float(
            st.slider(
                "Intensity",
                min_value=0.0,
                max_value=1.0,
                value=float(ss.intensity),
                step=0.01,
                key="intensity_slider",
            )
        )
        ss.chain = st.multiselect("Chain filters (max 3)", FILTER_NAMES, key="chain_multiselect")
        if len(ss.chain) > 3:
            ss.chain = ss.chain[:3]
        ss.zoom = float(
            st.slider(
                "Zoom",
                min_value=1.0,
                max_value=2.0,
                value=float(ss.zoom),
                step=0.01,
                key="zoom_slider",
            )
        )
        ss.bypass = bool(st.checkbox("Bypass filters", key="bypass_checkbox"))

        st.divider()
        st.subheader("Tunable thresholds")
        g = CONFIG["gestures"]
        g["swipe_dx_min"] = float(
            st.slider(
                "Swipe dx min",
                min_value=0.05,
                max_value=0.50,
                value=float(g["swipe_dx_min"]),
                step=0.01,
                key="swipe_dx_slider",
            )
        )
        g["pinch_start"] = float(
            st.slider(
                "Pinch start ratio",
                min_value=0.05,
                max_value=0.50,
                value=float(g["pinch_start"]),
                step=0.01,
                key="pinch_start_slider",
            )
        )
        g["pinch_end"] = float(
            st.slider(
                "Pinch end ratio",
                min_value=0.05,
                max_value=0.80,
                value=float(g["pinch_end"]),
                step=0.01,
                key="pinch_end_slider",
            )
        )
        g["hold_ms"] = int(
            st.slider(
                "Open-palm hold ms",
                min_value=400,
                max_value=2500,
                value=int(g["hold_ms"]),
                step=50,
                key="hold_ms_slider",
            )
        )
        g["arming_ttl_ms"] = int(
            st.slider(
                "Arming TTL ms",
                min_value=2000,
                max_value=20000,
                value=int(g["arming_ttl_ms"]),
                step=500,
                key="arming_ttl_slider",
            )
        )

        st.divider()
        st.subheader("Privacy")
        st.markdown(
            """
            - No network upload
            - No API key
            - Records only when you press record or trigger a recording gesture
            - Delete all generated data from the Gallery tab
            """
        )

    tabs = st.tabs(
        [
            "🎥 Live Studio",
            "🎨 Filter Lab",
            "🖐️ Gesture Console",
            "🧠 Custom Gestures",
            "🖼️ Gallery & Recorder",
            "📊 Analytics",
            "🧪 QA & Limits",
            "🚀 Onboarding",
        ]
    )

    # -------------------------------------------------------------------------
    # Live Studio
    # -------------------------------------------------------------------------
    with tabs[0]:
        if ss.webcam_error is not None:
            st.warning(f"Source error: {ss.webcam_error}")

        k1, k2, k3, k4, k5, k6 = st.columns(6)
        k1.metric("FPS", f"{float(ss.fps):.1f}")
        k2.metric("Frame time", f"{float(ss.frame_ms):.1f} ms")
        k3.metric("Events", len(ss.events))
        k4.metric("Captures", len(list(CAPTURES_DIR.glob('*.png'))))
        k5.metric("Recordings", len(list(RECORDINGS_DIR.glob('*.mp4'))))
        k6.metric("Armed", "YES" if ss.time_ms <= ss.armed_until else "NO")

        col_video, col_status = st.columns([2.7, 1.0])
        with col_video:
            st.markdown('<div class="gcs-video-shell">', unsafe_allow_html=True)
            video_slot = st.empty()
            st.markdown('</div>', unsafe_allow_html=True)

        status_slot = col_status.empty()

        if not ss.booted:
            ss.booted = True
            ss.run_frames = int(float(CONFIG["ui"]["autoplay_seconds"]) * float(CONFIG["capture"]["fps"]))

        if ss.run_frames > 0:
            n = int(ss.run_frames)
            ss.run_frames = 0
            for _ in range(n):
                ss.time_ms += 33
                frame, info = compose_frame(ss.time_ms)
                video_slot.image(frame, use_container_width=True)
                status_slot.markdown(status_html(info), unsafe_allow_html=True)
                time.sleep(1.0 / max(1.0, float(CONFIG["capture"]["fps"])))
        else:
            frame, info = compose_frame(ss.time_ms)
            video_slot.image(frame, use_container_width=True)
            status_slot.markdown(status_html(info), unsafe_allow_html=True)

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.button("Play 2s", key="play2_btn", on_click=cb_play, args=(2.0,))
        c2.button("Play 10s", key="play10_btn", on_click=cb_play, args=(10.0,))
        c3.button("Step +1", key="step_btn", on_click=cb_step)
        c4.button("Reset", key="reset_btn", on_click=cb_reset_timeline)
        c5.button("Capture", key="capture_btn", on_click=cb_capture)
        c6.button("Toggle record", key="record_btn", on_click=cb_toggle_record)

        q1, q2, q3, q4, q5 = st.columns(5)
        q1.button("Previous filter", key="prev_live_btn", on_click=cb_force, args=("SWIPE_LEFT",))
        q2.button("Next filter", key="next_live_btn", on_click=cb_force, args=("SWIPE_RIGHT",))
        q3.button("Arm now", key="arm_live_btn", on_click=cb_arm_now)
        q4.button("Toggle bypass", key="bypass_live_btn", on_click=cb_force, args=("FIST",))
        q5.button("Cheat sheet", key="cheat_live_btn", on_click=lambda: setattr(ss, "show_cheat", not ss.show_cheat))

        with st.expander("Gesture cheat sheet"):
            st.markdown(
                """
                - **Swipe left/right**: previous/next filter.
                - **Pinch**: control intensity.
                - **Open palm held 1s**: arm and take photo.
                - **Fist**: toggle original vs filtered.
                - **Victory**: start/stop recording when armed.
                - **Pointing up**: show cheat sheet.
                - **Thumb up/down**: save/undo.
                - **Circle**: hue rotate.
                - **Fingers 3-4**: filter group.
                - **Two hands spread**: zoom.
                - **Clap**: reset.
                """
            )

    # -------------------------------------------------------------------------
    # Filter Lab
    # -------------------------------------------------------------------------
    with tabs[1]:
        st.subheader("Presets")

        preset_options = list(ss.presets.keys()) or ["none"]
        if ss.get("preset_select") not in preset_options:
            ss["preset_select"] = preset_options[0]

        pcol1, pcol2, pcol3, pcol4, pcol5 = st.columns([2, 1, 1, 1, 2])
        preset_name = pcol1.text_input("Preset name", key="preset_name_input")
        pcol2.button("Save", key="save_preset_btn", on_click=cb_save_preset, args=(preset_name,))
        chosen_preset = pcol3.selectbox("Load preset", preset_options, key="preset_select")
        pcol4.button("Load", key="load_preset_btn", on_click=cb_load_preset, args=(chosen_preset,))
        pcol5.button(
            "Delete preset",
            key="delete_preset_btn",
            on_click=cb_delete_preset,
            args=(chosen_preset,),
            disabled=(chosen_preset == "none"),
        )

        st.markdown("**Current chain:** " + ", ".join([ss.filter] + [n for n in ss.chain if n != ss.filter]))

        dl1, dl2 = st.columns(2)
        dl1.download_button(
            "Export presets JSON",
            data=json.dumps(ss.presets, indent=2).encode("utf-8"),
            file_name="gesturecam_presets.json",
            mime="application/json",
            key="export_presets_btn",
        )
        uploaded = dl2.file_uploader("Import presets JSON", type=["json"], key="import_presets_uploader")
        if uploaded is not None:
            try:
                data = json.loads(uploaded.getvalue().decode("utf-8"))
                if isinstance(data, dict):
                    ss.presets.update(data)
                    save_presets()
                    st.success("Presets imported.")
                else:
                    st.error("Preset file must be a JSON object.")
            except Exception as exc:
                st.error(f"Import failed: {exc}")

        st.subheader("Filter previews")
        preview_key = (round(float(ss.intensity), 2), int(CONFIG["ui"]["preview_width"]), int(CONFIG["ui"]["preview_height"]))
        if st.button("Refresh filter previews", key="refresh_previews_btn") or not ss.filter_previews or ss.preview_key != preview_key:
            with st.spinner("Rendering filter previews..."):
                ss.filter_previews = build_filter_previews(float(ss.intensity))
                ss.preview_key = preview_key

        if ss.filter_previews:
            cols_per_row = 4
            names = list(ss.filter_previews.keys())
            for start in range(0, len(names), cols_per_row):
                row = names[start:start + cols_per_row]
                cols = st.columns(cols_per_row)
                for col, name in zip(cols, row):
                    img = ss.filter_previews.get(name)
                    if img is not None:
                        col.image(img, use_container_width=True)
                        safe_key = name.replace(" ", "_").replace("&", "and")
                        col.button(f"Use {name}", key=f"use_{safe_key}", on_click=cb_set_filter, args=(name,))

        st.subheader("Filter profiler")
        if ss.profiler:
            rows = "| Filter | Avg ms/frame | Calls |\n|---|---:|---:|\n"
            for name, stat in sorted(ss.profiler.items(), key=lambda kv: -kv[1]["ms"])[:15]:
                rows += f"| {name} | {stat['ms']:.2f} | {stat['n']} |\n"
            st.markdown(rows)
        else:
            st.write("No filter timings recorded yet. Run the live studio first.")

    # -------------------------------------------------------------------------
    # Gesture Console
    # -------------------------------------------------------------------------
    with tabs[2]:
        st.subheader("Force gesture events")
        st.caption("These buttons emit the same GestureEvent objects used by the simulated tracker.")

        b1, b2, b3, b4, b5 = st.columns(5)
        b1.button("Swipe left", key="swipe_left_btn", on_click=cb_force, args=("SWIPE_LEFT",))
        b1.button("Swipe right", key="swipe_right_btn", on_click=cb_force, args=("SWIPE_RIGHT",))
        b2.button("Pinch 0.8", key="pinch_btn", on_click=cb_force, args=("PINCH", {"intensity": 0.8}))
        b2.button("Open palm", key="open_palm_btn", on_click=cb_force, args=("OPEN_PALM",))
        b3.button("Fist", key="fist_btn", on_click=cb_force, args=("FIST",))
        b3.button("Victory", key="victory_btn", on_click=cb_force, args=("VICTORY",))
        b4.button("Thumb up", key="thumb_up_btn", on_click=cb_force, args=("THUMB_UP",))
        b4.button("Thumb down", key="thumb_down_btn", on_click=cb_force, args=("THUMB_DOWN",))
        b5.button("Point up", key="point_up_btn", on_click=cb_force, args=("POINTING_UP",))
        b5.button("Circle CW", key="circle_cw_btn", on_click=cb_force, args=("CIRCLE_CW",))

        b6, b7, b8, b9, b10 = st.columns(5)
        b6.button("Fingers 3", key="fingers3_btn", on_click=cb_force, args=("FINGERS_3",))
        b7.button("Fingers 4", key="fingers4_btn", on_click=cb_force, args=("FINGERS_4",))
        b8.button("Spread", key="spread_btn", on_click=cb_force, args=("SPREAD", {"zoom": 1.35}))
        b9.button("Clap", key="clap_btn", on_click=cb_force, args=("CLAP",))
        b10.button("Arm now", key="arm_console_btn", on_click=cb_arm_now)

        st.subheader("Engine feedback")
        fb = ss.engine.feedback()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Hold progress", f"{fb.get('hold_progress', 0.0) * 100:.0f}%")
        m2.metric("Last stable label", fb.get("last_label", "NONE"))
        m3.metric("Armed until ms", int(ss.armed_until))
        m4.metric("Current time ms", int(ss.time_ms))

        st.subheader("Replay harness")
        ss.replay_recording = bool(
            st.checkbox(
                "Record landmark stream to buffer",
                key="replay_record_checkbox",
            )
        )
        r1, r2, r3 = st.columns(3)
        r1.button("Save replay", key="save_replay_btn", on_click=cb_save_replay)
        r2.button("Replay latest file", key="replay_latest_btn", on_click=cb_replay_latest)
        r3.metric("Buffer frames", len(ss.replay_buffer))

        if ss.replay_result is not None:
            st.write(f"Replayed frames: {ss.replay_result['frames']}")
            st.json(ss.replay_result["events"][:40])

    # -------------------------------------------------------------------------
    # Custom Gestures
    # -------------------------------------------------------------------------
    with tabs[3]:
        st.subheader("Custom gesture trainer")
        st.caption(
            "Record the current hand pose, augment it locally, and train a tiny kNN-style model. "
            "This stays on your machine and saves to settings/custom_gestures.json."
        )

        c1, c2, c3 = st.columns([2, 1, 1])
        st.text_input("Custom gesture name", key="custom_label_name")
        c1.button("Record sample + augment", key="record_custom_btn", on_click=cb_record_custom_sample)
        c2.button("Delete label", key="delete_custom_btn", on_click=cb_clear_custom_label)
        c3.button("Clear all custom", key="clear_custom_btn", on_click=cb_clear_all_custom)

        summary = ss.custom_trainer.summary()
        if summary:
            st.markdown("**Stored samples**")
            st.bar_chart({"Gesture": list(summary.keys()), "Samples": list(summary.values())})
        else:
            st.write("No custom gestures trained yet.")

        lm = first_available(ss.engine.filtered, "right", "left")
        if lm is None:
            hands = current_hands(ss.time_ms)
            lm = first_available(hands, "right", "left")

        if lm is not None:
            vec = landmark_vector(lm)
            pred, conf = ss.custom_trainer.predict(vec, reject_distance=float(CONFIG["custom"]["reject_distance"]))
            p1, p2 = st.columns(2)
            p1.metric("Predicted custom gesture", pred)
            p2.metric("Confidence", f"{conf:.2f}")
        else:
            st.info("No hand landmarks available for prediction.")

    # -------------------------------------------------------------------------
    # Gallery & Recorder
    # -------------------------------------------------------------------------
    with tabs[4]:
        st.subheader("Captures")
        capture_files = sorted(CAPTURES_DIR.glob("*.png"))[-12:]
        if capture_files:
            cols = st.columns(min(4, len(capture_files)))
            for i, p in enumerate(capture_files):
                with cols[i % len(cols)]:
                    st.image(str(p), use_container_width=True)
                    st.caption(p.name)
        else:
            st.write("No captures yet. Trigger open palm or press Capture photo.")

        st.subheader("Recordings")
        recording_files = sorted(RECORDINGS_DIR.glob("*.mp4"))[-6:]
        if recording_files:
            for p in recording_files:
                st.write(p.name)
                try:
                    st.video(str(p.resolve()))
                except Exception:
                    st.write("Preview unavailable, but the file exists.")
                st.download_button(
                    "Download",
                    data=open(p, "rb").read(),
                    file_name=p.name,
                    mime="video/mp4",
                    key=f"dl_{p.name}",
                )
        else:
            st.write("No recordings yet. Arm with open palm, then use Victory or Toggle record.")

        st.subheader("Data controls")
        confirm = bool(
            st.checkbox(
                "I confirm deleting all generated captures, recordings, logs, demos, presets, and custom gestures",
                key="delete_confirm_checkbox",
            )
        )
        st.button("Delete all data", key="delete_all_btn", on_click=cb_delete_all, disabled=not confirm, type="secondary")

    # -------------------------------------------------------------------------
    # Analytics
    # -------------------------------------------------------------------------
    with tabs[5]:
        st.subheader("Runtime metrics")

        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Simulated FPS", f"{float(ss.fps):.1f}")
        m2.metric("Median frame ms", f"{percentile(ss.latency_samples, 50):.1f}")
        m3.metric("P95 frame ms", f"{percentile(ss.latency_samples, 95):.1f}")
        m4.metric("Logged events", len(ss.events))
        m5.metric("Custom labels", len(ss.custom_trainer.summary()))
        m6.metric("Replay buffer", len(ss.replay_buffer))

        st.subheader("Gesture event counts")
        if ss.events:
            counts = Counter(ev["type"] for ev in ss.events)
            st.bar_chart({"Gesture": list(counts.keys()), "Count": list(counts.values())})
            st.json(dict(counts))
        else:
            st.write("No events yet.")

        st.subheader("Top filter costs")
        if ss.profiler:
            rows = "| Filter | Avg ms/frame | Calls |\n|---|---:|---:|\n"
            for name, stat in sorted(ss.profiler.items(), key=lambda kv: -kv[1]["ms"])[:20]:
                rows += f"| {name} | {stat['ms']:.2f} | {stat['n']} |\n"
            st.markdown(rows)
        else:
            st.write("No profiler data yet.")

        st.subheader("Export report")
        report = {
            "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "fps": float(ss.fps),
            "median_frame_ms": percentile(ss.latency_samples, 50),
            "p95_frame_ms": percentile(ss.latency_samples, 95),
            "event_count": len(ss.events),
            "event_counts": dict(Counter(ev["type"] for ev in ss.events)),
            "profiler": ss.profiler,
            "custom_gestures": ss.custom_trainer.summary(),
            "config": _jsonable(CONFIG),
        }
        st.download_button(
            "Download report JSON",
            data=json.dumps(report, indent=2).encode("utf-8"),
            file_name="gesturecam_report.json",
            mime="application/json",
            key="download_report_btn",
        )

        if ss.events:
            buf = io.StringIO()
            buf.write("t_ms,type,hand,confidence\n")
            for ev in ss.events:
                buf.write(f"{ev['t_ms']},{ev['type']},{ev['hand']},{ev['confidence']}\n")
            st.download_button(
                "Download events CSV",
                data=buf.getvalue().encode("utf-8"),
                file_name="events.csv",
                mime="text/csv",
                key="download_events_btn",
            )

    # -------------------------------------------------------------------------
    # QA & Limits
    # -------------------------------------------------------------------------
    with tabs[6]:
        st.subheader("Self-tests")
        t1, t2 = st.columns(2)
        t1.button("Run self-tests", key="run_tests_btn", on_click=cb_run_tests)
        t2.button("Run scripted gesture evaluation", key="run_script_eval_btn", on_click=cb_run_script_eval)

        if ss.test_results:
            for name, passed, msg in ss.test_results:
                if passed:
                    st.success(f"{name}: {msg}")
                else:
                    st.error(f"{name}: {msg}")
        else:
            st.write("Press Run self-tests to validate geometry, smoothing, filters, replay, and custom training.")

        if ss.script_eval is not None:
            st.subheader("Scripted gesture evaluation")
            s1, s2 = st.columns(2)
            s1.metric("Synthetic recall", f"{ss.script_eval['recall']:.2f}")
            s2.metric("Missing gestures", len(ss.script_eval["missing"]))
            st.json(ss.script_eval)

        st.subheader("Known limits")
        st.markdown(
            """
            - Webcam input is supported through OpenCV.
            - If MediaPipe is not installed, webcam/video works but hand landmarks remain simulated.
            - Install optional `mediapipe` to enable real hand tracking on webcam/video frames.
            - Background segmentation is simulated with a procedural subject mask unless a real segmenter is added.
            - Audio recording, virtual camera, PyInstaller packaging, and the browser/WebGL version are not included in this single-file Streamlit build.
            - Thresholds are starting points and should be tuned on real recordings.
            - Accuracy claims are limited to the built-in synthetic tests; no multi-participant evaluation is performed here.
            """
        )

        st.subheader("Licenses")
        st.markdown(
            """
            - Streamlit: Apache-2.0  
            - NumPy: BSD-3-Clause  
            - Pillow: MIT-CMU / HPND-style license  
            - OpenCV-python-headless: Apache-2.0  
            - MediaPipe, if installed: Apache-2.0  
            - Python standard library: PSF License  
            - Synthetic demo visuals: generated locally by this app, no third-party media asset license required  
            """
        )

    # -------------------------------------------------------------------------
    # Onboarding
    # -------------------------------------------------------------------------
    with tabs[7]:
        st.subheader("Onboarding wizard")
        st.markdown(
            """
            ### Quick start
            1. Launch the app.
            2. The Live Studio starts with an animated simulated demo automatically.
            3. Switch Input source to **Real webcam** to use your camera.
            4. Press **Start webcam** in the sidebar.
            5. Use **Play 10s** to watch the full gesture script.
            6. Use the **Gesture Console** to force individual gestures.
            7. Use **Filter Lab** to preview and apply filters.
            8. Use **Gallery** to review captures and recordings.
            9. Use **Delete all data** whenever you want to clear generated files.
            """
        )

        st.markdown("### Practice gestures")
        p1, p2, p3, p4 = st.columns(4)
        p1.button("Practice open palm", key="practice_open_btn", on_click=cb_force, args=("OPEN_PALM",))
        p2.button("Practice swipe right", key="practice_swipe_btn", on_click=cb_force, args=("SWIPE_RIGHT",))
        p3.button("Practice fist", key="practice_fist_btn", on_click=cb_force, args=("FIST",))
        p4.button("Practice victory", key="practice_victory_btn", on_click=cb_force, args=("VICTORY",))

        st.markdown("### Demo reel")
        d1, d2 = st.columns(2)
        d1.button("Generate / refresh demo reel", key="generate_demo_btn", on_click=cb_generate_demo)

        if ss.demo_gif is not None and Path(ss.demo_gif).exists():
            st.image(str(ss.demo_gif), use_container_width=True)
            st.download_button(
                "Download GIF",
                data=open(ss.demo_gif, "rb").read(),
                file_name="gesturecam_demo.gif",
                mime="image/gif",
                key="download_demo_gif_btn",
            )

        if ss.demo_mp4 is not None and Path(ss.demo_mp4).exists():
            st.video(str(Path(ss.demo_mp4).resolve()))
            st.download_button(
                "Download MP4",
                data=open(ss.demo_mp4, "rb").read(),
                file_name="gesturecam_demo.mp4",
                mime="video/mp4",
                key="download_demo_mp4_btn",
            )

        st.markdown("### Privacy notice")
        st.info(
            "This app does not upload video. Webcam/video processing stays local. "
            "Generated captures, recordings, logs, demos, presets, and custom gesture models are stored locally "
            "and can be deleted from the Gallery tab."
        )

    st.markdown(
        """
        <div class="gcs-mini" style="margin-top:1.2rem;">
          GestureCam Studio Pro • local-first • webcam support • single-file Streamlit build
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()