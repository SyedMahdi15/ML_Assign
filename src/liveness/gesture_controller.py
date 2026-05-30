"""Touchless hand-gesture controller for liveness demo security workflows.

This module provides a standalone DNN-style gesture controller built on top of
MediaPipe Hands landmark inference. The controller is intentionally lightweight,
real-time friendly, and designed for webcam interaction loops.

Primary gesture policy:
- 0 fingers  -> emergency lock
- 1..5 fingers -> FILTER_1 .. FILTER_5 (one unique filter per finger count)
- filter-cycle trigger is a deliberate palm swipe (hand-level motion) with cooldown.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from enum import Enum
import math
from typing import Sequence

import cv2
import mediapipe as mp
import numpy as np


class GestureMode(str, Enum):
    """Public gesture mode values exposed to host applications."""

    NONE = "NONE"
    FILTER_1 = "FILTER_1"
    FILTER_2 = "FILTER_2"
    FILTER_3 = "FILTER_3"
    FILTER_4 = "FILTER_4"
    FILTER_5 = "FILTER_5"
    EMERGENCY_LOCK = "EMERGENCY_LOCK"


FILTER_SEQUENCE = [
    GestureMode.FILTER_1,
    GestureMode.FILTER_2,
    GestureMode.FILTER_3,
    GestureMode.FILTER_4,
    GestureMode.FILTER_5,
]


@dataclass(frozen=True)
class GestureState:
    """Snapshot of current gesture controller output."""

    finger_count: int
    mode: GestureMode
    label: str
    confidence: float
    cycle_default_requested: bool


class DNNGestureController:
    """Real-time MediaPipe Hands gesture controller.

    The implementation uses geometric rules over 21 hand landmarks to estimate
    the number of extended fingers. While not a trainable neural head by itself,
    it relies on MediaPipe's learned hand landmark detector and applies strict,
    deterministic decision logic for predictable behavior in security demos.
    """

    TIP_IDS = {
        "thumb": 4,
        "index": 8,
        "middle": 12,
        "ring": 16,
        "pinky": 20,
    }
    PIP_IDS = {
        "thumb": 3,
        "index": 6,
        "middle": 10,
        "ring": 14,
        "pinky": 18,
    }
    MCP_IDS = {
        "thumb": 2,
        "index": 5,
        "middle": 9,
        "ring": 13,
        "pinky": 17,
    }

    WRIST_ID = 0

    def __init__(
        self,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.6,
        min_tracking_confidence: float = 0.6,
        wave_cooldown_s: float = 1.4,
        swipe_dx_threshold: float = 0.11,
        flick_min_interval_s: float = 0.08,
    ) -> None:
        """Initialize MediaPipe Hands and internal runtime state."""
        self._mp_hands = mp.solutions.hands
        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        self._drawer = mp.solutions.drawing_utils
        self._connections = self._mp_hands.HAND_CONNECTIONS

        # Palm-swipe detector state.
        # The swipe system is intentionally separated from per-finger classification.
        # This allows users to keep one default visual mode and change it with an
        # explicit large hand movement, reducing accidental toggles from tiny jitters.
        self._palm_x_history: deque[float] = deque(maxlen=8)
        self._last_swipe_ts = 0.0
        self._swipe_dx_threshold = float(swipe_dx_threshold)
        self._flick_min_interval_s = float(flick_min_interval_s)
        self._wave_cooldown_s = float(wave_cooldown_s)
        self._last_cycle_ts = 0.0
        self._count_history: deque[int] = deque(maxlen=5)

    @staticmethod
    def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
        """Euclidean distance helper in normalized coordinate space."""
        return math.dist(a, b)

    @staticmethod
    def _pt(landmarks: Sequence, idx: int) -> tuple[float, float]:
        """Return normalized (x, y) coordinate for a landmark index."""
        lm = landmarks[idx]
        return float(lm.x), float(lm.y)

    def _finger_extended(self, landmarks: Sequence, finger: str) -> bool:
        """Determine if a non-thumb finger is extended.

        This implementation is orientation-tolerant:
        1) Uses a joint angle at the PIP to measure straightness.
        2) Uses distance growth from MCP->TIP relative to MCP->PIP.
        These two cues together are more stable than vertical-only rules.
        """
        tip = self._pt(landmarks, self.TIP_IDS[finger])
        pip = self._pt(landmarks, self.PIP_IDS[finger])
        mcp = self._pt(landmarks, self.MCP_IDS[finger])

        d_tip = self._distance(tip, mcp)
        d_pip = self._distance(pip, mcp)
        ratio = d_tip / max(1e-6, d_pip)
        angle = self._joint_angle_degrees(mcp, pip, tip)
        # A finger is accepted as extended only when:
        # 1) it is straight enough (angle),
        # 2) tip is clearly above PIP and MCP in image coordinates,
        # 3) length expansion is meaningfully larger than folded posture.
        upward_clearance = (pip[1] - tip[1]) > 0.015 and (mcp[1] - pip[1]) > 0.005
        return ratio > 1.16 and angle > 158.0 and upward_clearance

    @staticmethod
    def _joint_angle_degrees(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
        """Return angle ABC in degrees for 2D points."""
        ba = np.array([a[0] - b[0], a[1] - b[1]], dtype=np.float32)
        bc = np.array([c[0] - b[0], c[1] - b[1]], dtype=np.float32)
        nba = float(np.linalg.norm(ba))
        nbc = float(np.linalg.norm(bc))
        if nba <= 1e-6 or nbc <= 1e-6:
            return 0.0
        cosang = float(np.dot(ba, bc) / (nba * nbc))
        cosang = max(-1.0, min(1.0, cosang))
        return float(np.degrees(np.arccos(cosang)))

    def _thumb_extended(self, landmarks: Sequence, handedness_label: str | None) -> bool:
        """Determine thumb extension using spread and left/right orientation."""
        wrist = self._pt(landmarks, self.WRIST_ID)
        thumb_tip = self._pt(landmarks, self.TIP_IDS["thumb"])
        thumb_mcp = self._pt(landmarks, self.MCP_IDS["thumb"])
        index_mcp = self._pt(landmarks, self.MCP_IDS["index"])
        pinky_mcp = self._pt(landmarks, self.MCP_IDS["pinky"])

        palm_width = self._distance(index_mcp, pinky_mcp)
        spread = self._distance(thumb_tip, wrist)
        base = self._distance(thumb_mcp, wrist)
        is_spread = spread > base * 1.20 and spread > palm_width * 0.56

        # Orientation gate reduces false positives when thumb is folded.
        if handedness_label == "Right":
            lateral_ok = thumb_tip[0] < index_mcp[0] - 0.01
        elif handedness_label == "Left":
            lateral_ok = thumb_tip[0] > index_mcp[0] + 0.01
        else:
            lateral_ok = abs(thumb_tip[0] - index_mcp[0]) > 0.04
        return is_spread and lateral_ok

    def count_extended_fingers(self, landmarks: Sequence, handedness_label: str | None) -> tuple[int, dict[str, bool]]:
        """Count extended fingers and return per-finger extension flags."""
        flags: dict[str, bool] = {}
        flags["thumb"] = self._thumb_extended(landmarks, handedness_label)
        for finger in ["index", "middle", "ring", "pinky"]:
            flags[finger] = self._finger_extended(landmarks, finger)
        raw_count = int(sum(flags.values()))
        self._count_history.append(raw_count)

        # Majority voting across a short window prevents frame-level spikes,
        # such as sudden one-frame jumps from "one finger" to "open palm".
        # This improves UI stability for webcam demos under motion blur.
        voted_count = raw_count
        if len(self._count_history) >= 3:
            voted_count = Counter(self._count_history).most_common(1)[0][0]
        return voted_count, flags

    def _palm_center_x(self, landmarks: Sequence) -> float:
        """Estimate palm center x from wrist + MCP joints."""
        ids = [self.WRIST_ID, self.MCP_IDS["index"], self.MCP_IDS["middle"], self.MCP_IDS["ring"], self.MCP_IDS["pinky"]]
        xs = [float(landmarks[i].x) for i in ids]
        return float(sum(xs) / len(xs))

    def _detect_palm_swipe_cycle_request(self, landmarks: Sequence, timestamp_s: float) -> bool:
        """Detect deliberate left-right or right-left palm swipe to cycle default filter.

        We track palm center x across a short rolling window and require:
        1) strong horizontal span,
        2) enough time since last accepted swipe,
        3) global cooldown between cycle events.
        """
        palm_x = self._palm_center_x(landmarks)
        self._palm_x_history.append(palm_x)

        # Wait until the history buffer is full so we can estimate true span.
        if len(self._palm_x_history) < self._palm_x_history.maxlen:
            return False

        # Micro-cooldown to avoid double-trigger in adjacent frames.
        if (timestamp_s - self._last_swipe_ts) < self._flick_min_interval_s:
            return False

        span = max(self._palm_x_history) - min(self._palm_x_history)
        # Require sufficient horizontal palm travel to classify as a deliberate swipe.
        if span < self._swipe_dx_threshold:
            return False

        self._last_swipe_ts = timestamp_s
        # Global cooldown ensures one swipe corresponds to one mode cycle.
        if (timestamp_s - self._last_cycle_ts) >= self._wave_cooldown_s:
            self._last_cycle_ts = timestamp_s
            self._palm_x_history.clear()
            return True
        return False

    def _mode_from_finger_count(self, count: int) -> tuple[GestureMode, str, float]:
        """Map finger count to mode where each count has one unique filter."""
        if count == 0:
            return GestureMode.EMERGENCY_LOCK, "FIST", 1.0
        if count == 1:
            return GestureMode.FILTER_1, "ONE_FINGER", 0.90
        if count == 2:
            return GestureMode.FILTER_2, "TWO_FINGERS", 0.92
        if count == 3:
            return GestureMode.FILTER_3, "THREE_FINGERS", 0.94
        if count == 4:
            return GestureMode.FILTER_4, "FOUR_FINGERS", 0.95
        if count == 5:
            return GestureMode.FILTER_5, "OPEN_PALM", 0.96
        return GestureMode.NONE, "NO_VALID_GESTURE", 0.50

    def update(self, frame_bgr: np.ndarray, timestamp_s: float | None = None) -> tuple[GestureState, np.ndarray, bool]:
        """Run one gesture inference pass on a BGR frame.

        Returns:
        - GestureState: interpreted command + cycle request event
        - annotated frame: hand skeleton rendered
        - has_hand: whether a hand was detected
        """
        if timestamp_s is None:
            timestamp_s = 0.0

        annotated = frame_bgr.copy()
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self._hands.process(rgb)

        if not results.multi_hand_landmarks:
            self._palm_x_history.clear()
            self._count_history.clear()
            return GestureState(0, GestureMode.NONE, "NO_HAND", 0.0, False), annotated, False

        hand_landmarks = results.multi_hand_landmarks[0]
        handedness_label = None
        if results.multi_handedness:
            handedness_label = results.multi_handedness[0].classification[0].label
        self._drawer.draw_landmarks(annotated, hand_landmarks, self._connections)

        count, _ = self.count_extended_fingers(hand_landmarks.landmark, handedness_label)
        mode, label, confidence = self._mode_from_finger_count(count)
        cycle_default_requested = self._detect_palm_swipe_cycle_request(
            hand_landmarks.landmark,
            float(timestamp_s),
        )

        return GestureState(count, mode, label, confidence, cycle_default_requested), annotated, True

    def close(self) -> None:
        """Release MediaPipe graph resources."""
        self._hands.close()
