"""Interactive liveness challenge state machine for webcam anti-spoofing.

This module provides a strict three-stage challenge-response flow driven by
MediaPipe Face Mesh landmarks.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
import random
from typing import Sequence


class LivenessResult(str, Enum):
    """Public status values returned by the state machine."""

    LIVENESS_LOCKED = "LIVENESS_LOCKED"
    IN_PROGRESS = "IN_PROGRESS"
    PASSED = "PASSED"
    FAILED = "FAILED"


class ChallengeType(str, Enum):
    """Supported challenge primitives."""

    BLINK = "BLINK"
    TURN_LEFT = "TURN_LEFT"
    TURN_RIGHT = "TURN_RIGHT"
    MOUTH_OPEN = "MOUTH_OPEN"
    NOD_HEAD = "NOD_HEAD"
    SMILE = "SMILE"


@dataclass(frozen=True)
class LivenessState:
    """Snapshot of current liveness process state."""

    status: LivenessResult
    prompt: str
    stage_index: int
    total_stages: int
    seconds_remaining: float


class LivenessChallenge:
    """Challenge-response liveness gatekeeper for face verification."""

    LEFT_EYE = (33, 160, 158, 133, 153, 144)
    RIGHT_EYE = (362, 385, 387, 263, 373, 380)
    NOSE_TIP = 1
    FOREHEAD = 10
    CHIN = 152
    LEFT_CHEEK = 234
    RIGHT_CHEEK = 454
    LEFT_EYE_INNER = 133
    RIGHT_EYE_INNER = 362
    UPPER_LIP = 13
    LOWER_LIP = 14
    MOUTH_LEFT = 78
    MOUTH_RIGHT = 308
    SMILE_LEFT = 61
    SMILE_RIGHT = 291

    def __init__(
        self,
        timeout_per_stage_s: float = 5.0,
        blink_ear_threshold: float = 0.20,
        face_missing_grace_s: float = 1.2,
        random_seed: int | None = None,
    ) -> None:
        self.timeout_per_stage_s = float(timeout_per_stage_s)
        self.blink_ear_threshold = float(blink_ear_threshold)
        self.face_missing_grace_s = float(face_missing_grace_s)
        self._rng = random.Random(random_seed)

        self._sequence: list[ChallengeType] = []
        self._stage_index = 0
        self._stage_start_ts: float | None = None
        self._status = LivenessResult.LIVENESS_LOCKED

        self._blink_closed_seen = False
        self._nod_motion_seen = False
        self._face_missing_since_ts: float | None = None

        self._left_turn_ratio_max = 0.46
        self._right_turn_ratio_min = 0.54
        self._mouth_open_delta_threshold = 0.08
        self._mouth_open_absolute_threshold = 0.32
        self._nod_delta_threshold = 0.08
        self._nod_recovery_margin = 0.03
        self._smile_delta_threshold = 0.10

        self._mouth_baseline_ratio: float | None = None
        self._nod_baseline_ratio: float | None = None
        self._smile_baseline_ratio: float | None = None

    @property
    def status(self) -> LivenessResult:
        return self._status

    def reset(self) -> None:
        self._sequence = []
        self._stage_index = 0
        self._stage_start_ts = None
        self._status = LivenessResult.LIVENESS_LOCKED
        self._blink_closed_seen = False
        self._nod_motion_seen = False
        self._face_missing_since_ts = None
        self._mouth_baseline_ratio = None
        self._nod_baseline_ratio = None
        self._smile_baseline_ratio = None

    def start(self) -> None:
        """Start a strict randomized 3-stage challenge sequence."""
        pool = [
            ChallengeType.BLINK,
            ChallengeType.TURN_LEFT,
            ChallengeType.TURN_RIGHT,
            ChallengeType.MOUTH_OPEN,
            ChallengeType.NOD_HEAD,
            ChallengeType.SMILE,
        ]
        self._sequence = self._rng.sample(pool, k=3)
        self._stage_index = 0
        self._stage_start_ts = None
        self._status = LivenessResult.IN_PROGRESS
        self._blink_closed_seen = False
        self._nod_motion_seen = False
        self._face_missing_since_ts = None
        self._mouth_baseline_ratio = None
        self._nod_baseline_ratio = None
        self._smile_baseline_ratio = None

    def _get_xy(self, landmarks: Sequence, index: int) -> tuple[float, float]:
        lm = landmarks[index]
        return float(lm.x), float(lm.y)

    @staticmethod
    def _distance(p1: tuple[float, float], p2: tuple[float, float]) -> float:
        return math.dist(p1, p2)

    def _ear_for_eye(self, landmarks: Sequence, idx: tuple[int, int, int, int, int, int]) -> float:
        p1 = self._get_xy(landmarks, idx[0])
        p2 = self._get_xy(landmarks, idx[1])
        p3 = self._get_xy(landmarks, idx[2])
        p4 = self._get_xy(landmarks, idx[3])
        p5 = self._get_xy(landmarks, idx[4])
        p6 = self._get_xy(landmarks, idx[5])

        denom = 2.0 * self._distance(p1, p4)
        if denom <= 1e-8:
            return 0.0
        return (self._distance(p2, p6) + self._distance(p3, p5)) / denom

    def _average_ear(self, landmarks: Sequence) -> float:
        left = self._ear_for_eye(landmarks, self.LEFT_EYE)
        right = self._ear_for_eye(landmarks, self.RIGHT_EYE)
        return (left + right) / 2.0

    def check_blink(self, landmarks: Sequence) -> bool:
        """Strict blink: EAR must drop below threshold and then recover."""
        ear = self._average_ear(landmarks)
        if ear < self.blink_ear_threshold:
            self._blink_closed_seen = True
            return False
        if self._blink_closed_seen and ear >= (self.blink_ear_threshold + 0.02):
            self._blink_closed_seen = False
            return True
        return False

    def _nose_cheek_ratio(self, landmarks: Sequence) -> float:
        nose_x, _ = self._get_xy(landmarks, self.NOSE_TIP)
        left_x, _ = self._get_xy(landmarks, self.LEFT_CHEEK)
        right_x, _ = self._get_xy(landmarks, self.RIGHT_CHEEK)
        denom = right_x - left_x
        if abs(denom) <= 1e-8:
            return 0.5
        return (nose_x - left_x) / denom

    def check_head_turn(self, landmarks: Sequence) -> tuple[bool, str]:
        ratio = self._nose_cheek_ratio(landmarks)
        if ratio <= self._left_turn_ratio_max:
            return True, "LEFT"
        if ratio >= self._right_turn_ratio_min:
            return True, "RIGHT"
        return False, "CENTER"

    def _mouth_open_ratio(self, landmarks: Sequence) -> float:
        upper = self._get_xy(landmarks, self.UPPER_LIP)
        lower = self._get_xy(landmarks, self.LOWER_LIP)
        left = self._get_xy(landmarks, self.MOUTH_LEFT)
        right = self._get_xy(landmarks, self.MOUTH_RIGHT)

        mouth_width = self._distance(left, right)
        if mouth_width <= 1e-8:
            return 0.0
        mouth_height = self._distance(upper, lower)
        return mouth_height / mouth_width

    def check_mouth_open(self, landmarks: Sequence) -> bool:
        ratio = self._mouth_open_ratio(landmarks)
        if self._mouth_baseline_ratio is None:
            self._mouth_baseline_ratio = ratio
            return False
        required = max(
            self._mouth_open_absolute_threshold,
            self._mouth_baseline_ratio + self._mouth_open_delta_threshold,
        )
        return ratio >= required

    def _nose_vertical_ratio(self, landmarks: Sequence) -> float:
        _, nose_y = self._get_xy(landmarks, self.NOSE_TIP)
        _, forehead_y = self._get_xy(landmarks, self.FOREHEAD)
        _, chin_y = self._get_xy(landmarks, self.CHIN)
        denom = chin_y - forehead_y
        if abs(denom) <= 1e-8:
            return 0.5
        return (nose_y - forehead_y) / denom

    def check_nod_head(self, landmarks: Sequence) -> bool:
        ratio = self._nose_vertical_ratio(landmarks)
        if self._nod_baseline_ratio is None:
            self._nod_baseline_ratio = ratio
            return False

        delta = ratio - self._nod_baseline_ratio
        if abs(delta) >= self._nod_delta_threshold:
            self._nod_motion_seen = True
            return False

        if self._nod_motion_seen and abs(delta) <= self._nod_recovery_margin:
            self._nod_motion_seen = False
            return True

        return False

    def _smile_ratio(self, landmarks: Sequence) -> float:
        left_eye = self._get_xy(landmarks, self.LEFT_EYE_INNER)
        right_eye = self._get_xy(landmarks, self.RIGHT_EYE_INNER)
        mouth_left = self._get_xy(landmarks, self.SMILE_LEFT)
        mouth_right = self._get_xy(landmarks, self.SMILE_RIGHT)

        eye_distance = self._distance(left_eye, right_eye)
        if eye_distance <= 1e-8:
            return 0.0
        return self._distance(mouth_left, mouth_right) / eye_distance

    def check_smile(self, landmarks: Sequence) -> bool:
        """Strict smile: mouth-corner distance must rise above the resting baseline."""
        ratio = self._smile_ratio(landmarks)
        if self._smile_baseline_ratio is None:
            self._smile_baseline_ratio = ratio
            return False
        return ratio >= (self._smile_baseline_ratio + self._smile_delta_threshold)

    def _current_challenge(self) -> ChallengeType | None:
        if self._status != LivenessResult.IN_PROGRESS:
            return None
        if self._stage_index >= len(self._sequence):
            return None
        return self._sequence[self._stage_index]

    def _current_prompt(self) -> str:
        challenge = self._current_challenge()
        if challenge is None:
            if self._status == LivenessResult.LIVENESS_LOCKED:
                return "Liveness Locked"
            if self._status == LivenessResult.PASSED:
                return "Liveness Passed"
            if self._status == LivenessResult.FAILED:
                return "Liveness Failed"
            return "No Active Challenge"

        if challenge == ChallengeType.BLINK:
            return "Blink Now"
        if challenge == ChallengeType.TURN_LEFT:
            return "Turn Left"
        if challenge == ChallengeType.TURN_RIGHT:
            return "Turn Right"
        if challenge == ChallengeType.MOUTH_OPEN:
            return "Open Your Mouth"
        if challenge == ChallengeType.SMILE:
            return "Smile Now"
        return "Nod Your Head"

    def _seconds_remaining(self, timestamp_s: float) -> float:
        if self._stage_start_ts is None:
            return self.timeout_per_stage_s
        return max(0.0, self.timeout_per_stage_s - (timestamp_s - self._stage_start_ts))

    def _advance_stage(self, timestamp_s: float) -> None:
        self._stage_index += 1
        self._stage_start_ts = timestamp_s
        self._blink_closed_seen = False
        self._nod_motion_seen = False
        self._mouth_baseline_ratio = None
        self._nod_baseline_ratio = None
        self._smile_baseline_ratio = None
        if self._stage_index >= len(self._sequence):
            self._status = LivenessResult.PASSED

    def update(self, landmarks: Sequence | None, timestamp_s: float) -> LivenessState:
        if self._status == LivenessResult.LIVENESS_LOCKED:
            self.start()

        if self._status in {LivenessResult.PASSED, LivenessResult.FAILED}:
            return LivenessState(
                status=self._status,
                prompt=self._current_prompt(),
                stage_index=min(self._stage_index, len(self._sequence)),
                total_stages=len(self._sequence),
                seconds_remaining=0.0,
            )

        if landmarks is None:
            if self._face_missing_since_ts is None:
                self._face_missing_since_ts = timestamp_s
            missing_elapsed = timestamp_s - self._face_missing_since_ts
            if missing_elapsed >= self.face_missing_grace_s:
                self._status = LivenessResult.FAILED
                return LivenessState(
                    status=self._status,
                    prompt="Face Lost - Liveness Failed",
                    stage_index=self._stage_index,
                    total_stages=len(self._sequence),
                    seconds_remaining=0.0,
                )
            return LivenessState(
                status=LivenessResult.IN_PROGRESS,
                prompt="Keep Face Centered",
                stage_index=self._stage_index,
                total_stages=len(self._sequence),
                seconds_remaining=self._seconds_remaining(timestamp_s),
            )

        self._face_missing_since_ts = None

        if self._stage_start_ts is None:
            self._stage_start_ts = timestamp_s

        remaining = self._seconds_remaining(timestamp_s)
        if remaining <= 0.0:
            self._status = LivenessResult.FAILED
            return LivenessState(
                status=self._status,
                prompt="Stage Timeout - Liveness Failed",
                stage_index=self._stage_index,
                total_stages=len(self._sequence),
                seconds_remaining=0.0,
            )

        challenge = self._current_challenge()
        if challenge is None:
            self._status = LivenessResult.FAILED
            return LivenessState(
                status=self._status,
                prompt="Challenge State Error",
                stage_index=self._stage_index,
                total_stages=len(self._sequence),
                seconds_remaining=0.0,
            )

        if challenge == ChallengeType.BLINK:
            if self.check_blink(landmarks):
                self._advance_stage(timestamp_s)
        elif challenge in {ChallengeType.TURN_LEFT, ChallengeType.TURN_RIGHT}:
            matched, direction = self.check_head_turn(landmarks)
            if matched:
                if challenge == ChallengeType.TURN_LEFT and direction == "LEFT":
                    self._advance_stage(timestamp_s)
                elif challenge == ChallengeType.TURN_RIGHT and direction == "RIGHT":
                    self._advance_stage(timestamp_s)
        elif challenge == ChallengeType.MOUTH_OPEN:
            if self.check_mouth_open(landmarks):
                self._advance_stage(timestamp_s)
        elif challenge == ChallengeType.NOD_HEAD:
            if self.check_nod_head(landmarks):
                self._advance_stage(timestamp_s)
        elif challenge == ChallengeType.SMILE:
            if self.check_smile(landmarks):
                self._advance_stage(timestamp_s)

        return LivenessState(
            status=self._status,
            prompt=self._current_prompt(),
            stage_index=min(self._stage_index, len(self._sequence)),
            total_stages=len(self._sequence),
            seconds_remaining=self._seconds_remaining(timestamp_s)
            if self._status == LivenessResult.IN_PROGRESS
            else 0.0,
        )
