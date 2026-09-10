"""MediaPipe Holistic Landmarker extraction: video file in, `PoseSequence` out.

Runs MediaPipe's **Holistic Landmarker** (not Hands alone) per
`pose_library/PLAN.md` Section 2, but only persists a lightweight subset of
its output for this milestone: both hands' 21 landmarks each, plus a
minimal upper-body pose subset (shoulders/elbows/wrists, not the full 33
pose landmarks). Face landmarks/blendshapes are an intentional future
add-on (see PLAN.md Section 2's compute-vs-flexibility rationale), not
extracted here.

Frames where neither hand is detected are dropped rather than filled with
invented/interpolated values (PLAN.md Section 2, point 4) -- gaps are
recorded in `ExtractionStats` for the caller to persist alongside the
`PoseSequence` (e.g. in `manifest.json`), not silently discarded.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

from pose_library import config
from pose_library.types import PoseSequence

logger = logging.getLogger(__name__)

# The 21 MediaPipe hand landmarks, in their canonical index order. Hand
# Landmarker and Holistic Landmarker share this indexing.
HAND_LANDMARK_NAMES: tuple[str, ...] = (
    "wrist",
    "thumb_cmc",
    "thumb_mcp",
    "thumb_ip",
    "thumb_tip",
    "index_finger_mcp",
    "index_finger_pip",
    "index_finger_dip",
    "index_finger_tip",
    "middle_finger_mcp",
    "middle_finger_pip",
    "middle_finger_dip",
    "middle_finger_tip",
    "ring_finger_mcp",
    "ring_finger_pip",
    "ring_finger_dip",
    "ring_finger_tip",
    "pinky_mcp",
    "pinky_pip",
    "pinky_dip",
    "pinky_tip",
)

# Indices into MediaPipe Pose's full 33-landmark output for the minimal
# upper-body subset this project persists (PLAN.md Section 2: "shoulders/
# elbows/wrists only, not all 33, to keep file size down").
_POSE_SUBSET_INDICES: tuple[int, ...] = (11, 12, 13, 14, 15, 16)
POSE_SUBSET_NAMES: tuple[str, ...] = (
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
)

LANDMARK_NAMES: tuple[str, ...] = (
    tuple(f"left_hand_{name}" for name in HAND_LANDMARK_NAMES)
    + tuple(f"right_hand_{name}" for name in HAND_LANDMARK_NAMES)
    + POSE_SUBSET_NAMES
)

_ZERO_POINT: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class ExtractionStats:
    """Per-video extraction bookkeeping.

    Stored alongside (not inside) the `PoseSequence` -- e.g. in
    `manifest.json` -- since it's provenance/quality metadata, not part of
    the fixed `PoseSequence` schema.

    Attributes:
        total_frames_decoded: Number of frames read from the source video.
        frames_kept: Number of frames kept in the output `PoseSequence`
            (at least one hand detected).
        dropped_frame_indices: Original (0-indexed) frame numbers dropped
            because neither hand was detected -- an honest gap signal for a
            later smoothing/interpolation stage, per PLAN.md Section 2.
    """

    total_frames_decoded: int
    frames_kept: int
    dropped_frame_indices: list[int]


def _make_landmarker(model_path: Path) -> vision.HolisticLandmarker:
    """Builds a video-mode Holistic Landmarker from a local `.task` model file.

    Args:
        model_path: Path to the downloaded `holistic_landmarker.task` bundle.

    Returns:
        A `HolisticLandmarker` configured for `running_mode=VIDEO`.
    """
    options = vision.HolisticLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
    )
    return vision.HolisticLandmarker.create_from_options(options)


def _hand_points(hand_landmarks) -> list[tuple[float, float, float]] | None:
    """Converts one hand's MediaPipe landmarks to a plain point list.

    Args:
        hand_landmarks: `result.left_hand_landmarks` or
            `result.right_hand_landmarks` from a Holistic Landmarker result.

    Returns:
        A list of 21 `(x, y, z)` tuples, or `None` if the hand wasn't
        detected in this frame.
    """
    if not hand_landmarks:
        return None
    return [(lm.x, lm.y, lm.z) for lm in hand_landmarks]


def _pose_subset_points(pose_landmarks) -> list[tuple[float, float, float]]:
    """Extracts the shoulders/elbows/wrists subset from a pose result.

    Args:
        pose_landmarks: `result.pose_landmarks` from a Holistic Landmarker
            result (33 landmarks, or empty if no body was detected).

    Returns:
        A list of 6 `(x, y, z)` tuples, one per `POSE_SUBSET_NAMES` entry.
        Zero-filled if pose wasn't detected -- this is a best-effort
        secondary signal; frame-keep/drop decisions are driven by hand
        detection only (see module docstring), not pose detection.
    """
    if not pose_landmarks:
        return [_ZERO_POINT] * len(_POSE_SUBSET_INDICES)
    return [
        (pose_landmarks[i].x, pose_landmarks[i].y, pose_landmarks[i].z)
        for i in _POSE_SUBSET_INDICES
    ]


def extract_pose_sequence(
    video_path: Path,
    gloss: str,
    source: str,
    model_path: Path = config.HOLISTIC_MODEL_PATH,
) -> tuple[PoseSequence, ExtractionStats]:
    """Extracts a `PoseSequence` from a trimmed WLASL clip.

    Args:
        video_path: Path to the trimmed, gloss-relevant video clip.
        gloss: The gloss word this clip represents (stored uppercase in the
            returned `PoseSequence`, matching gloss_model's output
            convention).
        source: Attribution/provenance string (e.g. "wlasl:<source>:
            <video_id>"), stored on the returned `PoseSequence`.
        model_path: Path to the Holistic Landmarker `.task` model bundle.

    Returns:
        A tuple of `(PoseSequence, ExtractionStats)`. The `PoseSequence`
        contains only frames where at least one hand was detected;
        `ExtractionStats` records how many frames were decoded/kept/dropped
        so a caller can flag gaps in the manifest (PLAN.md Section 2/3).

    Raises:
        FileNotFoundError: If `video_path` or `model_path` doesn't exist.
        ValueError: If no frames could be decoded from `video_path` at all.
    """
    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Holistic Landmarker model not found: {model_path}")

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    ms_per_frame = 1000.0 / fps

    landmarker = _make_landmarker(model_path)
    frames: list[list[tuple[float, float, float]]] = []
    dropped_frame_indices: list[int] = []
    total_decoded = 0

    try:
        frame_idx = 0
        while True:
            ok, bgr_frame = cap.read()
            if not ok:
                break
            total_decoded += 1

            rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int(frame_idx * ms_per_frame)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            left_hand = _hand_points(result.left_hand_landmarks)
            right_hand = _hand_points(result.right_hand_landmarks)
            if left_hand is None and right_hand is None:
                dropped_frame_indices.append(frame_idx)
                frame_idx += 1
                continue

            left_hand = left_hand or [_ZERO_POINT] * len(HAND_LANDMARK_NAMES)
            right_hand = right_hand or [_ZERO_POINT] * len(HAND_LANDMARK_NAMES)
            pose_subset = _pose_subset_points(result.pose_landmarks)
            frames.append(left_hand + right_hand + pose_subset)
            frame_idx += 1
    finally:
        cap.release()
        landmarker.close()

    if total_decoded == 0:
        raise ValueError(f"No frames could be decoded from {video_path}")

    stats = ExtractionStats(
        total_frames_decoded=total_decoded,
        frames_kept=len(frames),
        dropped_frame_indices=dropped_frame_indices,
    )
    sequence = PoseSequence(
        gloss=gloss.upper(),
        fps=fps,
        landmark_names=LANDMARK_NAMES,
        frames=frames,
        source=source,
    )
    return sequence, stats
