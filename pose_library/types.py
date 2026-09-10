"""The `PoseSequence` dataclass shared by extraction, storage, and lookup.

See `pose_library/PLAN.md` Section 4 for the design rationale (return type,
serialization shape, why a small dataclass over a bare array/dict).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PoseSequence:
    """A single sign's extracted keypoint sequence.

    Attributes:
        gloss: The gloss word this sequence represents (uppercase, matching
            gloss_model's output convention).
        fps: Frame rate the sequence was extracted/should be replayed at.
        landmark_names: Ordered names/indices for each landmark column, so
            frame data is self-describing rather than a bare positional array.
        frames: One entry per frame; each frame is a list of (x, y, z)
            tuples (normalized image coordinates), one per landmark_names
            entry, in the same order.
        source: Attribution/provenance string (e.g. WLASL video_id + source
            site, or "fingerspelling:self-recorded"), kept for the C-UDA
            attribution requirement noted in PLAN.md Section 1/3.
    """

    gloss: str
    fps: float
    landmark_names: tuple[str, ...]
    frames: list[list[tuple[float, float, float]]]
    source: str

    def to_dict(self) -> dict[str, Any]:
        """Converts this sequence to a plain, JSON-serializable dict.

        Returns:
            A dict with the same fields as this dataclass, with tuples
            converted to lists (JSON has no tuple type).
        """
        return {
            "gloss": self.gloss,
            "fps": self.fps,
            "landmark_names": list(self.landmark_names),
            "frames": [[list(point) for point in frame] for frame in self.frames],
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PoseSequence:
        """Builds a `PoseSequence` from a dict produced by `to_dict`/`json.load`.

        Args:
            data: A dict with "gloss", "fps", "landmark_names", "frames", and
                "source" keys, as produced by `to_dict`.

        Returns:
            The reconstructed `PoseSequence`.
        """
        return cls(
            gloss=data["gloss"],
            fps=data["fps"],
            landmark_names=tuple(data["landmark_names"]),
            frames=[[tuple(point) for point in frame] for frame in data["frames"]],
            source=data["source"],
        )

    def as_array(self) -> Any:
        """Returns this sequence's frames as a numpy array.

        Returns:
            An array of shape `(num_frames, num_landmarks, 3)`. Requires
            numpy, which is not otherwise a hard dependency of this module --
            only imported here, lazily, for callers that want it (e.g. a
            renderer or smoothing stage).
        """
        import numpy as np

        return np.array(self.frames, dtype=float)
