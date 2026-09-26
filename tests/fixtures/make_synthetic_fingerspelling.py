"""Generates SYNTHETIC placeholder fingerspelling letters for manual testing.

**These are not real ASL letters.** They're geometric stand-in hands (a wrist
and five straight or curled finger chains), hand-constructed so the
fingerspelling pipeline can be exercised in a real browser before the real
26 letters are recorded (`pose_library/FINGERSPELLING_PLAN.md` Section 7).
The handshapes are arbitrary and must never be mistaken for, shipped as, or
copied into `pose_library/fingerspelling/`.

Output goes to `tests/fixtures/fingerspelling_synthetic/`: one `<letter>.json`
in the real `PoseSequence` shape (the real 48 `LANDMARK_NAMES`) plus a
`manifest.json` built with the real `build_letter_entry`, so the files are
structurally identical to what `build_fingerspelling.py` writes.

Usage (from the repo root, with the venv active):

    python -m tests.fixtures.make_synthetic_fingerspelling

Then point the dev server at it:

    FINGERSPELLING_POSES_DIR=../tests/fixtures/fingerspelling_synthetic npm run dev
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from pose_library.extract import LANDMARK_NAMES
from pose_library.fingerspelling import LETTER_HOLD_SECONDS, hand_indices
from pose_library.manifest import build_letter_entry, save_manifest
from pose_library.types import PoseSequence

OUTPUT_DIR = Path(__file__).resolve().parent / "fingerspelling_synthetic"
FPS = 30.0

# Same field shapes as real output, but values that say what these are --
# the source line reads "SOURCE: synthetic:placeholder" in the app.
SYNTHETIC_SOURCE = "synthetic:placeholder"
SYNTHETIC_LICENSE = (
    "SYNTHETIC PLACEHOLDER - not a real recording, not a real ASL letter"
)


# Per finger (thumb, index, middle, ring, pinky): 0 = straight, 1 = fully
# curled. Arbitrary shapes, chosen only to look *different* from each other.
SYNTHETIC_SHAPES = {
    "a": (0.2, 1.0, 1.0, 1.0, 1.0),
    "b": (1.0, 0.0, 0.0, 0.0, 0.0),
    "c": (0.4, 0.5, 0.5, 0.5, 0.5),
    "d": (0.6, 0.0, 1.0, 1.0, 1.0),
    "e": (0.9, 0.8, 0.8, 0.8, 0.8),
    "j": (0.8, 1.0, 1.0, 1.0, 0.0),  # motion letter: moved along a hook below
}

_FINGER_BASE_ANGLES = (-2.4, -1.85, -1.6, -1.35, -1.1)  # radians, fanned upward
_SEGMENT_LENGTHS = (0.035, 0.03, 0.025, 0.02)


def _hand_points(wrist: tuple[float, float], curls: tuple[float, ...]) -> list:
    """21 MediaPipe-ordered points for a synthetic right hand."""
    points = [(wrist[0], wrist[1], 0.0)]
    for angle0, curl in zip(_FINGER_BASE_ANGLES, curls):
        x, y, angle = wrist[0], wrist[1], angle0
        for segment, length in enumerate(_SEGMENT_LENGTHS):
            if segment > 0:
                angle += curl * 1.1  # each joint bends toward the palm
            x += length * math.cos(angle)
            y += length * math.sin(angle)
            points.append((x, y, 0.0))
    return points


def _frames(letter: str, count: int) -> list:
    hands = hand_indices(LANDMARK_NAMES)
    frames = []
    for k in range(count):
        if letter == "j":
            # A "J" hook: down, then curving left.
            t = k / (count - 1)
            wrist = (0.55 - 0.08 * max(0.0, t - 0.5) * 2, 0.55 + 0.12 * min(t * 2, 1.0))
        else:
            # A small, deterministic micro-motion so a hold doesn't look frozen.
            wrist = (0.55 + 0.002 * math.sin(k), 0.62 + 0.002 * math.cos(k))
        frame = [(0.0, 0.0, 0.0)] * len(LANDMARK_NAMES)
        for i, point in zip(
            hands["right"], _hand_points(wrist, SYNTHETIC_SHAPES[letter])
        ):
            frame[i] = point
        frames.append(frame)
    return frames


def main() -> None:
    """Writes the synthetic letters and their manifest."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    library = {}
    for letter in SYNTHETIC_SHAPES:
        count = 20 if letter == "j" else round(LETTER_HOLD_SECONDS * FPS)
        sequence = PoseSequence(
            gloss=letter.upper(),
            fps=FPS,
            landmark_names=LANDMARK_NAMES,
            frames=_frames(letter, count),
            source=SYNTHETIC_SOURCE,
        )
        with (OUTPUT_DIR / f"{letter}.json").open("w", encoding="utf-8") as f:
            json.dump(sequence.to_dict(), f)
        library[letter] = build_letter_entry(
            letter=letter,
            kind="motion" if letter == "j" else "static",
            signing_hand="right",
            source=SYNTHETIC_SOURCE,
            recording_total_frames=60,
            frames_with_hand=60,
            segment_start_frame=20,
            total_frames_decoded=count,
            frames_kept=count,
            dropped_frame_indices=[],
            issues=[],
        )
        library[letter]["license"] = SYNTHETIC_LICENSE
    save_manifest(library, OUTPUT_DIR / "manifest.json")
    print(f"Wrote {len(library)} SYNTHETIC letters to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
