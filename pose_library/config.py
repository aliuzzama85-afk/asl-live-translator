"""Configuration constants for the gloss-to-pose lookup library (Stage 4).

Single source of truth for on-disk paths, so `download.py`, `extract.py`,
`lookup.py`, and `build_target_vocab.py` stay in sync. See
`pose_library/PLAN.md` for the design rationale behind each path.
"""

from pathlib import Path

POSE_LIBRARY_DIR = Path(__file__).resolve().parent

# --- Gitignored, WLASL-derived / large local data (see PLAN.md Section 1/3) ---
DATA_DIR = POSE_LIBRARY_DIR / "data"

# Cached copy of https://raw.githubusercontent.com/dxli94/WLASL/master/
# start_kit/WLASL_v0.3.json -- metadata-only (no video bytes), see PLAN.md
# Section 1.
WLASL_METADATA_PATH = DATA_DIR / "wlasl" / "WLASL_v0.3.json"

# Reproducible, inspectable output of build_target_vocab.py: the resolved
# ~150-250 word target vocabulary (vocab_augmentation.csv stems that have a
# direct WLASL gloss match), not just implicit in code.
TARGET_VOCAB_PATH = DATA_DIR / "target_vocab.json"

# Raw (untrimmed) and trimmed WLASL video clips downloaded by download.py.
RAW_VIDEOS_DIR = DATA_DIR / "raw_videos"
TRIMMED_VIDEOS_DIR = DATA_DIR / "trimmed_videos"

# One JSON file per gloss word (PoseSequence), plus a small manifest.json
# recording word -> filename, WLASL source/video_id/instance_id, license tag,
# and extraction stats (dropped-frame gaps) -- PLAN.md Section 3.
POSES_DIR = DATA_DIR / "poses"
MANIFEST_PATH = POSES_DIR / "manifest.json"

# MediaPipe Holistic Landmarker task bundle (downloaded once, cached locally;
# not committed -- it's a large binary model file, not project code).
HOLISTIC_MODEL_PATH = DATA_DIR / "models" / "holistic_landmarker.task"
HOLISTIC_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/"
    "holistic_landmarker/float16/latest/holistic_landmarker.task"
)

WLASL_METADATA_URL = (
    "https://raw.githubusercontent.com/dxli94/WLASL/master/" "start_kit/WLASL_v0.3.json"
)

# --- Committed, self-recorded fingerspelling alphabet (FINGERSPELLING_PLAN.md) ---
# NOT gitignored, unlike DATA_DIR above: project-owned recordings with no
# WLASL licensing question. `.gitignore` carries an explicit negation for
# `raw/*.mp4` below its global `*.mp4` rule.
FINGERSPELLING_DIR = POSE_LIBRARY_DIR / "fingerspelling"
FINGERSPELLING_RAW_DIR = FINGERSPELLING_DIR / "raw"
FINGERSPELLING_POSES_DIR = FINGERSPELLING_DIR / "poses"
FINGERSPELLING_MANIFEST_PATH = FINGERSPELLING_POSES_DIR / "manifest.json"

# --- Third-party fingerspelling source: sid220/asl-now-fingerspelling (MIT) ---
# License record: pose_library/fingerspelling/THIRD_PARTY_LICENSE_asl-now-fingerspelling.md.
# Pinned to one dataset revision so a conversion is reproducible (the card
# says it "will be updated frequently"). The raw download is a re-fetchable
# cache under the gitignored DATA_DIR; only the converted letters are
# committed.
ASL_NOW_DATASET_ID = "sid220/asl-now-fingerspelling"
ASL_NOW_REVISION = "9b3c96ae0adb7744a2c9fc72692842e6b3e25e33"
ASL_NOW_SOURCE_URL = "https://huggingface.co/datasets/sid220/asl-now-fingerspelling"
ASL_NOW_CACHE_DIR = DATA_DIR / "asl_now_fingerspelling"

# vocab_augmentation.csv gloss column is the input to build_target_vocab.py
# (see gloss_model/config.py's AUGMENTATION_DATA_PATH for the same file used
# by Stage 3 -- reused here rather than duplicated, per CLAUDE.md).
VOCAB_AUGMENTATION_CSV_PATH = (
    POSE_LIBRARY_DIR.parent / "gloss_model" / "data" / "vocab_augmentation.csv"
)
