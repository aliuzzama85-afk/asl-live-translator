"""Tests for pose_library.convert_hf_fingerspelling.

Run against real dataset files: `tests/fixtures/asl_now_sample/` is a small,
unmodified subset of sid220/asl-now-fingerspelling (MIT; see that folder's
README). Where a test needs a deliberately broken or outlier sample, it
builds one in `tmp_path` from a real sample and says so.
"""

import json
import math
import shutil
from pathlib import Path

import numpy as np
import pytest

from pose_library.build_fingerspelling import build_alphabet
from pose_library.convert_hf_fingerspelling import (
    DATASET_SOURCE,
    HOLD_FRAMES,
    Sample,
    convert_dataset,
    convert_letter,
    is_in_frame,
    load_letter_samples,
    normalize_shape,
    parse_sample,
    sample_to_frame,
    select_representative,
)
from pose_library.extract import LANDMARK_NAMES
from pose_library.fingerspelling import FINGERSPELLING_SOURCE, hand_indices
from pose_library.manifest import (
    ASL_NOW_LICENSE_TAG,
    build_letter_entry,
    load_manifest,
    save_manifest,
)

FIXTURE = Path(__file__).parent / "fixtures" / "asl_now_sample"
OUT_OF_FRAME = "C/b2962280-fb9d-41c2-9f94-51a8e9f21c4a.json"
HANDS = hand_indices(LANDMARK_NAMES)


def _raw(rel):
    return json.loads((FIXTURE / rel).read_text(encoding="utf-8"))


def _real_a_samples():
    samples, _ = load_letter_samples(FIXTURE, "a")
    return samples


@pytest.fixture()
def dataset(tmp_path):
    """A writable copy of the real fixture subset."""
    root = tmp_path / "dataset"
    shutil.copytree(FIXTURE, root, ignore=shutil.ignore_patterns("*.md"))
    return root


# --- parsing / filtering (real files) ---------------------------------------


def test_every_real_fixture_file_matches_the_documented_format():
    for path in FIXTURE.glob("*/*.json"):
        points = parse_sample(json.loads(path.read_text(encoding="utf-8")))
        assert points is not None and len(points) == 21, path


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d[:20],  # too few landmarks
        lambda d: d + [d[0]],  # too many (e.g. a second frame appended)
        lambda d: [{"x": p["x"], "y": p["y"]} for p in d],  # missing z
        lambda d: [{**d[0], "x": float("nan")}] + d[1:],
        lambda d: [{**d[0], "x": "0.5"}] + d[1:],
        lambda d: [{**d[0], "x": True}] + d[1:],
        lambda d: {"frames": d},  # not a list
    ],
)
def test_parse_sample_rejects_malformed_variants_of_a_real_sample(mutate):
    real = _raw("A/" + min(p.name for p in (FIXTURE / "A").iterdir()))
    assert parse_sample(mutate(real)) is None


def test_a_real_out_of_frame_sample_is_detected():
    assert not is_in_frame(parse_sample(_raw(OUT_OF_FRAME)))
    assert all(is_in_frame(s.points) for s in _real_a_samples())


def test_load_reports_invalid_and_out_of_frame_samples(dataset):
    (dataset / "C" / "broken.json").write_text("[1, 2, 3]", encoding="utf-8")
    samples, report = load_letter_samples(dataset, "c")
    assert samples == []
    assert report.total == 2
    assert report.out_of_frame == [OUT_OF_FRAME]
    assert report.invalid == ["C/broken.json"]


# --- shape normalization ----------------------------------------------------


def test_normalize_shape_removes_position_and_size_but_keeps_orientation():
    points = _real_a_samples()[0].points
    base = normalize_shape(points)
    assert np.allclose(base[0], 0.0)
    assert np.linalg.norm(base[9]) == pytest.approx(1.0)

    moved = [(0.3 + 0.5 * x, 0.1 + 0.5 * y, z) for x, y, z in points]
    assert np.allclose(normalize_shape(moved), base)

    wx, wy, _ = points[0]
    rotated = [(wx - (y - wy), wy + (x - wx), z) for x, y, z in points]  # 90 degrees
    assert not np.allclose(normalize_shape(rotated), base)


# --- representative-sample selection ----------------------------------------


def test_selection_is_independent_of_file_order():
    samples = _real_a_samples()
    chosen = samples[select_representative(samples)[0]]
    reversed_samples = list(reversed(samples))
    assert reversed_samples[select_representative(reversed_samples)[0]] is chosen


def test_selection_picks_the_medoid_not_the_first_file():
    samples = _real_a_samples()
    shapes = [normalize_shape(s.points) for s in samples]
    medians = [
        np.median(
            [
                np.linalg.norm(shapes[i] - shapes[j], axis=1).mean()
                for j in range(len(shapes))
                if j != i
            ]
        )
        for i in range(len(shapes))
    ]
    index, distance = select_representative(samples)
    assert index == int(np.argmin(medians))
    assert distance == pytest.approx(min(medians))


def test_an_outlier_is_never_chosen_even_when_it_sorts_first():
    samples = _real_a_samples()
    before = samples[select_representative(samples)[0]]
    # Built for this test: a real A sample with its fingers scrambled, named to
    # sort first -- the kind of mis-signed/badly tracked sample to avoid.
    real = samples[0].points
    scrambled = [real[0]] + list(reversed(real[1:]))
    outlier = Sample(path="A/000-outlier.json", points=scrambled)
    with_outlier = [outlier] + samples
    after = with_outlier[select_representative(with_outlier)[0]]
    assert after is not outlier
    assert after is before


def test_selection_handles_a_single_sample_and_rejects_none():
    only = _real_a_samples()[:1]
    assert select_representative(only) == (0, 0.0)
    with pytest.raises(ValueError):
        select_representative([])


# --- frame placement --------------------------------------------------------


def test_sample_to_frame_places_the_hand_unchanged_and_zero_fills_the_rest():
    points = _real_a_samples()[0].points
    frame = sample_to_frame(points)
    assert len(frame) == len(LANDMARK_NAMES)
    assert [frame[i] for i in HANDS["right"]] == points
    others = [p for i, p in enumerate(frame) if i not in set(HANDS["right"])]
    assert all(p == (0.0, 0.0, 0.0) for p in others)


# --- the J/Z decision -------------------------------------------------------


@pytest.mark.parametrize("letter", ["j", "z"])
def test_motion_letters_are_not_converted_even_with_samples_present(
    dataset, tmp_path, letter
):
    assert list((dataset / "J").glob("*.json"))  # the dataset does have J files
    library = {}
    result = convert_letter(letter, dataset, library, tmp_path / "poses")
    assert not result.converted
    assert (
        "motion" in result.reason and "not converted rather than faked" in result.reason
    )
    assert letter not in library
    assert not (tmp_path / "poses" / f"{letter}.json").exists()


# --- end-to-end conversion + manifest ---------------------------------------


def test_convert_dataset_writes_real_held_poses_with_full_provenance(dataset, tmp_path):
    poses = tmp_path / "poses"
    results = {r.letter: r for r in convert_dataset(dataset, poses)}

    assert results["a"].converted and results["a"].usable == 8
    assert not results["c"].converted  # its only fixture sample is out of frame
    assert not results["j"].converted
    assert not results["b"].converted and results["b"].reason == "no usable samples"

    manifest = load_manifest(poses / "manifest.json")
    assert set(manifest) == {"a"}
    entry = manifest["a"]
    assert entry["source"] == DATASET_SOURCE == "hf:sid220/asl-now-fingerspelling"
    assert (
        entry["source_url"]
        == "https://huggingface.co/datasets/sid220/asl-now-fingerspelling"
    )
    assert entry["license"] == ASL_NOW_LICENSE_TAG and "MIT" in entry["license"]
    assert entry["kind"] == "static"
    assert entry["frames_kept"] == entry["total_frames_decoded"] == HOLD_FRAMES == 12
    assert entry["dropped_frame_indices"] == []
    assert entry["low_confidence"] is False
    sample = entry["dataset_sample"]
    assert sample["revision"] == "9b3c96ae0adb7744a2c9fc72692842e6b3e25e33"
    assert sample["samples_total"] == sample["samples_usable"] == 8
    assert (FIXTURE / sample["path"]).is_file()
    # Never presented as self-recorded or synthetic.
    assert entry["source"] != FINGERSPELLING_SOURCE
    assert "synthetic" not in json.dumps(entry)

    stored = json.loads((poses / "a.json").read_text(encoding="utf-8"))
    assert stored["gloss"] == "A"
    assert stored["source"] == f"{DATASET_SOURCE}:{sample['path']}"
    assert stored["landmark_names"] == list(LANDMARK_NAMES)
    assert len(stored["frames"]) == HOLD_FRAMES
    assert all(f == stored["frames"][0] for f in stored["frames"])
    raw = [[p["x"], p["y"], p["z"]] for p in _raw(sample["path"])]
    assert [stored["frames"][0][i] for i in HANDS["right"]] == raw  # values unaltered
    assert all(not math.isclose(sum(map(abs, p)), 0.0) for p in raw)


def _self_recorded_entry(letter):
    return build_letter_entry(
        letter=letter,
        kind="static",
        signing_hand="right",
        source=FINGERSPELLING_SOURCE,
        recording_total_frames=60,
        frames_with_hand=60,
        segment_start_frame=10,
        total_frames_decoded=12,
        frames_kept=12,
        dropped_frame_indices=[],
        issues=[],
    )


def test_conversion_keeps_self_recorded_letters_and_merges_the_rest(dataset, tmp_path):
    poses = tmp_path / "poses"
    poses.mkdir()
    save_manifest(
        {"a": _self_recorded_entry("a"), "j": _self_recorded_entry("j")},
        poses / "manifest.json",
    )
    results = {r.letter: r for r in convert_dataset(dataset, poses)}

    assert results["a"].reason == "self-recorded version kept"
    manifest = load_manifest(poses / "manifest.json")
    assert manifest["a"]["source"] == FINGERSPELLING_SOURCE
    assert manifest["j"]["source"] == FINGERSPELLING_SOURCE


def test_a_failed_recorded_build_never_deletes_a_dataset_letter(dataset, tmp_path):
    poses = tmp_path / "poses"
    convert_dataset(dataset, poses)
    raw = tmp_path / "raw"
    raw.mkdir()  # no recording of A exists

    results = build_alphabet(["a"], raw_dir=raw, poses_dir=poses)

    assert not results[0].success and results[0].error == "not recorded"
    assert load_manifest(poses / "manifest.json")["a"]["source"] == DATASET_SOURCE
    assert (poses / "a.json").is_file()
