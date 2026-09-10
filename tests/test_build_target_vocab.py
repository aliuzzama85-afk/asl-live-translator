"""Tests for pose_library.build_target_vocab."""

import json

from pose_library.build_target_vocab import build_target_vocab, save_target_vocab


def test_build_target_vocab_splits_matched_and_unmatched(tmp_path):
    vocab_csv = tmp_path / "vocab.csv"
    vocab_csv.write_text(
        "text,gloss\n"
        '"I am hungry .","X-I BE DESC-HUNGRY ."\n'
        '"I need an airport .","X-I NEED AIRPORT ."\n',
        encoding="utf-8",
    )
    wlasl_metadata = tmp_path / "WLASL_v0.3.json"
    wlasl_metadata.write_text(
        json.dumps(
            [{"gloss": "hungry", "instances": []}, {"gloss": "need", "instances": []}]
        ),
        encoding="utf-8",
    )

    result = build_target_vocab(vocab_csv, wlasl_metadata)

    assert result["matched"] == ["hungry", "need"]
    assert result["unmatched"] == ["airport", "be"]


def test_save_target_vocab_writes_readable_json(tmp_path):
    output_path = tmp_path / "target_vocab.json"
    save_target_vocab({"matched": ["hungry"], "unmatched": ["airport"]}, output_path)

    loaded = json.loads(output_path.read_text(encoding="utf-8"))
    assert loaded == {"matched": ["hungry"], "unmatched": ["airport"]}
