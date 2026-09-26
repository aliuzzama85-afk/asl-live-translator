# Fingerspelling alphabet

**Status: all 26 letters available.**

- **24 letters (A–Z except J, Z)** are converted from the MIT-licensed
  [`sid220/asl-now-fingerspelling`](https://huggingface.co/datasets/sid220/asl-now-fingerspelling)
  dataset by `python -m pose_library.convert_hf_fingerspelling`. License and
  provenance: [`THIRD_PARTY_LICENSE_asl-now-fingerspelling.md`](THIRD_PARTY_LICENSE_asl-now-fingerspelling.md).
- **J and Z are self-recorded** (`raw/j.mp4`, `raw/z.mp4`): they're motion
  letters, and the dataset only has single still frames. Recorded with
  `python -m pose_library.record_fingerspelling --letters jz` and built with
  `python -m pose_library.build_fingerspelling`. The same two commands
  re-record any letter; a recording replaces the dataset version.

Layout:

- `poses/<letter>.json` + `poses/manifest.json`: the letter library, the
  same `PoseSequence` shape as the WLASL word library. Each manifest entry's
  `source` says where the letter came from (`hf:sid220/asl-now-fingerspelling`
  or `fingerspelling:self-recorded`).
- `raw/<letter>.mp4`: webcam takes for recorded letters only (J and Z).

Unlike `pose_library/data/` (WLASL-derived, gitignored, local-only), this
directory **is committed**. Design and every decision:
[`../FINGERSPELLING_PLAN.md`](../FINGERSPELLING_PLAN.md) (Section 2b for the
dataset).

Synthetic placeholder letters used for plumbing tests live under
`tests/fixtures/fingerspelling_synthetic/`, never here.
