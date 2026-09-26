# Fingerspelling alphabet

**Status: 24 of 26 letters available; J and Z missing.**

- **24 letters (A–Z except J, Z)** are converted from the MIT-licensed
  [`sid220/asl-now-fingerspelling`](https://huggingface.co/datasets/sid220/asl-now-fingerspelling)
  dataset by `python -m pose_library.convert_hf_fingerspelling`. License and
  provenance: [`THIRD_PARTY_LICENSE_asl-now-fingerspelling.md`](THIRD_PARTY_LICENSE_asl-now-fingerspelling.md).
- **J and Z are missing**: they're motion letters, and the dataset only has
  single still frames. Words containing them are skipped with "NO
  FINGERSPELLING FOR "J"". To fill them in, record them:
  `python -m pose_library.record_fingerspelling --letters jz`, then
  `python -m pose_library.build_fingerspelling`.

Layout:

- `poses/<letter>.json` + `poses/manifest.json`: the letter library, the
  same `PoseSequence` shape as the WLASL word library. Each manifest entry's
  `source` says where the letter came from (`hf:sid220/asl-now-fingerspelling`
  or `fingerspelling:self-recorded`).
- `raw/<letter>.mp4`: webcam takes for recorded letters only (none yet).

Unlike `pose_library/data/` (WLASL-derived, gitignored, local-only), this
directory **is committed**. Design and every decision:
[`../FINGERSPELLING_PLAN.md`](../FINGERSPELLING_PLAN.md) (Section 2b for the
dataset).

Synthetic placeholder letters used for plumbing tests live under
`tests/fixtures/fingerspelling_synthetic/`, never here.
