# Fingerspelling alphabet (self-recorded)

**Status: not recorded yet.** This directory holds the project's own A–Z
fingerspelling recordings and the poses extracted from them. Until someone
records the letters, `raw/` is empty and `poses/` doesn't exist, and the
app shows "FINGERSPELLING ALPHABET NOT RECORDED YET" for any word outside the
pose library.

- `raw/<letter>.mp4`: one short webcam take per letter, written by
  `python -m pose_library.record_fingerspelling`.
- `poses/<letter>.json` + `poses/manifest.json`: written by
  `python -m pose_library.build_fingerspelling`, same `PoseSequence` shape
  as the WLASL word library.

Unlike `pose_library/data/` (WLASL-derived, gitignored, local-only), this is
project-owned content and **is committed**. Design, formats, and the full
recording steps: [`../FINGERSPELLING_PLAN.md`](../FINGERSPELLING_PLAN.md).

Anything synthetic used to test this pipeline lives under
`tests/fixtures/fingerspelling_synthetic/`, never here.
