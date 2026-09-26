# Real sample of sid220/asl-now-fingerspelling (MIT), for tests

Eleven **real, unmodified** sample files copied from the
[sid220/asl-now-fingerspelling](https://huggingface.co/datasets/sid220/asl-now-fingerspelling)
dataset (revision `9b3c96ae0adb7744a2c9fc72692842e6b3e25e33`), used by
`tests/test_convert_hf_fingerspelling.py`:

- `A/`: the first 8 A samples by filename.
- `C/b2962280-fb9d-41c2-9f94-51a8e9f21c4a.json`: a real sample with a
  landmark outside the image (`x` up to 1.073), used to test the in-frame
  filter.
- `J/`: the first 2 J samples, used to test that J isn't converted even
  though the dataset has files for it.

MIT License, Copyright 2024 Sidney Trzepacz. The full notice and provenance
are in
[`pose_library/fingerspelling/THIRD_PARTY_LICENSE_asl-now-fingerspelling.md`](../../../pose_library/fingerspelling/THIRD_PARTY_LICENSE_asl-now-fingerspelling.md).
