# Third-party data: ASLNow! isolated fingerspelling dataset (MIT)

The fingerspelling letters in `pose_library/fingerspelling/poses/` whose
manifest `source` is `hf:sid220/asl-now-fingerspelling` are converted from
this dataset. This file is the durable local record of where they came from
and under what terms, so it travels with the data. It is this project's first
standalone license file for a data source: WLASL's C-UDA terms are recorded in
`pose_library/PLAN.md` Section 1, `PROJECT_STATUS.md` Section 5, and a
per-entry license tag in the (gitignored, local-only) WLASL manifest, and
ASLG-PC12's CC BY-NC 4.0 in `PROJECT_STATUS.md` Section 5 and `README.md`.
This dataset gets its own file because it is redistributed here (converted
letters and test fixtures are committed), and MIT requires the copyright and
permission notice to accompany copies.

| | |
|---|---|
| Dataset | `sid220/asl-now-fingerspelling` ("ASLNow!") |
| Source | https://huggingface.co/datasets/sid220/asl-now-fingerspelling |
| Revision used | `9b3c96ae0adb7744a2c9fc72692842e6b3e25e33` (last modified 2023-12-19) |
| DOI | [10.57967/hf/1494](https://doi.org/10.57967/hf/1494) |
| Author | Sidney Trzepacz (Hugging Face `sid220`, GitHub `Sid220`) |
| License | MIT |
| Checked | 2026-09-26, against both primary sources below |

## What the sources actually say

**The Hugging Face dataset card** declares the license in its metadata
(`README.md` YAML front matter, verbatim):

```yaml
license: mit
```

and in its prose (verbatim): *"This dataset, used to train the fingerspelling
model is licensed under the MIT License."*

**The Hugging Face repo contains no `LICENSE` file** (checked: `/LICENSE`
returns 404, and the repo's only non-data files are `README.md` and
`.gitattributes`), so it publishes no copyright line of its own. The MIT
license requires the copyright notice and permission notice to be kept with
copies, so the notice below is taken from the author's own project repository,
[github.com/Sid220/asl-now](https://github.com/Sid220/asl-now). That's the
ASLNow! app this dataset was collected for, and its README links to this exact
dataset. Its [`LICENSE`](https://github.com/Sid220/asl-now/blob/main/LICENSE),
as of its current version on `main` (first added in commit `84ac5d06cda2`,
last changed in `172ee6d54f71`, both 2023-12-22), reads, verbatim:

```
Copyright 2024 Sidney Trzepacz

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```

Strictly, that `LICENSE` file covers the GitHub repository, and the dataset's
own MIT grant is the Hugging Face declaration quoted above. Both come from the
same author for the same project and name the same license, so there is no
conflict. This file records both rather than blurring them into one.

## How this project uses it

- **What was taken**: one sample per static letter (24 letters, A–Z except
  J and Z), each a single MediaPipe hand-landmark frame. The chosen sample is
  the per-letter medoid (see
  [`../FINGERSPELLING_PLAN.md`](../FINGERSPELLING_PLAN.md) Section 2b).
  Each converted `poses/<letter>.json` and its manifest entry names the exact
  source file (e.g. `A/<uuid>.json`), `source_url`, and license.
- **What was changed**: the 21 landmarks are placed into this project's
  48-landmark `PoseSequence` layout (the other hand and pose points are
  zero-filled as "not detected") and held for 0.4s. No landmark values are
  altered.
- **What was *not* taken**: J and Z. The dataset stores them as single static
  frames, but they are motion letters, so they're left unavailable rather
  than faked.
- **Raw dataset copy**: downloaded to the gitignored
  `pose_library/data/asl_now_fingerspelling/` cache by
  `python -m pose_library.convert_hf_fingerspelling`; not redistributed in
  this repository. A handful of real samples are committed as test fixtures
  under `tests/fixtures/asl_now_sample/`, under this same notice.
