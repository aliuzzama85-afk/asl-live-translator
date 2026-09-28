# ASL Live Translator

Real-time speech/text -> ASL sign-avatar translator. See CLAUDE.md for full architecture, build order, and conventions.

Status: all five stages built. Live speech input (Stages 1-2) uses the browser's Web Speech API and is automated-tested, but still needs a test with a real microphone by a person. Fingerspelling covers all 26 letters. See PROJECT_STATUS.md for full detail.

## Running it

Set up the Python environment once (`setup_env.ps1` on Windows, `setup_env.sh` elsewhere) and run `npm install` in `frontend/`. The WLASL pose library and the gloss model checkpoint are local-only (see the notes below), so they must exist on your machine.

From `frontend/`:

- **`npm run dev`**: the app alone, at http://localhost:5173. Typed input works in **EXACT WORDS** mode only. TRANSLATE mode and live speech will say the translation service isn't running.
- **`npm run dev:live`**: the app **and** the local gloss server (`python -m pipeline.gloss_server`) together, for live speech. Both stop when you press Ctrl+C. Use Chrome, Edge, or Safari (Firefox has no Web Speech API). Speech is processed by the browser's speech service (e.g. Google in Chrome), not on your device.

Typed input has two modes, switched with the toggle next to the input box:

- **TRANSLATE** (the default): type ordinary English, e.g. "where is the bathroom". It goes through the gloss model exactly like speech, and the gloss line shows what the model produced (e.g. `WHERE BE BATHROOM`) and which gloss words aren't signed. Needs the translation service (`npm run dev:live`).
- **EXACT WORDS**: type sign words, e.g. "about angry". Each word is looked up literally, without the model, and words not in the library are fingerspelled. Useful for testing the vocabulary; works without the translation service.

Before 2026-09-28, typed input was always the literal EXACT WORDS lookup and never reached the model. That was a design gap, now corrected. Model quality is limited (e.g. "hello" glosses to HALF, "taxi" to TITTLE), and typed input will show that more often than the tested phrases do. Words such as "where" are fingerspelled because they aren't in the 118-word sign library.

Developers can open http://localhost:5173/?asr=fake to run the live path with a scripted fake recognizer (no microphone needed; dev builds only).

Translation quality and licensing note: the gloss model is trained on ASLG-PC12, a rule-generated (not human-translated) corpus dominated by formal/parliamentary-register text rather than everyday conversational language — translation quality for informal, everyday vocabulary is limited outside a small hand-augmented word list. ASLG-PC12 is licensed CC BY-NC 4.0 (NonCommercial), confirmed directly at the primary source (achrafothman.net) rather than taken from the Hugging Face dataset card's own inconsistent metadata. There is also an unresolved provenance question: the actual English text in the dataset we use reads as Europarl (EU Parliament proceedings) rather than the Gutenberg literary text the original paper claims as its source, which puts the validity of that CC BY-NC 4.0 license grant itself in question. See gloss_model/PLAN.md, gloss_model/VOCAB_DIAGNOSIS.md, and PROJECT_STATUS.md Section 5 for full detail.

Data note: `pose_library`'s training/pose data (WLASL-derived videos and extracted pose JSON) is intentionally excluded from this repo — it's gitignored, not just untracked. WLASL is licensed under the Computational Use of Data Agreement, its own README additionally states no commercial usage is allowed, and the licensing terms for the underlying third-party source videos it indexes have not been independently verified. See pose_library/PLAN.md for the full licensing discussion and the documented download/build pipeline to regenerate the data yourself.
