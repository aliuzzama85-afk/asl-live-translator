# ASL Live Translator

Real-time speech/text -> ASL sign-avatar translator. See CLAUDE.md for full architecture, build order, and conventions.

Status: Stages 3-5 built and tested (gloss translation, pose library, single-word skeleton rendering); Stages 1-2 (live speech input) not started. See PROJECT_STATUS.md for full detail.

Translation quality note: the gloss model is trained on ASLG-PC12, a rule-generated (not human-translated) corpus dominated by formal/parliamentary-register text rather than everyday conversational language — translation quality for informal, everyday vocabulary is limited outside a small hand-augmented word list. See gloss_model/PLAN.md and gloss_model/VOCAB_DIAGNOSIS.md.

Data note: `pose_library`'s training/pose data (WLASL-derived videos and extracted pose JSON) is intentionally excluded from this repo — it's gitignored, not just untracked. WLASL is licensed under the Computational Use of Data Agreement, its own README additionally states no commercial usage is allowed, and the licensing terms for the underlying third-party source videos it indexes have not been independently verified. See pose_library/PLAN.md for the full licensing discussion and the documented download/build pipeline to regenerate the data yourself.
