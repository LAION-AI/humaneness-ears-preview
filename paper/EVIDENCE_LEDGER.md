# Humaneness Ears: claim and experiment ledger

Audit date: 9 October 2026. This ledger belongs to the local manuscript bundle. The frozen, checksum-listed files in `evidence/snapshots/` are the sources for numbers below; code paths are relative to the JUPITER home folder. The main text uses **Base S4 and Small S3 after two Gemini fine-tuning epochs**, released as Humaneness Ears **Base** and **Medium**. It does not substitute an earlier checkpoint when a final-model metric is available.

## Model identity and training lineage

| Claim in paper | Primary local evidence | Notes |
|---|---|---|
| Whisper audio encoder only; Base 6 layers/512 width, Medium initialized from Whisper Small 12 layers/768 width; final trainable parameter counts 26,441,735 / 97,102,031 | `whisper_score_regression/layered_multitask_model.py`; `WHISPER_BASE_SMALL_LAYERED_S1_S10_RESULTS_2026-10-03.html`; local release model configs | Decoder and ASR head are absent. “Medium” is a release name, not the OpenAI Whisper Medium model. Encoder counts are 20,590,592 / 88,154,112. |
| 192 scalar targets plus separate CPS, 128D Orange timbre and 250D Orange identity, frame and event heads, 53 event classes | `whisper_score_regression/layered_multitask_model.py`; `whisper_score_regression/train_layered_curriculum.py`; `whisper_score_regression/gemini_finetune/common.py` | Scalar count is 40+57+3+19+4+7+61+1. The last one is burst count; do not count it again in diagrams. Not all labels exist for every sample; losses are masked. |
| S1→S10 one continuous cosine schedule, 10% P3 in each stage, 71,934,743 exposures, 281,006 updates per model, 16 GH200 across four nodes, effective batch 256 | `whisper_score_regression/train_layered_curriculum.py`, `.sbatch`; `WHISPER_BASE_SMALL_LAYERED_S1_S10_RESULTS_2026-10-03.html`; release checkpoint configs | The ladder stages are nested, so “exposures” count repeated samples. They are not 71.9M unique recordings. |
| Curriculum peak encoder LRs 2e-5 / 1.5e-5, head 3e-4, 2% warmup (5,620 updates), 10% floor, AdamW decay 0.01, BF16 and clipping 1 | Executed stage run configs and `whisper_score_regression/train_layered_curriculum.py` | **Conflict resolved:** one model-card draft says 5% warmup. Executed configs record 2%; the manuscript uses 2%. |
| Selected S4 Base and S3 Small initializers | `whisper_score_regression/gemini_finetune/configs/whisper_base.json` and `whisper_small.json` | The final S10 stage is not the released Gemini initializer. |
| P3 source-component-disjoint train 1,031,454 / val 10,000 / test 9,693; excluded 68,853 candidates | `WHISPER_BASE_SMALL_LAYERED_S1_S10_RESULTS_2026-10-03.html`; frozen P3 manifest and checks in Scratch | Speaker overlap is not certified; source-component disjointness applies to the recorded source IDs. |
| Gemini candidate 66,199 exact-audio-unique clips / 197.36 h; schema-valid 65,163 / 194.17 h; 58,380 train, 3,239 validation, 3,544 test | `evidence/snapshots/selected_pool.json`; `valid_split_summary.json`; `whisper_score_regression/gemini_finetune/prepare.py` | 1,036 outputs were invalid or blocked. Split uses known speaker/family group hashes, then audio hash; unknown-speaker leakage cannot be ruled out. |
| Two Gemini epochs, four GH200, batch 128, 914 updates, encoder peaks 1e-5 / 5e-6, head 1e-4, 5% warmup, 10% floor | `whisper_score_regression/gemini_finetune/configs/whisper_base.json`, `whisper_small.json`; `train_whisper.py`; run logs | Optimizer is restarted for adaptation. Best validation checkpoint is epoch 2: 1.7684 / 1.6989. P3 is not replayed. |

## Which labels are human, teacher, or synthetic?

The Gemini fine-tuning set contains model-generated emotion, VoiceNet-style, genuineness/blend, speech-rate and timestamp labels. Overlapping VoiceCLAP emotion slots also take the Gemini value. AudioBox, DNSMOS, Orange, Empathic extras and the remaining VoiceCLAP targets are cached specialist-model outputs in audio-hash-bound sidecars. P3 insertion times derive from synthetic composition. **None of these are independent human annotations.** EmoNet-Voice and VoiceNet external tests carry expert labels; CREMA-D and RAVDESS carry acted-corpus labels. The paper distinguishes these categories in each table and in the discussion.

Event-class taxonomy: `whisper_score_regression/gemini_finetune/common.py` maps 78 allowed Gemini fine labels through aliases into 53 student classes, with unmapped rare items assigned to Other. Interval metrics match predicted and reference intervals one-to-one at IoU ≥ 0.5; frame F1 uses valid 20-ms bins and a fixed 0.5 threshold.

The bundle includes `training_normalization.json`, `display_normalization.json`, and `gemini_burst_mapping.json` fetched from released model revision `818506970d93c809a3295a0a02c40dee8ff2bfc3`. `target_schema.csv` exports all 192 ordered names and both training and display statistics from those pinned files. Display z-scores use a median and standard deviation reference and must not be substituted for the training mean/std transform.

## Evaluation source map

| Result | Frozen snapshot / local report | Comparability constraint |
|---|---|---|
| 3,544 Gemini test: emotion nMAE .587/.561, VoiceNet .497/.481, AudioBox .188/.174, DNSMOS .205/.198; timbre cosine .9038/.9223, identity .8541/.8741; CPS MAE .901/.931; frame F1 .6929/.6892; event F1 .19446/.19490 | `base_gemini_test_metrics.json`, `small_gemini_test_metrics.json` | Score metrics compare to machine teachers. Speaker cosine uses 1,718 valid single-speaker clips. Event precision .111/.112 and recall .775/.767 explain the low F1; start/end MAE .120/.197 and .115/.201 s are **matched-event only**. |
| Mapped EmoNet-Voice 12,000 pairs, Pearson .43396/.45119, Spearman .45768/.46785 | `base_public_metrics.json`, `small_public_metrics.json` | Original benchmark has 12,600 pairs; two queried labels are unmapped and 687 mapped clips exceed 30 s and are truncated. Not matched to published Empathic Insight sample subset. |
| VoiceNet-Emo 7,986 pairs, mean per-prompt Spearman .4288/.4282; VoiceNet-Ext 13,917 pairs, .1566/.1502 | Same public metrics JSON; `WHISPER_ALL_BENCHMARK_RESULTS_2026-10-07.json.gz` | >=2-rater local snapshot. VoiceNet paper uses a different Ext count (16,166) and voice–text cosine interface. Paper comparison is contextual only. |
| CREMA-D actor-CV 63.1013%/64.2300%; RAVDESS actor-CV 65.9722%/66.3889% | `base_crema_matched_adapter.json`, `small_crema_matched_adapter.json`, `base_ravdess_matched_adapter.json`, `small_ravdess_matched_adapter.json` | **Final Gemini checkpoints.** 192 score features → fold-train StandardScaler → 64 GELU → class logits. Five actor-disjoint outer folds, three inner folds, full-batch AdamW. This is supervised transfer. |
| Fixed crosswalk: CREMA-D 18.9196%/19.7393%; RAVDESS 24.7917%/24.7222% | `base_public_metrics.json`, `small_public_metrics.json` | No benchmark-label fitting. The poor calibration is a real limitation and should not be hidden by the supervised adapter. |
| Fixed 2,000 P3 test clips: pre/post frame Base .9622→.8051, Medium .9693→.8392; event Base .7499→.3864, Medium .7955→.4349 | `base_legacy_p3_test_metrics.json`, `base_gemini_p3_test_metrics.json`, `small_legacy_p3_test_metrics.json`, `small_gemini_p3_test_metrics.json` | Exactly the same 2,000 clips in all four files. |
| 9,693 P3 pre-tune: frame F1 .960/.969; binary-region F1 .937/.962 | `WHISPER_BASE_SMALL_COMPREHENSIVE_PUBLIC_BENCHMARKS_2026-10-04.html` | Historical S4/S3 checkpoints only; separate from 2,000 paired retention set. |
| 300-clip locator mixture: frame F1 .263/.279; binary-region F1 .065/.063 | Same 4 October benchmark report | **Pre-Gemini S4/S3 only.** No final-Gemini evaluation exists in the audited files; cannot call these post-tune results. Source overlap with P3 is unproven. |
| Early Tiny/Base S9/S10 100-target validation Huber: Base S9 .1362, Tiny S9 .1466; Base S10 .1662, Tiny S10 .1749 | `tiny_base_s9_s10_comparison.json` | S9 and S10 trained for three epochs each but differ in both data and optimizer updates. Not a controlled scale-only test. |
| CLAPv2 XS/M: Gemini event F1 .302/.299, timbre cosine .723/.713, mapped EmoNet Spearman .374/.391 and VoiceNet-Ext rho .157/.163 | `CLAP_V2_FULL_FINETUNE_BENCHMARK_2026-10-07.html`; consolidated `WHISPER_ALL_BENCHMARK_RESULTS_2026-10-07.json.gz` | CLAP uses one S8 warm-up pass, then two Gemini epochs; pretrained backbone differs. Same held-out Gemini test, but no architecture-only causal interpretation. |

### Historical results intentionally excluded from final-model table

`WHISPER_BASE_SMALL_COMPREHENSIVE_PUBLIC_BENCHMARKS_2026-10-04.html` and `whisper_score_regression/score_public_benchmarks.py` refer to original **Base S4 / Small S3 before Gemini tuning**. Their later 80-feature multinomial linear calibration yields CREMA-D 62.12%/63.37% and RAVDESS 67.15%/69.58%, and their fixed crosswalk yields CREMA-D 22.3%/23.8%, RAVDESS 36.7%/36.9%. Those are different checkpoints and protocols from the paper's final Gemini-tuned 192-feature MLP. Copying those numbers into the final-model row would be an error.

## Source and release locations

- Local model/training implementation: `whisper_score_regression/layered_multitask_model.py`, `train_layered_curriculum.py`, `gemini_finetune/train_whisper.py`, `gemini_finetune/data.py`.
- Local reports: `WHISPER_BASE_SMALL_LAYERED_S1_S10_RESULTS_2026-10-03.html`, `WHISPER_BASE_SMALL_COMPREHENSIVE_PUBLIC_BENCHMARKS_2026-10-04.html`, `CLAP_V2_FULL_FINETUNE_BENCHMARK_2026-10-07.html`, and `WHISPER_ALL_BENCHMARK_RESULTS_2026-10-07.json.gz`.
- Released model/code: <https://huggingface.co/laion/humaneness-ears-base-medium> and historical stage release <https://huggingface.co/laion/whisper-base-small-layered-audio-scores>.
- Supporting public method pages: <https://huggingface.co/laion/vocalburst-locator>, <https://huggingface.co/datasets/laion/tts-scaling-ladder-de-en>, <https://huggingface.co/Orange/Speaker-wavLM-tbr>, <https://huggingface.co/Orange/Speaker-wavLM-id>.

## Open issues before a submission claim

Run a matched-budget single-task or reduced-task ablation, and preferably a paired same-item comparison against the teacher models. Verify external test contamination at source and speaker level, manually audit a real-event burst test, evaluate the final tuned models on the locator mixture, and report uncertainty intervals. The current manuscript does not claim a causal advantage from joint learning or superiority over paper baselines on differing benchmark snapshots.
