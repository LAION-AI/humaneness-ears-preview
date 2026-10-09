# Humaneness Ears paper project

This is a self-contained **ICML-format, Overleaf-ready LaTeX draft** about the encoder-only Humaneness Ears Base and Medium models. `main.tex` is the root document; `references.bib` contains checked primary-source references; `figures/` contains two vector TikZ diagrams. Upload the folder or the lightweight `overleaf_upload.zip` to Overleaf, or connect this folder to a repository, and set `main.tex` as the main document. The `evidence/` directory is an optional audit bundle and is not needed to compile.

## Build

```bash
module load texlive/20250308  # JUPITER only; omit on Overleaf
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The project includes the official **ICML 2026** style and BibTeX file from the [2026 author kit](https://media.icml.cc/Conferences/ICML2026/Styles/icml2026.zip), downloaded 9 October 2026 (ZIP SHA-256 `8b29290f5828e176debb57ea9cc00252502973d55ea561a2f18a7f0a326bfc6c`). This is an authoring format, not a claim of ICML 2026 submission. The `preprint` style option avoids the template's default false “under review” notice. The venue and year of a future submission need a fresh official style and rules check. The 2026 ICML rule allowed at most eight main-text pages, followed by references and appendices; the draft is shorter. The manuscript has anonymous author/contact placeholders; the real author list, affiliations, disclosures, and final venue must be supplied by the authors before submission. It has not been posted or submitted by this task.

## Evidence and scientific scope

- [`evidence/snapshots/`](evidence/snapshots/) holds selected frozen JSON results, dataset counts, the 40-run benchmark study, and exact release normalization/class mapping files. [`evidence/SHA256SUMS`](evidence/SHA256SUMS) records their checksums. The release metadata was fetched from model revision `818506970d93c809a3295a0a02c40dee8ff2bfc3`.
- [`target_schema.csv`](target_schema.csv) enumerates all 192 output keys, training statistics, and separate display z-score statistics. Run `python3 export_target_schema.py` to regenerate it.
- [`EVIDENCE_LEDGER.md`](EVIDENCE_LEDGER.md) maps headline claims to files and records protocol caveats and conflicts.
- [`REFERENCES_AUDIT.md`](REFERENCES_AUDIT.md) records the publication status and official source URL for each cited work.
- [`MODEL_SOURCES.md`](MODEL_SOURCES.md) lists every backbone and annotation teacher, the purpose of each, and the student outputs they supervise.
- Run `python3 verify_bundle.py` to check evidence hashes, citation keys, and the final-model numbers in the paper.
- The [released Base/Medium weights and code](https://huggingface.co/laion/humaneness-ears-base-medium) and [original layered curriculum weights](https://huggingface.co/laion/whisper-base-small-layered-audio-scores) are external artifacts. The paper bundle does not copy weights, recordings, credentials, or source data.

The principal contribution currently supported by evidence is a compact, broad **joint student and evaluation**. The present experiments do not isolate a causal advantage of multitask learning over a suite of single-task models. Many target scores are specialist-model predictions. External human-label tests and synthetic event labels answer different questions and are identified separately in the manuscript. Acceptance at any venue cannot be inferred from this draft.

## Before submission

The highest-value additional experiment is a controlled single-task or reduced-task ablation under a matched audio budget. A revised burst mix with real manually checked events, replay during Gemini tuning, and confidence intervals on paired external-test differences would strengthen the claims. Also reconcile VoiceNet-Ext snapshot counts against its paper, confirm no train/test source leakage on every external benchmark, check source licenses for any data release, and fill the final author list and required disclosures.
