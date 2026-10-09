# Reference audit, 9 October 2026

The bibliography was checked against primary paper or dataset pages. OpenReview's search API was queried because its forum pages were browser-gated. It returned the VoiceNet note `IpNgNPi9WD`, whose venue field says **ICLR 2027 Conference Submission**; this is not an acceptance claim. The named-author arXiv preprint is the stable citation in `references.bib`.

| BibTeX key | Verified source | Publication status / usage |
|---|---|---|
| `radford2023whisper` | [PMLR ICML 2023](https://proceedings.mlr.press/v202/radford23a.html) | Published Whisper foundation. |
| `schuhmann2025emonetface` | [NeurIPS 2025 proceedings](https://proceedings.neurips.cc/paper_files/paper/2025/hash/8b2dd02001e72f65091115fb7f39d8c7-Abstract-Datasets_and_Benchmarks_Track.html), [OpenReview forum](https://openreview.net/forum?id=BqVIt5Dxxh) | Published Datasets and Benchmarks paper; source of 40-emotion taxonomy. OpenReview's generated BibTeX says year 2026 despite its venue being NeurIPS **2025**; bibliography uses proceedings year 2025 and the proceedings author spelling **Kalyan Sai Krishna**. |
| `schuhmann2025emonetvoice` | [arXiv 2506.09827](https://arxiv.org/abs/2506.09827) | Named-author preprint, v3 January 2026. A different [withdrawn ICLR 2026 submission](https://openreview.net/forum?id=jik0oNrrfB) has a changed title and author order; it is deliberately **not** cited as accepted. |
| `schuhmann2026voicenet` | [arXiv 2609.32016](https://arxiv.org/abs/2609.32016), [OpenReview submission](https://openreview.net/forum?id=IpNgNPi9WD) | Named-author preprint; OpenReview lists an anonymized ICLR 2027 submission. VoiceNet-Ext is characterized as preliminary in the paper. |
| `cao2014cremad` | [Original dataset repository](https://github.com/CheyneyComputerScience/CREMA-D), [DOI](https://doi.org/10.1109/TAFFC.2014.2336244) | IEEE TAC 2014; 7,442 clips/91 actors. |
| `livingstone2018ravdess` | [PLOS ONE DOI](https://doi.org/10.1371/journal.pone.0196391) | Published 2018 dataset paper; our test uses the 1,440 speech recordings, not its whole speech-and-song collection. |
| `tjandra2025audiobox` | [arXiv 2502.05139](https://arxiv.org/abs/2502.05139) | Four axes CE/CU/PC/PQ; these are machine-target scores in our test. |
| `reddy2021dnsmos` | [arXiv 2110.01763](https://arxiv.org/abs/2110.01763) | P.835 SIG/BAK/OVRL, plus the separate P808 score in our implementation. |
| `chen2022wavlm` | [Microsoft Research publication](https://www.microsoft.com/en-us/research/publication/wavlm-large-scale-self-supervised-pre-training-for-full-stack-speech-processing/), [DOI](https://doi.org/10.1109/JSTSP.2022.3188113) | Published IEEE JSTSP 2022, background for speech representations. |
| `gemmeke2017audioset` | [Google Research publication](https://research.google/pubs/audio-set-an-ontology-and-human-labeled-dataset-for-audio-events/) | Published ICASSP 2017. Cited in the source ledger; not essential to the short main text. |
| `laion2026locator` | [LAION model card](https://huggingface.co/laion/vocalburst-locator) | Model card created 4 June 2026 according to the Hugging Face model API; not a peer-reviewed benchmark paper. Citation does not assert independent human event labels. |
| `orange2026timbre`, `orange2026identity` | [Timbre card](https://huggingface.co/Orange/Speaker-wavLM-tbr), [identity card](https://huggingface.co/Orange/Speaker-wavLM-id) | Model cards; the 2026 BibTeX year denotes access date, not a verified publication date. Keep the two embedding spaces separate. |

The references do not cite the project itself as accepted or peer reviewed. Check all venue metadata again before final submission, especially the future ICML/Interspeech style and any new VoiceNet decision.
