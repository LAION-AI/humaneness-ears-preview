# Released model and training code

These files were copied from the public [Humaneness Ears model release](https://huggingface.co/laion/humaneness-ears-base-medium/tree/818506970d93c809a3295a0a02c40dee8ff2bfc3) at immutable revision `818506970d93c809a3295a0a02c40dee8ff2bfc3` on 9 October 2026. The original release is authoritative for weights, files, and later updates. This copy is included to make review of the paper convenient. It contains no model weights, recordings, API keys or private source data.

The core implementation is `model.py`, `inference.py`, `training/layered_multitask_model.py`, `training/gemini_finetune/train_whisper.py`, and `training/gemini_finetune/data.py`. Configurations and target normalizations are also included. Some training scripts require the source datasets, cluster tooling and pretrained models described in the paper; this directory alone does not reproduce a full run.
