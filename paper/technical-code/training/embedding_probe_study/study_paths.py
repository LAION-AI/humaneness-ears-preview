"""Pinned experiment paths and atomic, credential-free metadata."""
import hashlib
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE = HERE.parent
ROOT = Path('/e/scratch/reformo/schuhmann1_moss/whisper_score_regression/embedding_probe_study_20261006')
BENCH = ROOT.parent / 'bench_eval_20261004'
GEMINI = ROOT.parent / 'gemini_finetune_preparation_20261005'
RELEASE = ROOT.parent / 'hf_layered_release/model'
CLAP_CODE = Path('/e/project1/laionize/foerster5_jupiter/clap_v2_refactor/open_clip')
CLAP_BENCH = CLAP_CODE.parent / 'clap_benchmark'
TRANSFORMERS_REVISION = '14e738b5d0cc69aa27a95dde272aea41fde44f2f'
SEED = 20261006


def read(path):
    return json.loads(Path(path).read_text())


def json_number(value):
    # Keep the orchestration process stdlib-only while accepting NumPy metric
    # scalars/arrays produced by the compute jobs. Nonfinite values still fail.
    if type(value).__module__.split('.')[0] == 'numpy' and hasattr(value, 'tolist'):
        return value.tolist()
    raise TypeError('Unsupported metadata value: ' + type(value).__name__)


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp-' + str(os.getpid()))
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=json_number) + '\n')
    os.replace(temporary, path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()
