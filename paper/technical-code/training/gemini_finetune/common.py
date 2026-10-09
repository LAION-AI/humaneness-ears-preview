"""Shared paths, atomic outputs and random access to audio in WebDataset TARs."""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
import tarfile
from collections import OrderedDict
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE = HERE.parent
HOME = CODE.parent
ROOT = Path('/e/scratch/reformo/schuhmann1_moss/whisper_score_regression/gemini_finetune_preparation_20261005')
SELECTION = ROOT.parent / 'gemini_multisource_100h_20261004'
GEMINI = SELECTION / 'annotation_batch_20261005'
RELEASE = ROOT.parent / 'hf_layered_release/model'
MOSS_UPSTREAM = ROOT / 'upstream/MOSS-Audio'
MOSS_REVISION = 'd4dc5a6d8cd79b43dcd82884c75e303b1ecd016d'
UPSTREAM_REVISION = '66326e6e0db34f036c86a76ba005efa4830c69dd'
TEACHER_NAMES = ('orange', 'dnsmos', 'audiobox', 'voiceclap', 'empathic')
EMOTION_ALIASES = {
    'Hope_Enthusiasm_Optimism': 'hope_optimism',
    'Impatience_and_Irritability': 'impatience_irritability',
    'Intoxication_Altered_States_of_Consciousness': 'intoxication_altered_states',
    'Jealousy_and_Envy': 'jealousy_envy', 'Jealousy_&_Envy': 'jealousy_envy',
    'Thankfulness_Gratitude': 'gratitude',
}


def emotion_key(name):
    return EMOTION_ALIASES.get(name, name.lower())


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.incomplete')
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def jsonl(path):
    with Path(path).open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def put_jsonl(stream, row):
    stream.write(json.dumps(row, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n')


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def add_member(archive, name, payload):
    member = tarfile.TarInfo(name)
    member.size, member.mtime = len(payload), 0
    archive.addfile(member, io.BytesIO(payload))


class TarReader:
    """Small per-worker TAR cache; no extracted audio files or network calls."""
    def __init__(self, maximum=12):
        self.maximum = maximum
        self.archives = OrderedDict()

    def read(self, path, member):
        path = str(path)
        if path not in self.archives:
            if len(self.archives) >= self.maximum:
                _, old = self.archives.popitem(last=False)
                old.close()
            self.archives[path] = tarfile.open(path, 'r:')
        self.archives.move_to_end(path)
        handle = self.archives[path].extractfile(member)
        if handle is None:
            raise ValueError(f'Not a regular TAR member: {member}')
        return handle.read()

    def audio(self, row, verify=False):
        import numpy as np
        import soundfile as sf
        from scipy.signal import resample_poly
        raw = self.read(row['audio_tar'], row['audio_member'])
        if verify and hashlib.sha256(raw).hexdigest() != row['sha256_mp3']:
            raise ValueError('Audio hash differs from selected, annotated audio')
        wav, sr = sf.read(io.BytesIO(raw), dtype='float32', always_2d=True)
        wav = wav.mean(axis=1)
        if sr != 16000:
            divisor = math.gcd(sr, 16000)
            wav = resample_poly(wav, 16000 // divisor, sr // divisor).astype(np.float32)
        if not len(wav) or not np.isfinite(wav).all():
            raise ValueError('Empty or non-finite audio')
        if len(wav) > 480000 + 1600:
            raise ValueError('Selected audio exceeds the 30-second training contract')
        return wav[:480000]

    def close(self):
        for archive in self.archives.values():
            archive.close()
        self.archives.clear()
