"""Reusable Orange timbre and identity targets for Whisper training.

Build separate sorted UID indexes from the existing Parquet sidecars once.
During training, ``TimbreTargetCache.get_or_compute`` uses existing vectors
and computes only missing 128-D timbre or 250-D identity vectors from the
same decoded waveform. New vectors are persisted as rank-owned Parquet chunks;
at the next stage boundary, index those chunks so S2..S10 and the second
Whisper run read them without teacher inference.

Both teachers require compute-visible model snapshots. Their licenses are
staged alongside the weights. No CUDA work is done in DataLoader workers.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
import json
import os
from pathlib import Path
from typing import Sequence
import zlib

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from safetensors import safe_open
from torch import nn
from torch.nn import functional as F
from transformers import WavLMConfig
from transformers.models.wavlm.modeling_wavlm import WavLMModel


DIM = 128
IDENTITY_DIM = 250
UID_DTYPE = 'S128'
DEFAULT_MODEL = Path('/e/project1/reformo/schuhmann1/whisper_score_regression/models/orange_speaker_wavlm_tbr_b8d2608')
IDENTITY_MODEL = Path('/e/project1/reformo/schuhmann1/whisper_score_regression/models/orange_speaker_wavlm_id_abbb3c7')
EMOLIA_SOURCE = Path('/e/scratch/reformo/schuhmann1_moss/out/refassign/emolia_emb.parquet')
IDENTITY_SOURCE = Path('/e/project1/reformo/schuhmann1/whisper_score_regression/sources/orange_id_kartoffelphon')


class TopLayers(nn.Module):
    """Orange's two-layer projection; names match the published checkpoint."""

    def __init__(self, embd_size: int, intermediate: int):
        super().__init__()
        self.affine1 = nn.Conv1d(2048, intermediate, 1)
        self.batchnorm1 = nn.BatchNorm1d(intermediate, affine=False, eps=1e-3)
        self.affine2 = nn.Conv1d(intermediate, embd_size, 1)
        self.batchnorm2 = nn.BatchNorm1d(embd_size, affine=False, eps=1e-3)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.batchnorm1(self.activation(self.affine1(x)))
        out = self.batchnorm2(self.activation(self.affine2(out)))
        return F.normalize(out[:, :, 0], dim=-1)


class OrangeEmbeddingTeacher(nn.Module):
    """Orange WavLM speaker embedding model with masked padded batches."""

    def __init__(self, snapshot: Path, dim: int):
        super().__init__()
        config = WavLMConfig.from_pretrained(snapshot, local_files_only=True)
        assert config.embd_size == dim, (config.embd_size, dim)
        self.dim = dim
        self.wavlm = WavLMModel(config)
        self.top_layers = TopLayers(config.embd_size, config.top_interm_size)
        with safe_open(snapshot / 'model.safetensors', framework='pt', device='cpu') as file:
            state = {key: file.get_tensor(key) for key in file.keys()}
        self.load_state_dict(state, strict=True)
        self.eval()

    @torch.inference_mode()
    def forward(self, samples: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Input float32 16-kHz mono [B,T], output L2-normalized [B,dim]."""
        assert samples.ndim == 2 and lengths.ndim == 1 and samples.shape[0] == len(lengths)
        assert torch.all((lengths >= 1600) & (lengths <= samples.shape[1]))
        t = samples.shape[1]
        valid = (torch.arange(t, device=samples.device)[None, :] < lengths[:, None])
        n = lengths[:, None].float()
        mu = (samples * valid).sum(1, keepdim=True) / n
        var = (((samples - mu) * valid).square()).sum(1, keepdim=True) / (n - 1).clamp_min(1)
        x = ((samples - mu) / var.clamp_min(1e-10).sqrt()) * valid
        h = self.wavlm(input_values=x, attention_mask=valid.long(),
                       output_hidden_states=False).last_hidden_state.float()
        frame_lengths = self.wavlm._get_feat_extract_output_lengths(lengths)
        fvalid = (torch.arange(h.shape[1], device=h.device)[None, :] < frame_lengths[:, None])[:, :, None]
        nf = fvalid.sum(1).clamp_min(1)
        hm = (h * fvalid).sum(1) / nf
        hv = ((h - hm[:, None, :]).square() * fvalid).sum(1) / (nf - 1).clamp_min(1)
        pooled = torch.cat((hm, hv.clamp_min(1e-10).sqrt()), dim=1).unsqueeze(-1)
        return self.top_layers(pooled)

    @torch.inference_mode()
    def encode_waveforms(self, waveforms: Sequence[np.ndarray | torch.Tensor],
                         device: torch.device, batch_size: int = 4) -> np.ndarray:
        """Dynamic length buckets; waveforms must already be mono at 16 kHz."""
        out = np.empty((len(waveforms), self.dim), dtype=np.float16)
        order = sorted(range(len(waveforms)), key=lambda i: len(waveforms[i]))
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            lengths = torch.tensor([max(1600, min(len(waveforms[i]), 480000)) for i in indices],
                                   device=device, dtype=torch.long)
            max_len = max(1600, int(lengths.max()))
            audio = torch.zeros((len(indices), max_len), device=device, dtype=torch.float32)
            for row, i in enumerate(indices):
                w = torch.as_tensor(waveforms[i], dtype=torch.float32, device=device)
                count = min(len(w), 480000)
                audio[row, :count] = w[:count]
            pred = self(audio, lengths).float().cpu().numpy()
            assert pred.shape == (len(indices), self.dim) and np.isfinite(pred).all()
            out[indices] = pred.astype(np.float16)
        return out


class OrangeTimbreTeacher(OrangeEmbeddingTeacher):
    def __init__(self, snapshot: Path = DEFAULT_MODEL):
        super().__init__(snapshot, DIM)


class OrangeIdentityTeacher(OrangeEmbeddingTeacher):
    def __init__(self, snapshot: Path = IDENTITY_MODEL):
        super().__init__(snapshot, IDENTITY_DIM)


def _uid_array(values: Sequence[str]) -> np.ndarray:
    encoded = [u.encode('utf-8') for u in values]
    assert all(0 < len(u) <= 128 for u in encoded)
    return np.asarray(encoded, dtype=UID_DTYPE)


def _emb_array(column: pa.Array, dim: int = DIM) -> np.ndarray:
    # Both list<double> in Emolia and fixed_size_list<float16> in the Identity
    # source/generated chunks are accepted. Reject missing/wrong-width vectors.
    if isinstance(column, pa.ChunkedArray):
        column = column.combine_chunks()
    assert column.null_count == 0
    lens = np.diff(np.asarray(column.offsets)) if isinstance(column, pa.ListArray) else None
    if lens is not None:
        assert np.all(lens == dim), (lens.min(), lens.max())
    values = np.asarray(column.values.to_numpy(zero_copy_only=False), dtype=np.float32)
    result = values.reshape(len(column), dim)
    assert np.isfinite(result).all()
    norms = np.linalg.norm(result, axis=1)
    assert np.all((norms > .95) & (norms < 1.05)), (norms.min(), norms.max())
    return result.astype(np.float16)


def build_index(parquets: Sequence[Path], output: Path, allow_same_duplicates: bool = False,
                dim: int = DIM) -> None:
    """Create memory-mappable sorted UID→float16[dim] arrays atomically."""
    output.mkdir(parents=True, exist_ok=True)
    paths = [Path(p) for p in parquets]
    count = sum(pq.read_metadata(p).num_rows for p in paths)
    assert count > 0
    raw_uid = np.lib.format.open_memmap(output / 'uids.raw.npy', mode='w+', dtype=UID_DTYPE, shape=(count,))
    raw_emb = np.lib.format.open_memmap(output / 'emb.raw.npy', mode='w+', dtype=np.float16, shape=(count, dim))
    offset = 0
    for path in paths:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=32768, columns=['uid', 'emb']):
            n = batch.num_rows
            raw_uid[offset:offset+n] = _uid_array(batch.column(0).to_pylist())
            raw_emb[offset:offset+n] = _emb_array(batch.column(1), dim)
            offset += n
    assert offset == count
    raw_uid.flush(); raw_emb.flush()
    order = np.argsort(raw_uid, kind='stable')
    sorted_uid = np.lib.format.open_memmap(output / 'uids.next.npy', mode='w+', dtype=UID_DTYPE, shape=(count,))
    sorted_emb = np.lib.format.open_memmap(output / 'emb.next.npy', mode='w+', dtype=np.float16, shape=(count, dim))
    for start in range(0, count, 65536):
        ix = order[start:start+65536]
        sorted_uid[start:start+len(ix)] = raw_uid[ix]
        sorted_emb[start:start+len(ix)] = raw_emb[ix]
    sorted_uid.flush(); sorted_emb.flush()
    dup = np.flatnonzero(sorted_uid[1:] == sorted_uid[:-1])
    if len(dup):
        if not allow_same_duplicates:
            raise ValueError(f'{len(dup)} duplicate UIDs')
        for i in dup:
            if not np.allclose(sorted_emb[i], sorted_emb[i+1], atol=2e-3):
                raise ValueError(f'conflicting duplicate UID {sorted_uid[i]!r}')
        # A compacted index is required for binary search to return one row.
        keep = np.ones(count, dtype=bool); keep[dup+1] = False
        unique_n = int(keep.sum())
        u2 = np.lib.format.open_memmap(output / 'uids.unique.npy', mode='w+', dtype=UID_DTYPE, shape=(unique_n,))
        e2 = np.lib.format.open_memmap(output / 'emb.unique.npy', mode='w+', dtype=np.float16, shape=(unique_n, dim))
        positions = np.flatnonzero(keep)
        for start in range(0, unique_n, 65536):
            ix = positions[start:start+65536]
            u2[start:start+len(ix)] = sorted_uid[ix]
            e2[start:start+len(ix)] = sorted_emb[ix]
        u2.flush(); e2.flush()
        del u2,e2
        os.replace(output / 'uids.unique.npy', output / 'uids.npy')
        os.replace(output / 'emb.unique.npy', output / 'emb.npy')
        count = unique_n
    else:
        os.replace(output / 'uids.next.npy', output / 'uids.npy')
        os.replace(output / 'emb.next.npy', output / 'emb.npy')
    del raw_uid, raw_emb, sorted_uid, sorted_emb
    for name in ('uids.raw.npy','emb.raw.npy','uids.next.npy','emb.next.npy'):
        (output / name).unlink(missing_ok=True)
    (output / 'manifest.json').write_text(json.dumps({'rows': count, 'dim': dim,
        'sources': [str(p) for p in paths]}, indent=2) + '\n')


class TimbreIndex:
    def __init__(self, root: Path, expected_dim: int | None = None):
        manifest = json.loads((root / 'manifest.json').read_text())
        self.dim = manifest['dim']
        if expected_dim is not None:
            assert self.dim == expected_dim, (self.dim, expected_dim)
        self.uids = np.load(root / 'uids.npy', mmap_mode='r')
        self.emb = np.load(root / 'emb.npy', mmap_mode='r')
        assert self.emb.shape == (len(self.uids), self.dim)

    def lookup(self, uids: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        query = _uid_array(uids)
        pos = np.searchsorted(self.uids, query)
        valid = pos < len(self.uids)
        valid[valid] &= self.uids[pos[valid]] == query[valid]
        result = np.zeros((len(query), self.dim), dtype=np.float16)
        result[valid] = self.emb[pos[valid]]
        return result, valid


class TimbreChunkWriter:
    def __init__(self, output: Path, rank: int, flush_rows: int = 8192,
                 dim: int = DIM, model: str = 'Orange/Speaker-wavLM-tbr',
                 revision: str = 'b8d2608d56f18e2b6e27ba566cf69132e2c2ad6c'):
        self.output, self.rank, self.flush_rows = output, rank, flush_rows
        self.dim, self.model, self.revision = dim, model, revision
        output.mkdir(parents=True, exist_ok=True)
        self.config_path = output / 'cache_config.json'
        config = {'dim': dim, 'model': model, 'revision': revision,
                  'sample_rate': 16000, 'max_audio_samples': 480000,
                  'journal_format': 'uid-S128/emb-f16/crc32-v1'}
        if self.config_path.exists():
            assert json.loads(self.config_path.read_text()) == config, self.config_path
        else:
            temp = output / f'cache_config.rank-{rank:02d}.partial'
            temp.write_text(json.dumps(config, indent=2) + '\n')
            with temp.open('rb') as file:
                os.fsync(file.fileno())
            os.replace(temp, self.config_path)
            _sync_directory(output)
        self.seq = 1 + max((int(p.stem.rsplit('-',1)[-1]) for p in output.glob(f'rank-{rank:02d}-*.parquet')), default=-1)
        self.journal = output / f'rank-{rank:02d}.pending'
        self.record_dtype = np.dtype([('uid', UID_DTYPE), ('emb', '<f2', (dim,)),
                                      ('crc', '<u4')], align=False)
        self.uids: list[str] = []
        self.vectors: list[np.ndarray] = []
        self._recover()

    def _recover(self) -> None:
        """Replay durable records left by a killed training process."""
        if not self.journal.exists():
            return
        raw = self.journal.read_bytes()
        width = self.record_dtype.itemsize
        complete = len(raw) // width
        records = np.frombuffer(raw[:complete * width], dtype=self.record_dtype)
        good = 0
        for record in records:
            payload = record.tobytes()[:-4]
            if zlib.crc32(payload) != int(record['crc']):
                break
            good += 1
        if good != complete:
            raise ValueError(f'Corrupt sidecar journal {self.journal} at record {good}; refusing to discard generated targets')
        if complete * width != len(raw):
            with self.journal.open('r+b') as file:
                file.truncate(complete * width)
                file.flush()
                os.fsync(file.fileno())
        self.uids = [u.decode('utf-8') for u in records['uid'][:good].tolist()]
        self.vectors = [row.copy() for row in records['emb'][:good]]

    def append(self, uids: Sequence[str], emb: np.ndarray) -> None:
        assert emb.shape == (len(uids), self.dim)
        if not uids:
            return
        vectors = emb.astype(np.float16)
        records = np.empty(len(uids), dtype=self.record_dtype)
        records['uid'] = _uid_array(uids)
        records['emb'] = vectors
        for i in range(len(records)):
            records['crc'][i] = zlib.crc32(records[i].tobytes()[:-4])
        payload = records.tobytes()
        fd = os.open(self.journal, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o640)
        try:
            remaining = memoryview(payload)
            while remaining:
                written = os.write(fd, remaining)
                assert written > 0
                remaining = remaining[written:]
            os.fsync(fd)
        finally:
            os.close(fd)
        self.uids.extend(uids); self.vectors.extend(vectors)
        if len(self.uids) >= self.flush_rows:
            self.flush()

    def flush(self) -> None:
        if not self.uids:
            return
        emb = np.stack(self.vectors)
        column = pa.FixedSizeListArray.from_arrays(pa.array(emb.reshape(-1), type=pa.float16()), self.dim)
        table = pa.table({'uid': pa.array(self.uids), 'emb': column})
        table = table.replace_schema_metadata({
            b'model': self.model.encode(),
            b'revision': self.revision.encode(),
            b'sample_rate': b'16000',
            b'max_audio_samples': b'480000',
            b'embedding_dim': str(self.dim).encode(),
        })
        dst = self.output / f'rank-{self.rank:02d}-{self.seq:06d}.parquet'
        tmp = dst.with_suffix('.partial')
        pq.write_table(table, tmp, compression='zstd')
        with tmp.open('rb') as file:
            os.fsync(file.fileno())
        os.replace(tmp, dst)
        _sync_directory(self.output)
        # The Parquet rename happens first. If the process dies before this
        # truncation, journal replay may duplicate rows but never lose them.
        with self.journal.open('wb') as file:
            file.flush()
            os.fsync(file.fileno())
        self.seq += 1
        self.uids.clear(); self.vectors.clear()


def _sync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def recover_sidecars(parts: Path, dim: int) -> None:
    """Materialize any unflushed rank journals before building an index."""
    config_path = parts / 'cache_config.json'
    if not config_path.exists():
        return
    config = json.loads(config_path.read_text())
    assert config['dim'] == dim, (config['dim'], dim)
    for journal in sorted(parts.glob('rank-*.pending')):
        rank = int(journal.stem.split('-')[1])
        writer = TimbreChunkWriter(parts, rank, dim=dim, model=config['model'],
                                   revision=config['revision'])
        writer.flush()


class TimbreTargetCache:
    def __init__(self, precomputed: TimbreIndex | None, generated: TimbreIndex | None,
                 teacher: OrangeEmbeddingTeacher, writer: TimbreChunkWriter, device: torch.device):
        self.precomputed, self.generated = precomputed, generated
        self.teacher, self.writer, self.device = teacher, writer, device
        self.dim = teacher.dim
        assert writer.dim == self.dim
        assert precomputed is None or precomputed.dim == self.dim
        assert generated is None or generated.dim == self.dim
        self.recent: OrderedDict[str, np.ndarray] = OrderedDict()
        self.recent_limit = 8192
        self.computed = 0
        for uid, vector in zip(writer.uids, writer.vectors):
            self.recent[uid] = vector.copy()

    def get_or_compute(self, uids: Sequence[str], waveforms: Sequence[np.ndarray | torch.Tensor]) -> np.ndarray:
        assert len(uids) == len(waveforms)
        result = np.zeros((len(uids), self.dim), dtype=np.float16)
        have = np.zeros(len(uids), dtype=bool)
        for index in (self.precomputed, self.generated):
            if index is None: continue
            values, mask = index.lookup(uids)
            take = mask & ~have
            result[take] = values[take]; have |= take
        for i, uid in enumerate(uids):
            if not have[i] and uid in self.recent:
                result[i] = self.recent[uid]
                self.recent.move_to_end(uid)
                have[i] = True
        missing = np.flatnonzero(~have)
        if len(missing):
            values = self.teacher.encode_waveforms([waveforms[i] for i in missing], self.device)
            result[missing] = values
            self.writer.append([uids[i] for i in missing], values)
            self.computed += len(missing)
            for i, value in zip(missing, values):
                self.recent[uids[i]] = value.copy()
                if len(self.recent) > self.recent_limit:
                    self.recent.popitem(last=False)
        return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['index-emolia', 'index-identity', 'index-generated'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--parts', type=Path)
    parser.add_argument('--dim', type=int, choices=[DIM, IDENTITY_DIM], default=DIM)
    args = parser.parse_args()
    if args.command == 'index-emolia':
        build_index([EMOLIA_SOURCE], args.output)
    elif args.command == 'index-identity':
        build_index(sorted(IDENTITY_SOURCE.glob('*.parquet')), args.output,
                    allow_same_duplicates=True, dim=IDENTITY_DIM)
    else:
        assert args.parts is not None
        recover_sidecars(args.parts, args.dim)
        build_index(sorted(args.parts.glob('rank-*.parquet')), args.output,
                    allow_same_duplicates=True, dim=args.dim)


if __name__ == '__main__':
    main()
