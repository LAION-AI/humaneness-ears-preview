#!/usr/bin/env python3
"""Join P3 teacher JSONs into the Ladder target schema and extra task masks.

The five sidecars are immutable WebDataset companions to an audio TAR. This
adapter is also a contract check for future streaming loaders. It reads JSON
only; arrays remain in their sidecar TARs and are loaded on demand.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import re
import tarfile
from collections import Counter
from pathlib import Path

LADDER_COLUMNS = Path('/e/scratch/reformo/schuhmann1_moss/publish/scaling_ladder/COLUMNS.md')
EIV_ALIAS = {'Jealousy_and_Envy': 'Jealousy_&_Envy'}
DNSSCORES = ('SIG_raw', 'BAK_raw', 'OVRL_raw', 'SIG', 'BAK', 'OVRL', 'P808_MOS')
AUDIOBOX = ('CE', 'CU', 'PC', 'PQ')
SIDECARS = {
    'voiceclap': '.scores.json', 'audiobox': '.audiobox.json',
    'dnsmos': '.dnsmos.json', 'orange': '.orange.json',
    'empathic': '.eiv.json',
}


def target_names(columns_path: Path = LADDER_COLUMNS) -> list[str]:
    columns = re.findall(r'^\| `([^`]+)`', columns_path.read_text(), re.M)
    emo = [x for x in columns if x.startswith('emo_')]
    voice = [x for x in columns if x.startswith('vn_') and x.endswith('_reg')]
    assert len(emo) == 40 and len(voice) == 57, (len(emo), len(voice))
    return emo + voice + ['genuineness_0_6', 'blend_0_10', 'R_quality']


def json_members(path: Path, suffix: str) -> dict[str, dict]:
    rows = {}
    with tarfile.open(path, 'r:') as tf:
        for member in tf:
            if member.isfile() and member.name.endswith(suffix):
                key = member.name[:-len(suffix)]
                if key in rows:
                    raise RuntimeError(f'duplicate key {key} in {path}')
                rows[key] = json.load(tf.extractfile(member))
    return rows


def rasterize_burst_frames(frame_target: dict) -> tuple[list[int], list[int]]:
    """Rasterize event/sample intervals at 20 ms with explicit valid frames."""
    hop = int(frame_target['hop_samples'])
    end = int(frame_target['end_sample'])
    count = (end + hop - 1) // hop
    labels = [0] * count
    valid = [int(bool(frame_target['valid']))] * count
    for start, stop in frame_target['positive_intervals_samples']:
        for frame in range(start // hop, min(count, (stop + hop - 1) // hop)):
            labels[frame] = 1
    return labels, valid


def union_samples(intervals: list[list[int]]) -> int:
    merged_end = total = 0
    for start, stop in sorted(intervals):
        total += max(0, stop - max(start, merged_end))
        merged_end = max(merged_end, stop)
    return total


def adapt(meta: dict, sidecars: dict[str, dict], names: list[str],
          extra_eiv_names: list[str], locator: dict | None) -> dict:
    key = meta['sample_id']
    for kind, doc in sidecars.items():
        if doc['sample_id'] != key:
            raise RuntimeError(f'{key}: {kind} sidecar key mismatch')
    voice = sidecars['voiceclap']
    attributes = voice.get('voiceclap_attributes') or {}
    eiv = sidecars['empathic']
    dns = sidecars['dnsmos']
    box = sidecars['audiobox']
    orange = sidecars['orange']
    speech = bool(voice['validity']['speech_scores'] and eiv['speech_domain_valid'])
    values = {}
    for name in names:
        if name.startswith('emo_'):
            score = eiv['scores'].get(EIV_ALIAS.get(name[4:], name[4:])) if speech else None
        elif name.startswith('vn_'):
            dimension = name[3:-4]
            score = voice['voicenet'].get(dimension, {}).get('regression') if speech else None
        elif name == 'genuineness_0_6':
            score = voice.get(name) if speech else None
        elif name == 'blend_0_10':
            score = voice.get('vocalburst_blend_0_10') if voice['validity']['vocalburst_blend'] else None
        else:
            # R_quality is cell-normalized using the Ladder selection pool;
            # that calibration does not exist for a P3 synthetic mixture.
            score = None
        values[name] = score if isinstance(score, (int, float)) and math.isfinite(score) else None
    if speech and any(values[name] is None for name in names[:97]):
        raise RuntimeError(f'{key}: incomplete speech emotion/VoiceNet targets')
    extra = {name: (eiv['scores'] or {}).get(name) for name in extra_eiv_names}
    if len(extra) != 19:
        raise RuntimeError(f'{key}: expected 19 non-Ladder EIV values, got {len(extra)}')
    n = int(meta['n_vocal_bursts'])
    if n != len(meta['events']):
        raise RuntimeError(f'{key}: count and event list disagree')
    positive = n > 0
    # Pilot locator flags are a quarantine signal, not a release certificate.
    negative_screened = bool(locator and not locator['n_locator_spans'])
    frame_valid = positive or negative_screened
    end_sample = round(float(meta['duration_s']) * int(meta['sample_rate']))
    intervals = []
    for event in meta['events']:
        a, b = int(event['start_sample']), int(event['end_sample'])
        if not 0 <= a < b <= end_sample:
            raise RuntimeError(f'{key}: invalid event interval {(a, b)}')
        intervals.append([a, b])
    speech_intervals = [[int(c['start_sample']), int(c['end_sample'])]
                        for c in meta['components'] if c['role'] == 'speech']
    intersects = [[max(a, c), min(b, d)] for a, b in intervals
                  for c, d in speech_intervals if min(b, d) > max(a, c)]
    burst_covered = union_samples(intervals)
    empty = meta.get('known_empty_intervals_samples')
    auxiliary = {
        'duration_s': float(meta['duration_s']),
        'burst_rate_per_minute': 60 * n / float(meta['duration_s']),
        'burst_occupancy_fraction': burst_covered / end_sample,
        'burst_speech_overlap_fraction':
            union_samples(intersects) / burst_covered if burst_covered else 0.0,
        'speech_component_count': len(speech_intervals),
        'known_empty_fraction': union_samples(empty) / end_sample if empty is not None else None,
        'has_fx': int(any(c['role'] == 'fx' for c in meta['components'])),
        'has_music': int(any(c['role'] == 'music' for c in meta['components'])),
        'characters_per_second': None,
        'characters_per_second_teacher':
            attributes.get('talking_speed')
            if voice['validity'].get('attribute_speech_domain') else None,
    }
    return {
        'sample_id': key, 'ladder_scores': values,
        'extra_eiv_scores': extra if speech else {name: None for name in extra},
        'audiobox_scores': {name: box['scores'].get(name) for name in AUDIOBOX},
        'dnsmos_scores': {name: dns['scores'].get(name) if dns['valid_for_speech'] else None
                          for name in DNSSCORES},
        'burst_count_log1p': math.log1p(n),
        'burst_frame_target': {'hop_samples': 320, 'end_sample': end_sample,
                               'positive_intervals_samples': intervals,
                               'valid': frame_valid,
                               'negative_screening': 'locator_unflagged_provisional'
                               if negative_screened and not positive else
                               'quarantined' if not positive else 'synthetic_insertions'},
        'negative_subtype': meta.get('negative_subtype'),
        'known_empty_intervals_samples': meta.get('known_empty_intervals_samples', []),
        'auxiliary_targets': auxiliary,
        'orange_timbre': {'file': orange['timbre_file'], 'dim': 128,
                          'valid_single_speaker': orange['valid_single_speaker_target']},
        'orange_identity': {'file': orange['identity_file'], 'dim': 250,
                            'valid_single_speaker': orange['valid_single_speaker_target']},
        'voiceclap_embedding': voice['voiceclap_embedding'],
        'voiceclap_attribute_scores': attributes,
        'voiceclap_attributes_valid_for_speech':
            bool(voice['validity'].get('attribute_speech_domain')),
        'bud_e_embedding': eiv['pooled_embedding_file'],
        'teacher_revisions': {'voiceclap': voice['teacher_revisions'],
                              'audiobox': box['teacher_revision'],
                              'orange': orange['teacher_revisions'],
                              'eiv': eiv['teacher'], 'dnsmos': dns['teacher']},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--audio-tar', type=Path, required=True)
    for kind in SIDECARS:
        ap.add_argument('--' + kind, type=Path, required=True)
    ap.add_argument('--locator-audit', type=Path)
    ap.add_argument('--report', type=Path, required=True)
    ap.add_argument('--targets-sidecar', type=Path,
                    help='optional key-aligned WebDataset TAR with all joined targets and 20-ms frames')
    args = ap.parse_args()
    names = target_names()
    audio = json_members(args.audio_tar, '.json')
    side = {kind: json_members(getattr(args, kind), suffix)
            for kind, suffix in SIDECARS.items()}
    for kind, rows in side.items():
        if set(rows) != set(audio):
            raise RuntimeError(f'{kind}: sidecar keys differ from audio keys')
    locator = {}
    if args.locator_audit:
        locator = {row['sample_id']: row for row in
                   json.loads(args.locator_audit.read_text())['rows']}
        if set(locator) != set(audio):
            raise RuntimeError('locator audit keys differ from audio keys')
    example_scores = next((doc['scores'] for doc in side['empathic'].values()
                           if doc['scores'] is not None), None)
    if example_scores is None:
        raise RuntimeError('no EIV score schema in sidecars')
    mapped_eiv = {EIV_ALIAS.get(name[4:], name[4:]) for name in names[:40]}
    extra_eiv_names = sorted(set(example_scores) - mapped_eiv)
    targets = [adapt(audio[key], {kind: rows[key] for kind, rows in side.items()},
                     names, extra_eiv_names, locator.get(key)) for key in sorted(audio)]
    if args.targets_sidecar:
        args.targets_sidecar.parent.mkdir(parents=True, exist_ok=True)
        tmp = args.targets_sidecar.with_suffix('.tar.incomplete')
        with tarfile.open(tmp, 'w') as tf:
            for row in targets:
                labels, mask = rasterize_burst_frames(row['burst_frame_target'])
                payload = {**row, 'burst_frame_labels': labels,
                           'burst_frame_valid_mask': mask}
                encoded = json.dumps(payload, separators=(',', ':')).encode()
                member = tarfile.TarInfo(row['sample_id'] + '.targets.json')
                member.size, member.mtime = len(encoded), 0
                tf.addfile(member, io.BytesIO(encoded))
        os.replace(tmp, args.targets_sidecar)
    valid = Counter()
    frame_positive = frame_valid = 0
    for row in targets:
        for key, value in row['ladder_scores'].items():
            if value is not None:
                valid[key] += 1
        frame_labels, frame_mask = rasterize_burst_frames(row['burst_frame_target'])
        frame_positive += sum(a * b for a, b in zip(frame_labels, frame_mask))
        frame_valid += sum(frame_mask)
    result = {'samples': len(targets), 'ladder_target_count': len(names),
              'ladder_valid_min': min(valid[name] for name in names),
              'ladder_valid_max': max(valid[name] for name in names),
              'emotion_valid': valid[names[0]], 'voicenet_valid': valid[names[40]],
              'genuineness_valid': valid['genuineness_0_6'],
              'blend_valid': valid['blend_0_10'], 'R_quality_valid': valid['R_quality'],
              'audiobox_valid': sum(all(v is not None for v in r['audiobox_scores'].values())
                                     for r in targets),
              'dnsmos_valid': sum(all(v is not None for v in r['dnsmos_scores'].values())
                                   for r in targets),
              'extra_eiv_dimensions': len(targets[0]['extra_eiv_scores']),
              'orange_single_speaker_valid': sum(r['orange_timbre']['valid_single_speaker']
                                                 for r in targets),
              'frame_target_valid': sum(r['burst_frame_target']['valid'] for r in targets),
              'frame_target_quarantined': sum(not r['burst_frame_target']['valid'] for r in targets),
              'valid_frame_count': frame_valid, 'positive_frame_count': frame_positive,
              'positive_frame_fraction': frame_positive / frame_valid if frame_valid else None,
              'auxiliary_target_count': len(targets[0]['auxiliary_targets']),
              'voiceclap_attribute_dimensions':
                  len(targets[0]['voiceclap_attribute_scores']),
              'auxiliary_valid': {name: sum(r['auxiliary_targets'][name] is not None
                                            for r in targets)
                                  for name in targets[0]['auxiliary_targets']},
              'target_schema': 'p3-ladder-multitask-v1',
              'targets_sidecar': str(args.targets_sidecar) if args.targets_sidecar else None,
              'targets_sidecar_samples': len(targets) if args.targets_sidecar else 0,
              'status': 'schema_join_pilot; locator-only negative screening is provisional'}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
