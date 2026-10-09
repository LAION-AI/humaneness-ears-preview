#!/usr/bin/env python3
"""Incremental Gemini target parser, coverage audit, backfill queue and SFT export.

No provider calls, no teacher inference, no audio extraction. Outputs are snapshots;
the final training gate requires complete source/Gemini manifests and all sidecars.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from common import (CODE, GEMINI, HERE, RELEASE, ROOT, SELECTION, TEACHER_NAMES,
                    TarReader, emotion_key, finite, jsonl, put_jsonl, read_json, write_json)
sys.path.insert(0, str(CODE / 'gemini_s10_smoke'))
from annotate import PROMPT, contract, eligible_burst_taxonomy, validate

TASK_PROMPTS = {
    'caption': 'Describe the audible voice, delivery, emotions, vocal bursts and recording environment in English. Ground the description in this audio.',
    'director_caption': 'Write English voice-director instructions that reproduce the audible performance, including delivery, emotion and vocal bursts. Ground the instructions in this audio.',
    'burst_caption': 'Describe only the vocal burst audible from {start:.6f} to {end:.6f} seconds in the full recording. Times are relative to the beginning of the supplied audio. Use surrounding audio as context. Write a detailed English acoustic and expressive description of this event.',
}


def choose_annotations():
    expected = contract(PROMPT.read_text())
    best = {}
    attempts = 0
    for path in sorted((GEMINI / 'sidecars').glob('*.jsonl'), key=lambda x: (x.stat().st_mtime_ns, x.name)):
        for row in jsonl(path):
            attempts += 1
            annotation = row.get('annotation')
            errors = validate(annotation, expected, row['duration_s']) if isinstance(annotation, dict) else ['no JSON annotation']
            feedback = row.get('prompt_feedback') or {}
            usable = not errors and not row.get('request_error') and not feedback.get('blockReason')
            rank = (int(usable), -len(errors))
            old = best.get(row['sha256_mp3'])
            if old is None or rank >= old[0]:
                best[row['sha256_mp3']] = (rank, {**row, 'parse_validation_errors': errors, 'parser_usable': bool(usable)})
    return {key: row for key, (_, row) in best.items()}, attempts


def source_cache(selections, output):
    """Read each metadata TAR once, persist original targets in a compact SQLite file."""
    database = sqlite3.connect(output / 'source_metadata.sqlite')
    database.execute('CREATE TABLE IF NOT EXISTS metadata (sha TEXT PRIMARY KEY, doc TEXT NOT NULL)')
    known = {row[0] for row in database.execute('SELECT sha FROM metadata')}
    by_tar = defaultdict(dict)
    for row in selections:
        if row['sha256_mp3'] in known:
            continue
        for membership in row['source_memberships']:
            if membership.get('metadata_tar') and membership.get('metadata_member'):
                by_tar[membership['metadata_tar']].setdefault(membership['metadata_member'], []).append(row['sha256_mp3'])
                break
    import tarfile
    for path, wanted in by_tar.items():
        with tarfile.open(path, 'r:') as archive:
            for member in archive:
                if member.name in wanted:
                    document = json.load(archive.extractfile(member))
                    for sha in wanted[member.name]:
                        database.execute('INSERT OR REPLACE INTO metadata VALUES (?, ?)', (sha, json.dumps(document, ensure_ascii=False)))
        database.commit()
    return database


def split_group(selection, metadata):
    # Explicit speaker identities take priority; related synthetic voice variants
    # share their original family, and exact duplicates always share one SHA.
    speaker = metadata.get('speaker_id') or metadata.get('speaker') or metadata.get('voice')
    if isinstance(speaker, str) and speaker:
        return 'speaker:' + str(metadata.get('dataset', selection['source_memberships'][0]['source'])) + ':' + speaker
    uid = metadata.get('uid') or selection['source_memberships'][0].get('uid', '')
    if isinstance(uid, str) and '__' in uid:
        return 'audio-family:' + uid.split('__', 1)[0]
    return 'audio-sha256:' + selection['sha256_mp3']


def assign_split(group):
    fraction = int(hashlib.sha256(('gemini-finetune-v1:' + group).encode()).hexdigest()[:12], 16) / 16**12
    return 'test' if fraction < .05 else 'validation' if fraction < .10 else 'train'


def backfill_index():
    import tarfile
    rows = {}
    for path in sorted((ROOT / 'teacher_sidecars').glob('*.tar')):
        with tarfile.open(path, 'r:') as archive:
            for member in archive:
                if member.name.endswith('.teachers.json'):
                    doc = json.load(archive.extractfile(member))
                    # Only same-audio SHA-keyed sidecars are accepted. Array refs
                    # are bound to this immutable archive, never matched by UID.
                    sha = doc['sha256_mp3']
                    old = rows.setdefault(sha, {'scores': {}, 'embeddings': {}, 'teachers': {}})
                    old['scores'].update(doc.get('scores', {}))
                    old['teachers'].update(doc.get('teachers', {}))
                    for key, reference in doc.get('embeddings', {}).items():
                        old['embeddings'][key] = {**reference, 'tar': str(path)}
    return rows


def conversation(selection, task, prompt, target, split, event=None):
    result = {
        'schema': 'moss-caption-sft-v1', 'sha256_mp3': selection['sha256_mp3'],
        'split': split, 'task': task, 'duration_s': selection['duration_s'],
        'audio_tar': selection['audio_tar'], 'audio_member': selection['audio_member'],
        'full_audio': True,
        'conversation': [
            {'role': 'user', 'message_type': 'audio', 'content': {'tar': selection['audio_tar'], 'member': selection['audio_member']}},
            {'role': 'user', 'message_type': 'text', 'content': prompt},
            {'role': 'assistant', 'message_type': 'text', 'content': target.strip()},
        ],
    }
    if event:
        result['event_start_s'], result['event_end_s'] = event['start'], event['end']
        result['event_index'] = event['i']
    return result


def prepare(args):
    annotation_finished = (GEMINI / 'ANNOTATION_FINISHED.json').exists()
    selected_path = SELECTION / 'combined_unique_manifest.jsonl'
    if not selected_path.exists():
        selected_path = GEMINI / 'latest_selection_manifest.jsonl'
    if not selected_path.exists():
        selected_path = SELECTION / 'available_unique_manifest.jsonl'
    if args.manifest:
        selected_path = args.manifest
    annotations, attempts = choose_annotations()
    selections = list(jsonl(selected_path))
    if args.limit:
        selections = [row for row in selections if annotations.get(row['sha256_mp3'], {}).get('parser_usable')][:args.limit]
    metadata = source_cache(selections, args.out)
    teachers = backfill_index()
    normalization = read_json(RELEASE / 'training_normalization.json')
    names = normalization['score_names']
    assert len(names) == 192
    _, vn_keys, emo_keys = contract(PROMPT.read_text())
    assert {emotion_key(n[4:]) for n in names[:40]} == emo_keys
    assert {n[3:-4].lower() for n in names[40:97]} == vn_keys
    classes = read_json(RELEASE / 'classes.json')
    class_ids = {name: index for index, name in enumerate(classes['names'])}
    taxonomy = eligible_burst_taxonomy()
    class_mapping = {name: class_ids.get(classes['aliases'].get(name, name), 0) for name in taxonomy}
    write_json(args.out / 'gemini_burst_mapping.json', {'fine_to_group23': taxonomy, 'fine_to_checkpoint_class': class_mapping,
        'checkpoint_classes': classes['names'], 'unrepresented_fine_classes': [n for n, index in class_mapping.items() if not index]})
    handles = {}
    filenames = ['targets.jsonl', 'teacher_backfill_queue.jsonl', 'annotations_review.jsonl'] + [f'moss_{task}_{split}.jsonl' for task in TASK_PROMPTS for split in ('train', 'validation', 'test')]
    for name in filenames:
        handles[name] = (args.out / (name + '.incomplete')).open('w')
    counts, coverage, missing, task_counts, split_counts = Counter(), Counter(), Counter(), Counter(), Counter()
    moments = {name: [0, 0., 0.] for name in names}
    try:
        for selection in selections:
            sha = selection['sha256_mp3']
            found = metadata.execute('SELECT doc FROM metadata WHERE sha=?', (sha,)).fetchone()
            original = json.loads(found[0]) if found else {}
            gemini = annotations.get(sha)
            valid = bool(gemini and gemini['parser_usable'])
            answer = gemini['annotation'] if valid else {}
            computed = teachers.get(sha, {})
            scores = {name: value for name, value in (original.get('scores') or {}).items() if name in names and finite(value)}
            provenance = {name: 'original_source_metadata' for name in scores}
            for name, value in computed.get('scores', {}).items():
                if name in names and finite(value):
                    scores[name], provenance[name] = value, 'teacher_sidecar'
            if valid:
                for name in names[:40]:
                    scores[name], provenance[name] = answer['emonet'][emotion_key(name[4:])], 'gemini-3.8-flash'
                for name in names[40:97]:
                    scores[name], provenance[name] = answer['voicenet'][name[3:-4].lower()], 'gemini-3.8-flash'
                scores['genuineness_0_6'] = answer['genuineness']
                provenance['genuineness_0_6'] = 'gemini-3.8-flash'
                if answer['vocal_burst_blend'] is not None:
                    scores['blend_0_10'] = answer['vocal_burst_blend']
                    provenance['blend_0_10'] = 'gemini-3.8-flash'
                else:
                    scores.pop('blend_0_10', None)
                    provenance.pop('blend_0_10', None)
                for name in names[131:]:
                    key = emotion_key(name.split(':', 1)[1])
                    if key in answer['emonet']:
                        scores[name], provenance[name] = answer['emonet'][key], 'gemini-3.8-flash'
            events = []
            if valid:
                # Provider allows +0.3 s end tolerance; retain raw timestamps and
                # clamp training spans at actual audio end, recording the clamp.
                for event in answer['vocal_bursts']:
                    if event['start'] >= selection['duration_s']:
                        continue
                    events.append({**event, 'end': min(event['end'], selection['duration_s']),
                        'raw_end': event['end'], 'checkpoint_class_id': class_mapping[event['fine']],
                        'timestamp_clamped': event['end'] > selection['duration_s']})
                scores['burst_count_log1p'] = math.log1p(len(events))
                provenance['burst_count_log1p'] = 'gemini-3.8-flash'
            group = split_group(selection, original)
            split = assign_split(group)
            split_counts[split] += 1
            counts['selected'] += 1
            counts['gemini_valid'] += int(valid)
            counts['gemini_pending_or_invalid'] += int(not valid)
            embeddings = computed.get('embeddings', {})
            required = []
            # Originals keep their scalars. Backfill the additional objectives
            # from the exact selected waveform, not an upstream speaker UID.
            checks = {
                'orange': all(embeddings.get(n, {}).get('dim') == d for n, d in [('timbre', 128), ('identity', 250)]),
                'dnsmos': all(n in scores for n in names[123:130]),
                'audiobox': all(n in scores for n in names[119:123]),
                'voiceclap': all(n in scores for n in names[131:]) and embeddings.get('voiceclap', {}).get('dim') == 768,
                'empathic': all(n in scores for n in names[100:119]) and embeddings.get('bud_e_pooled', {}).get('dim') == 3072,
            }
            for teacher, complete in checks.items():
                # All selected audio receives independent teacher annotations,
                # even if its Flash response is pending or ultimately excluded.
                if not complete:
                    required.append(teacher)
                    missing[teacher] += 1
            transcript = answer.get('transcript', '') if valid else original.get('source_transcript', '')
            speech_valid = bool(re.sub(r'\[[^\]]*\]', '', transcript).strip()) if isinstance(transcript, str) else False
            # Teacher inference can finish before Flash. Recompute the loss-domain
            # masks from the latest annotation, rather than freezing an unknown
            # pending transcript as a permanent non-speech/speaker exclusion.
            explicit_speaker = original.get('speaker_id') or original.get('speaker') or original.get('voice')
            source_types = {m['source'] for m in selection['source_memberships']}
            one_explicit_speaker = isinstance(explicit_speaker, (str, int)) and bool(explicit_speaker)
            single_speaker = bool(one_explicit_speaker or 'talent' in source_types or
                ('S1_100h' in source_types and '/synth' in str(original.get('language_source', ''))))
            if original.get('speaker_count') is not None:
                single_speaker = original['speaker_count'] == 1
            validity_source = 'explicit_metadata_or_single_voice_synthetic_source' if single_speaker else 'unknown_masked'
            teacher_metadata = {name: dict(doc) for name, doc in computed.get('teachers', {}).items()}
            for family, field, value in (
                ('orange', 'valid_single_speaker_target', single_speaker and speech_valid),
                ('dnsmos', 'valid_for_speech', speech_valid),
                ('voiceclap', 'speech_domain_valid', speech_valid),
                ('empathic', 'speech_domain_valid', speech_valid),
            ):
                if family in teacher_metadata:
                    doc = teacher_metadata[family]
                    doc['validity_at_teacher_annotation'] = doc.get(field)
                    doc[field] = value
                    doc['current_validity_source'] = 'latest_valid_flash_transcript_and_source_metadata' if valid else 'original_source_metadata'
            if 'orange' in teacher_metadata:
                teacher_metadata['orange']['speaker_validity_source'] = validity_source
            if required:
                put_jsonl(handles['teacher_backfill_queue.jsonl'], {**selection, 'split': split, 'missing_teachers': required,
                    'gemini_language': answer.get('language'), 'gemini_transcript': answer.get('transcript'),
                    'gemini_valid': valid, 'speech_target_valid': speech_valid,
                    'speaker_target_valid': single_speaker and speech_valid,
                    'speaker_validity_source': validity_source})
            cps = None
            if valid and isinstance(answer.get('transcript'), str):
                # Literal Unicode character count, including spaces and punctuation;
                # [unclear] placeholders are removed, denominator is full clip length.
                text = re.sub(r'\[[^\]]*\]', '', answer['transcript']).strip()
                cps = len(text) / selection['duration_s']
            if cps is None and finite(original.get('source_characters_per_second')):
                cps = original['source_characters_per_second']
            row = {**selection, 'schema': 'gemini-whisper-targets-v1', 'split': split, 'split_group': group,
                'gemini_usable': valid, 'annotation': answer or None,
                'annotation_provenance': {key: gemini.get(key) for key in ('annotation_model', 'prompt_sha256', 'batch_tag', 'batch_name')} if gemini else None,
                'raw_scores': scores, 'score_provenance': provenance,
                'vocal_bursts': events, 'burst_frames_valid': valid,
                'characters_per_second': cps,
                'cps_source': 'gemini_transcript' if valid else 'original_source_metadata' if cps is not None else None,
                'embeddings': embeddings, 'missing_teachers': required,
                'teacher_metadata': teacher_metadata,
                'speaker_target_policy': 'same_audio_teacher_vector; multi_speaker_or_non_speech_is_masked',
                'original_annotation': original,
                'ready_for_whisper_training': valid and not required,
            }
            put_jsonl(handles['targets.jsonl'], row)
            counts['ready_for_whisper_training'] += int(row['ready_for_whisper_training'])
            for name, value in scores.items():
                coverage[name] += 1
                if split == 'train' and valid:
                    moment = moments[name]
                    moment[0] += 1
                    delta = value - moment[1]
                    moment[1] += delta / moment[0]
                    moment[2] += delta * (value - moment[1])
            if gemini and not valid:
                put_jsonl(handles['annotations_review.jsonl'], {'sha256_mp3': sha, 'errors': gemini['parse_validation_errors'],
                    'batch_tag': gemini.get('batch_tag'), 'prompt_feedback': gemini.get('prompt_feedback')})
            if valid:
                for task in ('caption', 'director_caption'):
                    if not answer[task].strip():
                        continue
                    example = conversation(selection, task, TASK_PROMPTS[task], answer[task], split)
                    put_jsonl(handles[f'moss_{task}_{split}.jsonl'], example)
                    task_counts[f'{task}:{split}'] += 1
                for event in events:
                    example = conversation(selection, 'burst_caption', TASK_PROMPTS['burst_caption'].format(**event), event['caption'], split, event)
                    put_jsonl(handles[f'moss_burst_caption_{split}.jsonl'], example)
                    task_counts[f'burst_caption:{split}'] += 1
    finally:
        metadata.close()
        for handle in handles.values():
            handle.close()
    for name in filenames:
        os.replace(args.out / (name + '.incomplete'), args.out / name)
    write_json(args.out / 'checkpoint_normalization.json', normalization)
    write_json(args.out / 'gemini_train_statistics.json', {
        'basis': 'current training partition only; provisional until final source/annotation completion',
        'score_names': names, 'count': [moments[n][0] for n in names],
        'mean': [moments[n][1] for n in names],
        'std': [max(1e-4, math.sqrt(moments[n][2] / max(1, moments[n][0]))) for n in names],
        'default_training_normalization': 'checkpoint_normalization.json; preserves existing output units; emotions remain 0..4',
    })
    complete = (GEMINI / 'ANNOTATION_FINISHED.json').exists() and (SELECTION / 'COMBINED_READY.json').exists() and not args.limit
    ready = complete and counts['gemini_valid'] > 0 and not missing
    summary = {'schema': 'gemini-finetune-preparation-v1', 'updated_utc': datetime.now(timezone.utc).isoformat(),
        'selected_manifest': str(selected_path), 'snapshot_only': not complete,
        'gemini_annotations_finished': (GEMINI / 'ANNOTATION_FINISHED.json').exists(),
        'counts': dict(counts), 'splits': dict(split_counts), 'teacher_backfill_missing': dict(missing),
        'valid_score_counts': dict(coverage), 'moss_examples': dict(task_counts),
        'annotation_attempts_seen': attempts, 'ready_to_train': ready,
        'training_gate': 'Final source and Gemini manifests, complete teacher backfill for all selected clips. Invalid/blocked Flash responses are quarantined and excluded from tuning, with their IDs and reasons retained.',
        'gemini_priority': 'Strictly schema-valid Flash annotation supersedes matching prior scores and event spans; original labels retained for audit.',
        'output': str(args.out)}
    write_json(args.out / 'coverage.json', summary)
    write_json(args.out / 'excluded_annotations.json', {
        'policy': 'schema-invalid, provider-blocked, or missing final responses are excluded from new tuning and all caption tasks',
        'final': complete, 'clips': counts['gemini_pending_or_invalid']})
    caption_marker = args.out / 'CAPTION_DATA_READY.json'
    if complete and counts['gemini_valid'] > 0:
        write_json(caption_marker, {'caption_examples': dict(task_counts), 'unique_valid_clips': counts['gemini_valid'],
            'excluded_invalid_clips': counts['gemini_pending_or_invalid'], 'source_manifest': str(selected_path)})
    elif caption_marker.exists():
        caption_marker.unlink()
    # Invalid responses remain in the review ledger and never become targets.
    marker = args.out / 'TRAINING_READY.json'
    if ready:
        write_json(marker, summary)
    elif marker.exists():
        marker.unlink()
    print(json.dumps({key: summary[key] for key in ('counts', 'teacher_backfill_missing', 'moss_examples', 'ready_to_train', 'output')}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=ROOT / 'prepared')
    parser.add_argument('--limit', type=int, default=0, help='small real-data parser canary; never marks training ready')
    parser.add_argument('--manifest', type=Path, help='explicit existing selected manifest for a small canary')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / '.prepare.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        prepare(args)


if __name__ == '__main__':
    main()
