#!/usr/bin/env python3
"""Sequential, resumable Gemini audio annotation; never writes the API key."""
from __future__ import annotations

import argparse
import base64
import getpass
import json
import math
import os
import re
import time
import urllib.error
import urllib.request
from functools import lru_cache
from pathlib import Path

ROOT = Path('/e/scratch/reformo/schuhmann1_moss/whisper_score_regression/gemini_s10_smoke_20261004')
PROMPT = Path(__file__).with_name('gemini_voice_annotation_master_prompt_detailed.txt')
MODEL = 'gemini-3.8-flash'
BASE = 'https://generativelanguage.googleapis.com/v1beta'


def request(key: str, route: str, body: dict | None = None, timeout: int = 300) -> dict:
    payload = None if body is None else json.dumps(body, ensure_ascii=False).encode()
    req = urllib.request.Request(BASE + route, data=payload,
                                 headers={'x-goog-api-key': key,
                                          'Content-Type': 'application/json'},
                                 method='GET' if body is None else 'POST')
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read(4000).decode('utf-8', 'replace')
        raise RuntimeError(f'Gemini HTTP {exc.code}: {detail}') from None


def contract(prompt: str) -> tuple[set[str], set[str], set[str]]:
    start = prompt.index('{\n  "language"')
    end = prompt.index('\n\nFORMAT RULES:', start)
    example = json.loads(prompt[start:end])
    assert len(example['voicenet']) == 57, len(example['voicenet'])
    assert len(example['emonet']) == 40, len(example['emonet'])
    return set(example), set(example['voicenet']), set(example['emonet'])


@lru_cache(maxsize=1)
def eligible_burst_taxonomy() -> dict[str, str]:
    """Read the exact fine-to-group mapping from the versioned master prompt."""
    mapping = dict(re.findall(r'^- (.*?) \| group23=([a-z_]+) \|',
                              PROMPT.read_text(), re.MULTILINE))
    non_vocal = {'Finger Snaps', 'Hand Scratching Head', 'Hand Slaps', 'Slap Face'}
    assert len(mapping) == 82 and non_vocal <= mapping.keys(), 'Unexpected burst taxonomy'
    return {fine: group for fine, group in mapping.items() if fine not in non_vocal}


def validate(answer: dict, expected: tuple[set[str], set[str], set[str]], duration: float) -> list[str]:
    top, vn, emo = expected
    errors = []
    if not isinstance(answer, dict):
        return ['Response is not a JSON object']
    if set(answer) != top:
        errors.append(f'Top-level keys differ: missing={sorted(top-set(answer))}, extra={sorted(set(answer)-top)}')
    for field, keys, maximum in [('voicenet', vn, 6), ('emonet', emo, 4)]:
        values = answer.get(field)
        if not isinstance(values, dict) or set(values) != keys:
            errors.append(f'{field} keys/count invalid')
            continue
        for key, value in values.items():
            limit = 4 if field == 'voicenet' and key == 'bkgn' else 2 if field == 'voicenet' and key == 'expl' else maximum
            if type(value) is not int or not 0 <= value <= limit:
                errors.append(f'{field}.{key} outside integer 0..{limit}')
    for field, maximum in [('genuineness', 6), ('vocal_burst_count', 100)]:
        value = answer.get(field)
        if type(value) is not int or not 0 <= value <= maximum:
            errors.append(f'{field} invalid')
    bursts = answer.get('vocal_bursts')
    if not isinstance(bursts, list):
        errors.append('vocal_bursts must be an array')
    elif answer.get('vocal_burst_count') != len(bursts):
        errors.append('vocal_burst_count differs from array length')
    else:
        taxonomy = eligible_burst_taxonomy()
        previous_start = -1.0
        for index, burst in enumerate(bursts, 1):
            if not isinstance(burst, dict) or set(burst) != {'i', 'start', 'end', 'fine', 'group23', 'caption'}:
                errors.append(f'burst {index}: keys invalid')
                continue
            start, end = burst['start'], burst['end']
            if (burst['i'] != index or not isinstance(start, (int, float))
                    or not isinstance(end, (int, float)) or not math.isfinite(start)
                    or not math.isfinite(end) or not 0 <= start < end <= duration + 0.3):
                errors.append(f'burst {index}: index or timing invalid')
            if isinstance(start, (int, float)) and start < previous_start:
                errors.append(f'burst {index}: events are not chronological')
            if isinstance(start, (int, float)):
                previous_start = start
            fine = burst['fine']
            if not isinstance(fine, str) or fine not in taxonomy:
                errors.append(f'burst {index}: ineligible fine label {fine!r}')
            elif burst['group23'] != taxonomy[fine]:
                errors.append(f'burst {index}: wrong group23 for {fine!r}')
            if not isinstance(burst['caption'], str) or not burst['caption'].strip():
                errors.append(f'burst {index}: caption is empty')
    blend = answer.get('vocal_burst_blend')
    if blend is not None and (type(blend) is not int or not 0 <= blend <= 10):
        errors.append('vocal_burst_blend invalid')
    if isinstance(bursts, list) and bool(bursts) != (blend is not None):
        errors.append('vocal_burst_blend null/presence mismatch')
    for field in ('language', 'accent', 'transcript', 'caption', 'director_caption'):
        if not isinstance(answer.get(field), str):
            errors.append(f'{field} must be a string')
    return errors


def annotate_one(key: str, prompt: str, sample: dict, cache_name: str | None) -> dict:
    audio = (ROOT / sample['audio_file']).read_bytes()
    body = {
        'contents': [{'role': 'user', 'parts': [
            {'text': 'Analyze the complete audio file. Return exactly the required JSON object.'},
            {'inlineData': {'mimeType': 'audio/mp3', 'data': base64.b64encode(audio).decode()}}
        ]}],
        'generationConfig': {'temperature': 0, 'responseMimeType': 'application/json',
                             'maxOutputTokens': 8192,
                             'thinkingConfig': {'thinkingLevel': 'LOW'}},
    }
    if cache_name:
        body['cachedContent'] = cache_name
    else:
        body['systemInstruction'] = {'parts': [{'text': prompt}]}
    started = time.monotonic()
    response = request(key, f'/models/{MODEL}:generateContent', body)
    elapsed = time.monotonic() - started
    candidates = response.get('candidates') or []
    text = ''.join(part.get('text', '') for candidate in candidates
                   for part in candidate.get('content', {}).get('parts', []))
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None
    return {'id': sample['id'], 'model': MODEL, 'elapsed_s': round(elapsed, 2),
            'usage': response.get('usageMetadata', {}),
            'prompt_feedback': response.get('promptFeedback'),
            'finish_reasons': [candidate.get('finishReason') for candidate in candidates],
            'response_text': text, 'annotation': parsed,
            'validation_errors': validate(parsed, contract(prompt), sample['duration_s'])}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=25)
    parser.add_argument('--only', nargs='*', help='retry only listed sample IDs')
    parser.add_argument('--key-file', type=Path,
                        help='existing mode-0600 Google AI Studio key file; never copied into outputs')
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    prompt = PROMPT.read_text()
    contract(prompt)
    key = (args.key_file.read_text().strip() if args.key_file else
           os.environ.get('GEMINI_API_KEY') or getpass.getpass('Google AI Studio API key (hidden): '))
    model_info = request(key, f'/models/{MODEL}', timeout=20)
    print('AUTH_OK', model_info.get('name'), flush=True)
    cache_name = None
    try:
        cache = request(key, '/cachedContents', {
            'model': f'models/{MODEL}',
            'systemInstruction': {'parts': [{'text': prompt}]},
            'contents': [{'role': 'user', 'parts': [{'text': 'Apply the cached annotation contract to the next complete audio file.'}]}],
            'ttl': '3600s',
        }, timeout=60)
        cache_name = cache['name']
        print('PROMPT_CACHE_OK', cache_name, 'tokens', cache.get('usageMetadata', {}).get('totalTokenCount'), flush=True)
    except Exception as exc:
        print('PROMPT_CACHE_UNAVAILABLE; using inline system prompt:', str(exc)[:240], flush=True)
    (ROOT / 'responses').mkdir(exist_ok=True)
    for sample in manifest['samples'][:args.limit]:
        if args.only and sample['id'] not in args.only:
            continue
        path = ROOT / 'responses' / f"{sample['id']}.json"
        if path.exists():
            prior = json.loads(path.read_text())
            if not prior.get('validation_errors'):
                print('SKIP_VALID', sample['id'], flush=True)
                continue
        for attempt in range(2):
            try:
                result = annotate_one(key, prompt, sample, cache_name)
                path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
                print('DONE', sample['id'], 'attempt', attempt + 1,
                      'seconds', result['elapsed_s'], 'errors', len(result['validation_errors']),
                      'bursts', len((result['annotation'] or {}).get('vocal_bursts', [])), flush=True)
                if not result['validation_errors']:
                    break
            except Exception as exc:
                print('ERROR', sample['id'], type(exc).__name__, str(exc)[:350], flush=True)
                if attempt == 1:
                    path.write_text(json.dumps({'id': sample['id'], 'error': str(exc)[:2000],
                                                'validation_errors': ['API request failed']}, indent=2) + '\n')
                else:
                    time.sleep(3)
    files = sorted((ROOT / 'responses').glob('S10-*.json'))
    valid = sum(not json.loads(p.read_text()).get('validation_errors') for p in files)
    print('COMPLETE', valid, 'valid of', len(files), flush=True)


if __name__ == '__main__':
    main()
