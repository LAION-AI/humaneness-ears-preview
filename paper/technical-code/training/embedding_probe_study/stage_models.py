#!/usr/bin/env python3
"""Stage pinned public weights with two resumable streams before offline jobs."""
import concurrent.futures
import os
import time
import urllib.request
from datetime import datetime, timezone
from study_paths import ROOT, read, write


def fetch(task):
    url, path, size = task
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size == size:
        return
    part = path.with_name(path.name + '.part')
    for attempt in range(5):
        offset = part.stat().st_size if part.exists() else 0
        if offset == size:
            part.replace(path)
            return
        request = urllib.request.Request(url, headers={'Range': f'bytes={offset}-'} if offset else {})
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                if offset and response.status != 206:
                    offset = 0
                began = time.monotonic()
                sent = 0
                with part.open('ab' if offset else 'wb') as stream:
                    while chunk := response.read(1024 * 1024):
                        stream.write(chunk)
                        sent += len(chunk)
                        delay = sent / (8 * 1024 * 1024) - (time.monotonic() - began)
                        if delay > 0:
                            time.sleep(delay)
            if part.stat().st_size != size:
                raise RuntimeError('Incomplete pinned file')
            part.replace(path)
            return
        except Exception:
            if attempt == 4:
                raise RuntimeError('Pinned download failed: ' + path.name) from None
            time.sleep(min(30, 2 ** attempt))


def main():
    cfg = read(ROOT / 'study.json')
    tasks = []
    for model in cfg['models']:
        if not model.get('repo'):
            continue
        repo, revision = model['repo'], model['revision']
        with urllib.request.urlopen(f'https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true', timeout=30) as response:
            info = __import__('json').load(response)
        if info['sha'] != revision:
            raise RuntimeError('Model revision mismatch')
        folder = model.get('subfolder', '')
        for file in info['siblings']:
            name = file['rfilename']
            if folder and not name.startswith(folder + '/'):
                continue
            if not name.endswith(('.json', '.safetensors', '.model', '.jinja', '.py', '.txt', '.md')):
                continue
            if name.startswith('adapter/') or name.endswith('zero_to_fp32.py'):
                continue  # VoiceCLAP root is the merged released backbone.
            size = file.get('size') or file.get('lfs', {}).get('size')
            if size is None:
                raise RuntimeError('Missing pinned size metadata')
            local = ROOT / 'models' / model['id'] / name
            tasks.append((f'https://huggingface.co/{repo}/resolve/{revision}/{name}', local, size))
        write(ROOT / 'models' / model['id'] / 'MODEL_SOURCE.json', {'repo': repo, 'revision': revision, 'subfolder': folder})
    write(ROOT / 'staging_status.json', {'state': 'DOWNLOADING', 'pid': os.getpid(), 'files': len(tasks),
                                       'bytes': sum(t[2] for t in tasks), 'streams': 2,
                                       'started_utc': datetime.now(timezone.utc).isoformat()})
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        for _ in pool.map(fetch, tasks):
            pass
    write(ROOT / 'MODELS_READY.json', {'files': len(tasks), 'bytes': sum(t[2] for t in tasks),
                                       'finished_utc': datetime.now(timezone.utc).isoformat()})
    write(ROOT / 'staging_status.json', {'state': 'COMPLETE', 'files': len(tasks),
                                       'bytes': sum(t[2] for t in tasks), 'updated_utc': datetime.now(timezone.utc).isoformat()})


if __name__ == '__main__':
    main()
