#!/usr/bin/env python3
"""Publish the requested main benchmark report and its exact result inventory."""
import hashlib
import os
import re
from pathlib import Path

# Compute inference stays offline; this separate publication process needs Hub
# access and inherits the compute launcher environment on the final step.
os.environ.pop('HF_HUB_OFFLINE', None)
os.environ.pop('TRANSFORMERS_OFFLINE', None)
from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

from benchmark_supplement import BUNDLE, HOME, MANIFEST, ROOT, read
from study_paths import HERE, write


def main():
    repo = 'laion/whisper-base-small-emotion-voice-burst'
    report = HOME / 'WHISPER_BASE_SMALL_COMPREHENSIVE_PUBLIC_BENCHMARKS_2026-10-04.html'
    manifest = read(MANIFEST)
    if manifest['probe_runs'] != 36 or manifest['whisper_runs'] != 4:
        raise RuntimeError('Incomplete consolidated result inventory')
    api = HfApi()
    info = api.repo_info(repo_id=repo, repo_type='space')
    original = Path(hf_hub_download(repo, 'README.md', repo_type='space', revision=info.sha)).read_text()
    begin, end = '<!-- CONSOLIDATED_BENCHMARK_20261007_START -->', '<!-- CONSOLIDATED_BENCHMARK_20261007_END -->'
    section = begin + '''
## Completed comprehensive evaluation — 7 October 2026

[Plain-language overview, task winners and every measured result](https://laion-whisper-base-small-emotion-voice-burst.static.hf.space/benchmark.html).
All nine frozen audio backbones × linear/MLP heads × legacy/Flash target phases
(36 probe configurations), plus original and fully Flash-tuned Whisper Base/Small,
have completed training and evaluation. The report includes EmoNet-Voice,
VoiceNet-Emo, VoiceNet-Ext (emolia-dim), matched actor-disjoint CREMA-D/RAVDESS
MLPs, 192 scalar targets, CPS, Orange speaker cosines and timed/classified vocal bursts.
Whisper tuning improves Flash-target agreement but reduces P3 burst retention;
both domains and the limitations of comparison to paper results are explicit.
Exact aggregate metrics, configurations and fold/job receipts are downloadable
from the report. Human-label benchmarks are distinguished from teacher-target agreement.
''' + end
    card = re.sub(re.escape(begin) + '.*?' + re.escape(end), lambda _: section, original, flags=re.S) if begin in original else original.rstrip() + '\n\n' + section + '\n'
    folder = ROOT / 'consolidated_benchmark'
    card_path = folder / 'README.md'; card_path.write_text(card)
    sources = {
        'benchmark.html': report,
        'benchmark-results-2026-10-07.json.gz': BUNDLE,
        'benchmark-result-manifest.json': MANIFEST,
        'embedding-probe-code/benchmark_supplement.py': HERE / 'benchmark_supplement.py',
        'embedding-probe-code/publish_consolidated_benchmark.py': HERE / 'publish_consolidated_benchmark.py',
        'embedding-probe-code/evaluate_validation.py': HERE / 'evaluate_validation.py',
        'embedding-probe-code/validation.sbatch': HERE / 'validation.sbatch',
        'write_comprehensive_benchmark_html.py': HERE.parent / 'write_comprehensive_benchmark_html.py',
        'README.md': card_path,
    }
    for tree in ('probes', 'whisper'):
        for path in sorted((ROOT / tree).rglob('validation_metrics.json')):
            sources['embedding-probe-results/' + str(path.relative_to(ROOT))] = path
    operations = [CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(path)) for name, path in sources.items()]
    commit = api.create_commit(repo_id=repo, repo_type='space', operations=operations,
                              commit_message='Consolidate all completed probe and full Whisper tuning benchmark results',
                              parent_commit=info.sha)
    write(folder / 'publication.json', {
        'commit': commit.oid, 'commit_url': commit.commit_url,
        'report_url': 'https://laion-whisper-base-small-emotion-voice-burst.static.hf.space/benchmark.html',
        'artifacts': {name: {'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                      for name, path in sources.items()},
    })
    print(commit.commit_url)


if __name__ == '__main__':
    main()
