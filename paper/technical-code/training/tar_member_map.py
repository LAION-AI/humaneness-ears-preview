#!/usr/bin/env python3
"""Packed, read-only map from TAR path/member name to byte offset and size.

The corpus stays in the existing large TAR containers.  A single compressed
record per TAR replaces repeated full TAR-directory scans on every DDP rank.
"""
from collections import OrderedDict
import json
import os
from pathlib import Path
import threading
import zlib


class TarMemberMap:
    def __init__(self, manifest, capacity=256):
        self.manifest_path = Path(manifest)
        spec = json.loads(self.manifest_path.read_text())
        assert spec['status'] == 'complete' and spec['format'] == 'zlib-json-tar-member-map-v1'
        root = self.manifest_path.parent
        index_path = root / spec['index']
        data_path = root / spec['data']
        assert index_path.stat().st_size == spec['index_bytes']
        assert data_path.stat().st_size == spec['data_bytes']
        index = json.loads(index_path.read_text())
        assert len(index['paths']) == spec['tars']
        assert all(len(index[k]) == spec['tars'] for k in ('offsets', 'sizes', 'members', 'tar_sizes'))
        self.rows = {path: (int(index['offsets'][i]), int(index['sizes'][i]),
                            int(index['members'][i]), int(index['tar_sizes'][i]))
                     for i, path in enumerate(index['paths'])}
        self.fd = os.open(data_path, os.O_RDONLY)
        self.capacity = int(capacity)
        self.local = threading.local()

    def _cache(self):
        if not hasattr(self.local, 'members'):
            self.local.members = OrderedDict()
        return self.local.members

    def locate(self, path, member):
        cache = self._cache()
        if path not in cache:
            assert path in self.rows, f'TAR absent from packed member map: {path}'
            offset, size, expected_members, expected_tar_size = self.rows[path]
            assert os.path.getsize(path) == expected_tar_size, f'TAR size changed after index build: {path}'
            compressed = os.pread(self.fd, size, offset)
            assert len(compressed) == size
            decoded = json.loads(zlib.decompress(compressed))
            assert len(decoded) == expected_members
            cache[path] = decoded
            while len(cache) > self.capacity:
                cache.popitem(last=False)
        cache.move_to_end(path)
        value = cache[path].get(member)
        assert value is not None, f'Member absent from packed map: {path}::{member}'
        return int(value[0]), int(value[1])

