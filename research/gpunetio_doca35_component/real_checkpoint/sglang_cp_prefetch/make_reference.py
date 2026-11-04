"""Create a private, small independent checkpoint-row reference for the CP probe."""
import argparse
import base64
import importlib.util
import json
import os
import sys
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--modules', type=Path, required=True)
p.add_argument('--model', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
sys.path.insert(0, str(a.modules))
spec = importlib.util.spec_from_file_location('real_checkpoint_owner', a.modules / 'real_checkpoint_owner.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
manifest = module.inspect_checkpoint(a.model)
result = {'checkpoint': manifest['checkpoint_identity']}
for layer, total in zip((1, 14), manifest['rows']):
    rows = [0] + [(i * 104729) % total for i in range(1, 4096)]
    if len(set(rows)) != 4096:
        raise ValueError('checkpoint row count does not support the reference sequence')
    parts = []
    for kind in ('weight', 'scale'):
        tensor = next(x for x in manifest['specs'] if x['layer'] == layer and x['kind'] == kind)
        shard = manifest['shards'][tensor['shard']]
        fd = os.open(shard['path'], os.O_RDONLY)
        try:
            values = [os.pread(fd, tensor['columns'], shard['data_start'] +
                               tensor['data_offsets'][0] + row * tensor['columns']) for row in rows]
        finally:
            os.close(fd)
        if any(len(value) != tensor['columns'] for value in values):
            raise ValueError('short checkpoint read')
        parts.append(values)
    result[str(layer)] = {'total': total, 'rows': rows,
                          'bytes': [base64.b64encode(w + s).decode() for w, s in zip(*parts)]}
fd = os.open(a.out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(fd, 'w') as stream:
    json.dump(result, stream)
print('PASS: private row reference created')
