"""Compare actual vLLM provider lookups using independently read checkpoint rows."""
import argparse
import hashlib
import json
import os
import statistics
from pathlib import Path
from types import SimpleNamespace

import torch
from vllm.distributed.engram_share.rdma.owner import inspect_checkpoint
from vllm.distributed.engram_share.rdma.provider import VllmRemoteReader

p = argparse.ArgumentParser()
p.add_argument('--model', type=Path, required=True)
p.add_argument('--ready', type=Path, required=True)
p.add_argument('--library', type=Path, required=True)
p.add_argument('--port', type=int, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
os.environ.update(ENGRAM_SHARED_MODEL=str(a.model), ENGRAM_SHARED_READY=str(a.ready),
                  ENGRAM_SHARED_LIBRARY=str(a.library), ENGRAM_SHARED_READER_PORT_BASE=str(a.port),
                  ENGRAM_SHARED_MAX_QUERIES='6144')
torch.cuda.set_device(0)
torch.set_num_threads(2)
m = inspect_checkpoint(a.model)
reader = VllmRemoteReader()
results = []
try:
    for layer, count_rows in zip((1, 14), m['rows']):
        specs = [next(s for s in m['specs'] if s['layer'] == layer and s['kind'] == kind)
                 for kind in ('weight', 'scale')]
        table = SimpleNamespace(dim=256, block_size=32, head_start=0, part_n_hash_cols=2,
                                rdma_layer_id=layer, vocab_start_idx=0,
                                vocab_end_idx=count_rows, num_embeddings=count_rows)
        for count in (384, 6144):
            ids_cpu = [(i * 104729 + 17) % count_rows for i in range(count)]
            assert len(set(ids_cpu)) == count
            parts = []
            for spec in specs:
                shard = m['shards'][spec['shard']]
                width = spec['columns']
                fd = os.open(shard['path'], os.O_RDONLY)
                try:
                    blob = b''.join(os.pread(fd, width, shard['data_start'] +
                                    spec['data_offsets'][0] + row * width) for row in ids_cpu)
                finally:
                    os.close(fd)
                assert len(blob) == count * width
                parts.append(torch.frombuffer(bytearray(blob), dtype=torch.uint8).view(count, width))
            expected = torch.cat(parts, dim=1)
            ids = torch.tensor(ids_cpu, dtype=torch.int32, device='cuda').view(-1, 2)
            out = torch.empty((count // 2, 2, 264), dtype=torch.uint8, device='cuda')
            def lookup():
                reader.lookup(table, ids, out)
            for _ in range(3): lookup()
            torch.cuda.synchronize()
            assert torch.equal(out.view(count, 264).cpu(), expected)
            samples = []
            for _ in range(7):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record(); lookup(); end.record(); end.synchronize()
                stats = reader.runtime.stats.cpu().tolist()
                assert stats[0] == 0 and stats[4] == count, stats
                samples.append(start.elapsed_time(end) * 1000)
            assert torch.equal(out.view(count, 264).cpu(), expected)
            results.append({'layer': layer, 'queries': count, 'unique_rows': count,
                            'gpu_event_us': samples, 'median_us': statistics.median(samples),
                            'validation': 'PASS'})
finally:
    reader.runtime.close()
a.output.write_text(json.dumps({'library_sha256': hashlib.sha256(a.library.read_bytes()).hexdigest(),
                                'checkpoint': m['checkpoint_identity'], 'warmups': 3,
                                'samples': 7, 'results': results}, indent=2) + '\n')
print('PASS: real checkpoint vLLM provider data, completions and timing', flush=True)
