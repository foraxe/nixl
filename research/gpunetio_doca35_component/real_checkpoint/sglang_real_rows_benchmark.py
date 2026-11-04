"""SGLang ReaderRuntime/ReadPlan real rows, dedup, graph and 16-lane benchmark."""
import argparse
import hashlib
import importlib.util
import json
import os
import statistics
import sys
from pathlib import Path

import torch

p = argparse.ArgumentParser()
p.add_argument('--modules', required=True)
p.add_argument('--library', type=Path, required=True)
p.add_argument('--model', type=Path, required=True)
p.add_argument('--ready', type=Path, required=True)
p.add_argument('--port', type=int, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
sys.path.insert(0, a.modules)
def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(a.modules) / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
load('engram_transport', 'sglang_runtime.py')
load('owner_control', 'owner_control.py')
kernels = load('row264_kernels', 'row264_kernels.py')
adapter = load('shared_model_adapter', 'shared_model_adapter.py')
checkpoint = load('real_checkpoint_owner', 'real_checkpoint_owner.py')
os.environ.update(ENGRAM_SHARED_READY=str(a.ready), ENGRAM_SHARED_LIBRARY=str(a.library),
                  ENGRAM_SHARED_MODEL=str(a.model), ENGRAM_SHARED_READER_PORT_BASE=str(a.port),
                  ENGRAM_SHARED_MAX_QUERIES='4096', ENGRAM_SHARED_PREFETCH='1',
                  ENGRAM_SHARED_READ_PLAN='1', ENGRAM_SHARED_READ_LANES='16',
                  ENGRAM_SHARED_READ_GRAPH_SIZES='4096')
torch.cuda.set_device(0)
torch.set_num_threads(2)
manifest = checkpoint.inspect_checkpoint(a.model)
reader = adapter.ReaderRuntime()
results = []
try:
    for li, (layer, total) in enumerate(zip((1, 14), manifest['rows'])):
        for unique in (4096, 2048):
            choices = [(i * 104729 + 17) % total for i in range(unique)]
            assert len(set(choices)) == unique
            ids_cpu = [choices[i % unique] for i in range(4096)]
            pieces = []
            for kind in ('weight', 'scale'):
                spec = next(s for s in manifest['specs'] if s['layer'] == layer and s['kind'] == kind)
                shard = manifest['shards'][spec['shard']]
                width = spec['columns']
                fd = os.open(shard['path'], os.O_RDONLY)
                try:
                    data = b''.join(os.pread(fd, width, shard['data_start'] +
                                    spec['data_offsets'][0] + r * width) for r in choices)
                finally:
                    os.close(fd)
                assert len(data) == unique * width
                pieces.append(torch.frombuffer(bytearray(data), dtype=torch.uint8).view(unique, width))
            ref = torch.cat(pieces, dim=1)
            order = torch.arange(4096) % unique
            expected_raw = ref[order]
            expected = (expected_raw[:, :256].contiguous().view(torch.float8_e4m3fn).float() *
                        torch.exp2(expected_raw[:, 256:].float() - 127).repeat_interleave(32, 1)).to(torch.bfloat16)
            ids = torch.tensor(ids_cpu, dtype=torch.int64, device='cuda')
            output = torch.empty((4096, 256), dtype=torch.bfloat16, device='cuda')
            def lookup():
                _, weights, scales = reader.fetch(ids, layer, 0, total)
                reps = reader.context.representatives(li)
                assert reps is not None
                kernels.gather_row264(weights.data_ptr(), ids, output, 0, total,
                                      query_order=True, representatives=reps)
                reader.record_consumed(layer)
                return weights, scales, reps
            for _ in range(3): weights, scales, reps = lookup()
            torch.cuda.synchronize()
            assert torch.equal(output.cpu().view(torch.int16), expected.view(torch.int16))
            assert torch.equal(torch.cat((weights.view(torch.uint8)[reps], scales[reps]), 1).cpu(), expected_raw)
            samples = []
            for i in range(15):
                for queued in ((False, True) if i % 2 == 0 else (True, False)):
                    torch.cuda.synchronize()
                    if queued: torch.cuda._sleep(10000000)
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record(); lookup(); end.record(); pending = not start.query(); end.synchronize()
                    stats = reader.layer_stats[li].cpu().tolist()
                    lanes = reader.context.stats[:, 4].cpu().tolist()
                    assert stats[0] == 0 and stats[4] == unique and len(lanes) == 16, (stats, lanes)
                    assert sum(lanes) == unique and all(x > 0 for x in lanes), lanes
                    samples.append({'queued': queued, 'start_pending_after_submit': pending,
                                    'gpu_event_us': start.elapsed_time(end) * 1000})
            assert all(x['start_pending_after_submit'] for x in samples if x['queued']), 'sleep too short'
            assert torch.equal(output.cpu().view(torch.int16), expected.view(torch.int16))
            results.append({'layer': layer, 'queries': 4096, 'unique_rows': unique,
                            'samples': samples, 'validation': 'PASS'})
    assert reader.context.graph_captures == 2 and reader.context.graph_replays >= 4
    captures, replays = reader.context.graph_captures, reader.context.graph_replays
finally:
    reader.close()
a.output.write_text(json.dumps({'library_sha256': hashlib.sha256(a.library.read_bytes()).hexdigest(),
                                'checkpoint': manifest['checkpoint_identity'],
                                'graph_captures': captures, 'graph_replays': replays,
                                'results': results}, indent=2) + '\n')
print('PASS: SGLang actual ReaderRuntime, dedup, 16 lanes, graph, bytes and timings', flush=True)
