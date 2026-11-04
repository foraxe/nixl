"""Exercise SGLang CP early/deferred prefetch with independent row references."""
import argparse
import base64
import importlib.util
import json
import os
import sys
from pathlib import Path

import torch

p = argparse.ArgumentParser()
p.add_argument('--modules', required=True)
p.add_argument('--library', required=True)
p.add_argument('--model', required=True)
p.add_argument('--ready', required=True)
p.add_argument('--reference', required=True)
p.add_argument('--port', type=int, required=True)
p.add_argument('--out', type=Path, required=True)
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
os.environ.update(ENGRAM_SHARED_READY=a.ready, ENGRAM_SHARED_LIBRARY=a.library,
                  ENGRAM_SHARED_MODEL=a.model, ENGRAM_SHARED_READER_PORT_BASE=str(a.port),
                  ENGRAM_SHARED_MAX_QUERIES='4096', ENGRAM_SHARED_PREFETCH='1',
                  ENGRAM_SHARED_READ_PLAN='1', ENGRAM_SHARED_READ_LANES='16',
                  ENGRAM_SHARED_READ_GRAPH_SIZES='4096', ENGRAM_SHARED_DEFER_LAYER14='1')
reference = json.loads(Path(a.reference).read_text())
torch.cuda.set_device(0)
reader = adapter.ReaderRuntime()
reader.ranges = {layer: (0, reference[str(layer)]['total']) for layer in (1, 14)}
results = []
try:
    for iteration in range(2):
        # CP8, 253 tokens and 16 hash columns: three token rows are padded.
        full = torch.empty((253, 2, 16), dtype=torch.int64)
        for li, layer in enumerate((1, 14)):
            choices = reference[str(layer)]['rows']
            assert len(choices) == 4096 and choices[0] == 0 and len(set(choices)) == 4096
            rotated = choices[iteration:4048] + choices[:iteration]
            full[:, li, :] = torch.tensor(rotated, dtype=torch.int64).view(253, 16)
        ordered = torch.zeros((8, 32, 2, 16), dtype=torch.int64)
        for rank in range(8):
            part = full[rank::8]
            ordered[rank, :part.shape[0], :, :] = part
        reader.start_forward(True)
        reader.begin_prefetch(full.cuda(), local_rows=32, cp_size=8)
        assert reader.prefetch_issued == {1}, reader.prefetch_issued
        for li, layer in enumerate((1, 14)):
            if layer == 14:
                reader.issue_prefetch_layer(14)
                assert reader.prefetch_issued == {1, 14}, reader.prefetch_issued
            spec = reference[str(layer)]
            ids_cpu = ordered[:, :, li, :].reshape(-1)
            indices = {value: i for i, value in enumerate(spec['rows'])}
            order = torch.tensor([indices[int(v)] for v in ids_cpu], dtype=torch.int64)
            ref = torch.tensor([list(base64.b64decode(v)) for v in spec['bytes']], dtype=torch.uint8)
            expected_raw = ref[order]
            ids = ids_cpu.cuda()
            _, weights, scales = reader.fetch(ids, layer, 0, spec['total'])
            reps = reader.context.representatives(li)
            assert reps is not None
            output = torch.empty((4096, 256), dtype=torch.bfloat16, device='cuda')
            kernels.gather_row264(weights.data_ptr(), ids, output, 0, spec['total'],
                                  query_order=True, representatives=reps)
            reader.record_consumed(layer)
            torch.cuda.synchronize()
            packed = torch.cat((weights.view(torch.uint8)[reps], scales[reps]), dim=1)
            assert torch.equal(packed.cpu(), expected_raw)
            expected = (expected_raw[:, :256].contiguous().view(torch.float8_e4m3fn).float() *
                        torch.exp2(expected_raw[:, 256:].float() - 127).repeat_interleave(32, 1)).to(torch.bfloat16)
            assert torch.equal(output.cpu().view(torch.int16), expected.view(torch.int16))
            stats = reader.layer_stats[li].cpu().tolist()
            lanes = reader.context.stats[:, 4].cpu().tolist()
            assert stats[0] == 0 and stats[4] == 4048, stats
            assert len(lanes) == 16 and all(x > 0 for x in lanes) and sum(lanes) == 4048, lanes
            results.append({'iteration': iteration, 'layer': layer, 'queries': 4096,
                            'unique_reads': int(stats[4]), 'lane_reads': lanes})
    captures, replays = reader.context.graph_captures, reader.context.graph_replays
    assert captures == 2 and replays >= 4, (captures, replays)
finally:
    reader.close()
a.out.write_text(json.dumps({'validation': 'PASS', 'cp_size': 8, 'input_tokens': 253,
                             'padded_tokens': 3, 'deferred_layer14': True,
                             'checkpoint': reference['checkpoint'], 'results': results,
                             'captures': captures, 'replays': replays,
                             'lease_released': True}, indent=2) + '\n')
print('PASS: CP8 early/deferred prefetch, padding, 16 lanes, graph and real rows', flush=True)
