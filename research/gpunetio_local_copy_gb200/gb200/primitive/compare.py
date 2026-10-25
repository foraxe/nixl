import json
from pathlib import Path
from analyze import analyze

root = Path(__file__).resolve().parent / 'evidence'
paths = [root / f'confirm-seed-{seed}' for seed in (20261029, 20261030)]
assert (paths[0] / 'hashes.sha256').read_text() == (paths[1] / 'hashes.sha256').read_text()
runs = [analyze(p / 'results.jsonl') for p in paths]
assert [r['config']['seed'] for r in runs] == [20261029, 20261030]
named = []
for r in runs:
    selected = [x for x in r['comparisons'] if x['path'] == 4 and x['bytes'] == 512 and x['fragments'] == 32 and x['work'] in (0, 1024)]
    assert len(selected) == 2
    for x in selected:
        assert x['versus_legacy_e2e_ms']['favorable_blocks'] == x['versus_legacy_wall_ms']['favorable_blocks'] == 7
        if x['work']:
            assert x['async_versus_serial_e2e_ms']['favorable_blocks'] == x['async_versus_serial_wall_ms']['favorable_blocks'] == 7
    named.append({'seed': r['config']['seed'], 'comparisons': selected})
print(json.dumps({'status': 'PASS', 'matching_source_binary_hashes': True,
                  'measured_kernels': sum(r['samples'] for r in runs), 'validated_warmup_kernels': 2*450*7,
                  'scope': 'single-CTA in-kernel registered host-to-HBM copy, independent integer work, wait and full-payload consume; not model throughput',
                  'named_case': named}, indent=2))
