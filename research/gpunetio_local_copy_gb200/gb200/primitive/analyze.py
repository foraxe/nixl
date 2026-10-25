"""One-kernel primitive timing and phase diagnostics, without sample trimming."""
import argparse
from collections import defaultdict
import json
import gzip
from pathlib import Path
import statistics as st

NAMES = ['scalar_legacy', 'scalar_batch_wait', 'warp_legacy', 'warp_batch_wait',
         'warp_parallel_wait', 'SM_control', 'TMA_HBM_control']


def analyze(path):
    if not path.exists() and path.with_suffix(path.suffix + '.gz').exists():
        path = path.with_suffix(path.suffix + '.gz')
    if path.suffix == '.gz':
        with gzip.open(path, 'rt') as source:
            rows = [json.loads(s) for s in source]
    else:
        rows = [json.loads(s) for s in path.read_text().splitlines()]
    assert rows[-1] == {'kind': 'complete', 'status': 'PASS'}
    cfg = next(r for r in rows if r['kind'] == 'config')
    for name in ('thin_example', 'contracts', 'negative', 'intermediate_cqe_error'):
        assert next(r for r in rows if r['kind'] == name)['status'] == 'PASS'
    groups = defaultdict(lambda: defaultdict(list))
    keys = ('path', 'serialized', 'bytes', 'fragments', 'work')
    for r in rows:
        if r['kind'] != 'result':
            continue
        assert r['mismatch'] == 0
        if r['path'] < 5:
            assert r['issued'] == r['retired']
        groups[tuple(r[k] for k in keys)][r['block']].append(r)
    assert len(groups) == cfg['cases']
    if cfg['phase'] == 'confirm':
        assert (cfg['blocks'], cfg['reps'], cfg['rounds'], cfg['cases']) == (7, 3, 128, 450)
        assert cfg['profile'] is False
        assert next(r for r in rows if r['kind'] == 'warp_example')['status'] == 'PASS'
        expected = {(kind, serial, size, fragments, work)
                    for size in (64, 512, 4096) for fragments in (1, 8, 31, 32)
                    for work in (0, 128, 1024) for kind in range(7) for serial in (0, 1)
                    if not (kind == 5 and serial) and not (kind == 6 and size * fragments > 16384)}
        assert set(groups) == expected
        assert all(r['submit_cycles'] == r['compute_cycles'] == r['wait_cycles'] == r['consume_cycles'] == 0 for r in rows if r['kind'] == 'result')
    results = []
    for key, blocks in sorted(groups.items()):
        assert sorted(blocks) == list(range(cfg['blocks']))
        assert all(sorted(r['trial'] for r in sample) == list(range(cfg['reps'])) for sample in blocks.values())
        x = dict(zip(keys, key)); x['name'] = NAMES[x['path']]
        for field in ('e2e_ms', 'wall_ms', 'submit_cycles', 'compute_cycles', 'wait_cycles', 'consume_cycles'):
            vals = [st.mean(r[field] for r in blocks[b]) for b in range(cfg['blocks'])]
            x[field] = {'median': st.median(vals), 'block_means': vals}
        x['us_per_round'] = x['e2e_ms']['median'] * 1000 / cfg['rounds']
        results.append(x)
    comparisons = []
    if not cfg['profile']:
        for x in results:
            if x['path'] not in (1, 3, 4) or x['serialized']:
                continue
            basepath = 0 if x['path'] == 1 else 2
            base = next(y for y in results if y['path'] == basepath and all(y[k] == x[k] for k in keys[1:]))
            serial = next(y for y in results if y['path'] == x['path'] and y['serialized'] == 1 and all(y[k] == x[k] for k in keys[2:]))
            item = {k: x[k] for k in keys}
            for label, other in [('versus_legacy', base), ('async_versus_serial', serial)]:
                for field in ('e2e_ms', 'wall_ms'):
                    changes = [100 * (1 - a / b) for a, b in zip(x[field]['block_means'], other[field]['block_means'])]
                    item[f'{label}_{field}'] = {'median_reduction_percent': st.median(changes),
                                                 'favorable_blocks': sum(v > 0 for v in changes), 'paired_reductions': changes}
            comparisons.append(item)
    return {'config': cfg, 'samples': sum(len(s) for b in groups.values() for s in b.values()),
            'results': results, 'comparisons': comparisons}


if __name__ == '__main__':
    p = argparse.ArgumentParser();p.add_argument('path', type=Path)
    args = p.parse_args();print(json.dumps(analyze(args.path), indent=2))
