#!/usr/bin/env python3
"""Generate public synthetic 6144-row queries with exactly 1561 owned by rank 1."""
import argparse
import hashlib
import random
import numpy as np

p = argparse.ArgumentParser()
p.add_argument('--out', required=True)
a = p.parse_args()
rng = random.Random(20261104)
owned = [rng.randrange(65536, 131072) for _ in range(1561)]
other = [v if v < 65536 else v + 65536 for v in (rng.randrange(196608) for _ in range(4583))]
rows = owned + other
rng.shuffle(rows)
rows = np.asarray(rows, dtype='<i8')
assert len(rows) == 6144 and int(((rows >= 65536) & (rows < 131072)).sum()) == 1561
np.savez(a.out, rows=rows)
print('rows=6144 owned=1561 raw_sha256=' + hashlib.sha256(rows.tobytes()).hexdigest())
