# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU interval/ownership check only; not CUDA lifecycle qualification of arbitrary shapes."""
counts = (
    1,
    2,
    3,
    31,
    32,
    33,
    63,
    64,
    127,
    128,
    383,
    384,
    511,
    512,
    513,
    1023,
    1024,
    1025,
    2047,
    2048,
    2049,
    4095,
    4096,
)
checked = 0
for count in counts:
    for contexts in range(1, 65):
        for requested_ctas in range(1, contexts + 1):
            active = min(count, contexts)
            chunk = (count - 1) // active + 1
            active = (count - 1) // chunk + 1
            grid = min(active, requested_ctas)
            group_size = (active + grid - 1) // grid
            kernel_chunk = (count + active - 1) // active
            assert chunk == kernel_chunk
            owners = []
            intervals = []
            for block in range(grid):
                base = block * group_size
                if base >= active:
                    continue  # Uniform empty-CTA guard before any barriers.
                lanes = min(group_size, active - base)
                assert 0 < lanes <= 64
                for local in range(lanes):
                    q = base + local
                    owners.append(q)
                    start, end = q * chunk, min(count, (q + 1) * chunk)
                    assert 0 <= start < end <= count
                    intervals.append((start, end))
                    window = min(512, chunk)
                    total = 0
                    for offset in range(0, chunk, window):
                        ops = min(
                            window, chunk - offset, max(0, count - start - offset)
                        )
                        assert 0 <= ops < 2048
                        total += ops
                    assert total == end - start
            assert sorted(owners) == list(range(active))
            intervals.sort()
            assert intervals[0][0] == 0 and intervals[-1][1] == count
            assert all(a[1] == b[0] for a, b in zip(intervals, intervals[1:]))
            checked += 1
print(f"PASS {checked} CPU interval/unique-QP-owner configurations")
