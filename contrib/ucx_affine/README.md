# UCX affine WRITE reproducer

This opt-in replay compares ordinary UCX SGL WRITE with
`ucx_affine_write_v1` on the same dynamic, fragmented GPU workload. It measures
the sender from address-plan construction through request DONE. The receiver
checks the payload, untouched gaps, ready epochs, and notifications on the GPU.
The replay is a component test; its numbers are not serving TTFT or TPOT.

The fixed maximum shape is 27 layers × 64 rows × 576 bytes × four logical
peers. `PD_REPLAY_LAYERS` and `PD_REPLAY_ROWS` can reduce the first two
dimensions on both endpoints. Each dispatch changes source and destination
windows and recreates request handles. The sender uses 1,729 ordinary
descriptors per peer or 27 affine runs plus one ordinary marker.

Build the same replay on both endpoints against the same NIXL core and a UCX
build that supports UCP SGL WRITE. Adjust the three paths for each endpoint:

```bash
UCX_AFFINE_BUILD=/tmp/ucx-affine-replay \
  bash contrib/ucx_affine/build.sh "$NIXL_INCLUDE" "$NIXL_LIB" "$CUDA_ROOT"
```

Copy `config.example.json` to a private configuration, set the endpoint
commands, GPUs, paths, network device, and ports, then run from a machine that
can execute both `shell_prefix` commands:

```bash
python3 contrib/ucx_affine/run.py /path/to/config.private.json /path/to/new-results
```

The baseline and candidate plugins must be built against the **same** NIXL
core. `baseline_plugin_dir` points to the ordinary UCX plugin from the pinned
base; `affine_plugin_dir` points to this change. The receiver uses the
baseline plugin in both arms. The runner alternates fresh processes, checks
GPU availability, records binary hashes and mapped libraries, validates every
dispatch, and writes phase samples plus `summary.json`. Keep the private
configuration and endpoint identities out of public results.

The candidate sends an option such as
`ucx_affine_write_v1=27,64,576,2304,576`: 27 runs, 64 rows per run, 576-byte
rows, 2304-byte source stride, and 576-byte destination stride. The last
marker descriptor is transferred normally. `NIXL_UCX_SGL_ENABLE=1` is set by
the runner. To test separate layer registrations, odd row counts, repeated
posts, or standalone notifications, set `PD_REPLAY_LAYER_MRS=1`,
`PD_REPLAY_ROWS=63`, `PD_REPLAY_REPOSTS=5`, or `PD_REPLAY_STANDALONE=1` in both
endpoint `env` objects. Use a fresh result directory for each configuration.
