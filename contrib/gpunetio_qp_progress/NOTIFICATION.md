# Notification progress replay

Manual companion to `qp_progress_replay.cu`; not added to production PRs or CI.
See README.md for matching release libraries, three distinct GPUs, numeric
control IPv4, fresh shared metadata directory and four distinct ports.

Build against the candidate API V2 installation:

```bash
nvcc -std=c++20 -O3 -arch=sm_90 --cudart shared \
  -I"$PREFIX/include" notification_send_replay.cu -L"$NIXL_LIBDIR" \
  -lnixl -lnixl_build -lgtest -Xcompiler=-pthread -o notification_send_replay
export QP_WARMUP=200 QP_MEASURED=1000 SEND_PRESSURE=0
export CUDA_MODULE_LOADING=LAZY
unset LD_PRELOAD CUDA_DEVICE_MAX_CONNECTIONS
```

For each synchronous/candidate plugin arm, set `NIXL_PLUGIN_DIR` while keeping
the same core, executable, loaded libraries, QPs, workload and CPU/GPU affinity.
Use the paired environment documented in README.md and a fresh coordination
directory for every run; run target and source on distinct GPUs:

```bash
# Target terminal: expose two target GPUs; source terminal exposes another GPU.
NIXL_QP_PROGRESS_ROLE=target ./notification_send_replay \
  --gtest_filter=QpProgress.NotificationSendIsolation
NIXL_QP_PROGRESS_ROLE=source ./notification_send_replay \
  --gtest_filter=QpProgress.NotificationSendIsolation
```

Use five fresh pairs per arm with alternating arm order; pin the two workers
to the same distinct physical CPU cores across arms. Keep all JSON/log files.
Both endpoints must pass, not skip. The source writes its JSON to the shared
coordinate directory. Report A and B p50/p99, paired changes and win counts.

A carries a4KiB WRITE plus approximately7.9KiB notification; B carries an
independent4KiB WRITE. Payload epochs and notification content/count are checked
outside the measured API-completion window. Normal measurements use no receiver
pause or injected delay. `SEND_PRESSURE=1` is an excluded exploratory mode, not
part of the historical result or a validated correctness configuration.

The historical synchronous control2efba22c already had the notification state
machine and blocked only in SEND CQ polling; candidatecbf41ec0 differed only in
that poll loop. The new data-only5b1ab80f versus notification29b3190d comparison
changes the whole notification implementation, not only the final poll loop.
Re-measure it; do not label historical figures as the new branch's result.
