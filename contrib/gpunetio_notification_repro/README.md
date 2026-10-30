# Standalone GPUNETIO notification comparison

This opt-in component tool runs one sender and one continuously polling receiver
on two distinct GPUs. It validates all 4,096 distinct messages and measures
message construction plus the host `genNotif()` call for 3,968 messages after
128 warmups. It reports medians
and paired changes over five alternating fresh-process pairs. No RNR, receive
pause, CQ resizing or MPS is injected. It is not a serving benchmark.

Build a release NIXL core and both GPUNETIO plugins with the same SDK and
compiler. Run both plugins against that same core and the same CUDA/DOCA
runtime. Then build and run, replacing the placeholders for your environment:

```bash
export PREFIX=/path/to/common-nixl-install
export NIXL_LIBDIR="$PREFIX/lib/x86_64-linux-gnu"
export CUDA_HOME=/path/to/cuda
export BASELINE_PLUGIN_DIR=/path/to/main/plugin
export CANDIDATE_PLUGIN_DIR=/path/to/candidate/plugin
export NIC=your_rdma_device OOB_INTERFACE=your_network_interface GID_INDEX=0
export SOURCE_GPU=0 TARGET_GPU=1
# Add the matching DOCA/CUDA runtime libraries to LD_LIBRARY_PATH first.
export LD_LIBRARY_PATH="$NIXL_LIBDIR:$CUDA_HOME/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
c++ -std=c++20 -O2 -I"$PREFIX/include" -I"$CUDA_HOME/include" \
  ordinary_notif.cpp -L"$NIXL_LIBDIR" -L"$CUDA_HOME/targets/x86_64-linux/lib" \
  -lnixl -lnixl_build -lcudart -ldl -pthread -o ordinary-notif
bash run_ordinary.sh
```

Set `IBV_DRIVER_PATH` if required. Optionally set `CPUSET` to two suitable Linux
CPU cores and `RESULT_DIR` to a fresh directory. The runner checks both endpoint
message identity/counts, process exits and binary hashes, and writes per-run logs
and `summary.json`. Keep GPU placement, SDK and external workload matched.
