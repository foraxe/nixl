# Experimental quiesced peer retirement

This research branch extends the [legacy DOCA 3.1 GET reproducer](../vllm_engram_repro/README.md) with explicit owner-side peer retirement. It is not NIXL main, a supported API, or a production fix. The caller must first stop admission, drain GPU/NIC work, and destroy the remote reader; the owner must not retire an unknown or active peer. A crashed reader remains an uncertain lease.

Build the targets and ABI-3 shim using the linked reproducer instructions. The additional shim includes NIXL's internal headers; add the Abseil include directory to `CPATH` when invoking `build_against_m1.py`, for example:

```bash
export CPATH="$WORK/absl/include:${CPATH:-}"
python3 research/engram_shared_rdma_m2/build_against_m1.py --build "$WORK/build"
```

The `engram_retire_idle_peer` shim is used by the opt-in SGLang research owner. Retrying reported QP/CQ/UAR/UMEM teardown failures retains the peer and its lease until successful. Failures not reported by the DOCA SDK cannot be detected here. Peer setup and retirement are serialized, but this does not provide automatic safe crash recovery or authentication.

A previous build of the same native source passed synthetic 20-cycle sequential, overlapping-reader, 16-lane graph and cross-node tests, six injected teardown-error tests, and a real SGLang 1P1D controlled drain. Those results predate this separately published packaging commit and used DOCA 3.1; see the application branch's evidence before making performance or current-SDK claims.
