"""Full-payload in-kernel validation using real PyTorch allocations."""
import argparse
import ctypes as C
import json
import time
import torch
from local_copy import LocalCopy


def run(args):
    torch.cuda.set_device(args.gpu)
    cases = [(1, 1, 1, 3), (4096, 1, 1, 3), (1 << 20, 1, 1, 3),
             (64, 37, 8, 5), (4096, 37, 16, 5), (64, 32, 32, 2200)]
    if args.owner_only:
        cases = []
    rows = []
    for epoch, (size, count, window, rounds) in enumerate(cases, 1):
        src = torch.empty(256 + count * (size + 17), dtype=torch.uint8, pin_memory=True)
        dst = torch.empty(256 + count * (size + 32), dtype=torch.uint8, device=f"cuda:{args.gpu}")
        values = ((torch.arange(src.numel(), dtype=torch.int64) * 17 + epoch * 31) % 251).byte()
        src.copy_(values)
        with LocalCopy(src, dst, nic=args.nic, gid=args.gid, library=args.library) as copy:
            probe = copy.lib.lc_probe
            probe.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p, C.c_size_t,
                              C.c_int, C.c_int, C.c_int, C.c_int, C.c_int,
                              C.c_void_p, C.c_void_p]
            result = torch.zeros(6, dtype=torch.int64, device=dst.device)
            start = time.perf_counter()
            assert probe(copy.device_handle, src.data_ptr(), dst.data_ptr(), size, count,
                         window, rounds, epoch, 0, result.data_ptr(),
                         torch.cuda.current_stream().cuda_stream) == 0
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            evidence = result.cpu().tolist()
            assert evidence[0] == 0 and evidence[1] == 0, evidence
            assert evidence[2] == evidence[3] and evidence[2] >= count * rounds, evidence
            # Repeat on the same registrations after changing the host tensor.
            next_epoch = epoch + 100
            src.copy_(((torch.arange(src.numel(), dtype=torch.int64) * 17 + next_epoch * 31) % 251).byte())
            result.zero_()
            assert probe(copy.device_handle, src.data_ptr(), dst.data_ptr(), size, count,
                         window, 1, next_epoch, 0, result.data_ptr(),
                         torch.cuda.current_stream().cuda_stream) == 0
            torch.cuda.synchronize()
            assert result.cpu().tolist()[:2] == [0, 0]
            for mode in (1, 2, 3, 4):
                result.zero_()
                assert probe(copy.device_handle, src.data_ptr(), dst.data_ptr(), size, count,
                             window, 1, epoch, mode, result.data_ptr(),
                             torch.cuda.current_stream().cuda_stream) == 0
                torch.cuda.synchronize()
                assert result[0].item() == 0, (mode, result.cpu().tolist())
            rows.append(dict(size=size, fragments=count, window=window, rounds=rounds,
                             result=evidence, seconds=elapsed))
    qualifications = []
    for mode in ((40,) if args.owner_only else (5, 6, 7, 10, 30, 40, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59)):
        src = torch.empty(1 << 20, dtype=torch.uint8, pin_memory=True)
        src.copy_(((torch.arange(src.numel(), dtype=torch.int64) * 17 + 31) % 251).byte())
        dst = torch.empty_like(src, device=f"cuda:{args.gpu}")
        result = torch.zeros(16, dtype=torch.int64, device=dst.device)
        extra = []
        with LocalCopy(src, dst, nic=args.nic, gid=args.gid, library=args.library) as copy:
            probe = copy.lib.lc_probe
            probe.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p, C.c_size_t,
                              C.c_int, C.c_int, C.c_int, C.c_int, C.c_int,
                              C.c_void_p, C.c_void_p]
            if mode == 40:
                owner_probe = copy.lib.lc_owner_probe
                owner_probe.argtypes = [C.c_void_p] * 4
                torch.cuda.synchronize()
                assert owner_probe(copy.device_handle, src.data_ptr(), dst.data_ptr(), result.data_ptr()) == 0
                assert result.cpu().tolist()[:5] == [1, 1, 0, 0, -16], result.cpu().tolist()
            else:
                assert probe(copy.device_handle, src.data_ptr(), dst.data_ptr(), 64, 37, 8,
                             3, 1, mode, result.data_ptr(),
                             torch.cuda.current_stream().cuda_stream) == 0
            # mode30 deliberately closes after issue, without a device wait or host sync.
            if mode not in (30, 40):
                torch.cuda.synchronize()
                values = result.cpu().tolist()
                assert values[0] == 0, (mode, values)
                if mode == 52:
                    assert values[1] == -5 and values[9:11] == [-5, 1], values
                elif mode == 10:
                    assert values[1:3] == [-5, 1], values
                else:
                    assert values[1] == 0, (mode, values)
            if mode == 50:
                for epoch, (size, count, window, rounds) in enumerate(
                        [(64, 32, 32, 2200), (4096, 37, 32, 3), (4096, 37, 16, 3)], 101):
                    src.copy_(((torch.arange(src.numel(), dtype=torch.int64) * 17 + epoch * 31) % 251).byte())
                    result.zero_()
                    assert probe(copy.device_handle, src.data_ptr(), dst.data_ptr(), size, count, window,
                                 rounds, epoch, mode, result.data_ptr(), torch.cuda.current_stream().cuda_stream) == 0
                    torch.cuda.synchronize()
                    values = result.cpu().tolist()
                    assert values[:2] == [0, 0] and values[2] == values[3], values
                    extra.append(dict(size=size, fragments=count, window=window, rounds=rounds, result=values))
        qualifications.append(dict(mode=mode, result=result.cpu().tolist(), extra=extra))
    print(json.dumps(dict(status="PASS", gpu=args.gpu, torch=torch.__version__,
                          cases=rows, qualifications=qualifications), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=7)
    parser.add_argument("--nic", default="mlx5_bond_0")
    parser.add_argument("--gid", type=int, default=3)
    parser.add_argument("--library")
    parser.add_argument("--owner-only", action="store_true")
    run(parser.parse_args())
