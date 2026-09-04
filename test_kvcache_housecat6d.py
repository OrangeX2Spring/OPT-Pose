# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Project-owned portions of this file are licensed under CC-BY-NC-SA-4.0.
# See LICENSE and NOTICE for details. Third-party notices remain applicable.

"""
Step 1a of the causal-readout line: a real KV cache for the aggregator's global
attention, and what it costs and saves.

Step 0.5 established that a training-free causal readout does not break the
geometry heads. It is not, however, faster: it recomputes the reference block on
every forward. This measures the version that does not.

WHAT IS CACHED. The 24 global-attention blocks, and nothing else. That is where
the only cost that grows with cache size lives -- frame attention is per-frame,
and the DPT heads decode per frame, so neither scales with N. Caching sonata's
reference features, the camera head's trunk (four caches per block, one per
refinement iteration) and skipping the reference frames' dense decode are the
next constant factors, and they are Step 1b.

Consequence, and the reason this script stops at the aggregator: with only the
aggregator cached, a query-only pass through the FULL model is not equivalent to
the readout. Paths 2-5 and 7 of the seven (camera-head trunk, pointmap centre,
cross-frame keypoint attention, z_obj pooling, sonata's collated batch) all still
need the reference frames present. So this compares aggregator outputs, not
predictions, and it is honest about that rather than reporting an end-to-end
number the implementation does not support yet.

THREE PASSES per object sequence, on one preloaded window:

  READOUT   aggregator over frames [0, n] with num_ref_frames = n.
            Step 0.5's arithmetic. Its query-frame outputs are the target.
  BUILD     aggregator over frames [0, n) with collect_kv. Produces the cache.
            This is the "mapping phase" a tracker runs once.
  QUERY     aggregator over the query frame ALONE with kv_cache. The thing under
            test: one frame of work plus attention against the stored block.

THE GATE, and why it is not bit-equality. Step 0 established that this model's
outputs are bit-reproducible only when the compared passes have the SAME shapes:
projections over a different token count tile the GEMM differently and leave
~1e-6, irreducibly. QUERY projects P tokens where READOUT projects (n+1)*P, so
FIDELITY cannot be zero, and demanding it would repeat the mistake that cost
Step 0 a session. What is gated instead:

  FIDELITY  QUERY vs READOUT's query frame. Must land at the GEMM floor -- the
            same 1e-6 order Step 0.5 measured for SELFCHECK -- not orders above.
            --tol_fidelity is the ceiling, on the RELATIVE difference.
  REPEAT    QUERY vs a second BUILD+QUERY on freshly built tensors. The floor of
            the cached path itself. The aggregator has no sonata in it, so this
            should be exactly 0.
Peak GPU memory is reported per cell, from torch.cuda.max_memory_allocated. It is
not the quantity jobstats reports -- it excludes the allocator's reserved slack
and the CUDA context -- but it is the right instrument HERE, because what is
being checked is whether the cache costs what its own numel says it costs. It did
not, for four runs: k and v came back as views into a fused qkv projection, so
the pair pinned twice what it accounted for.

  ZEROED    QUERY with a zeroed cache, against READOUT. A negative control: if
            this is not orders WORSE than FIDELITY, the cache is not being read
            and FIDELITY proves nothing. Step 0 shipped two controls that could
            not fail; this one can.

PRECISION (--dtype). fp32 is what the HouseCat6D path does today: no autocast
anywhere, and torch 2.x leaves TF32 matmul OFF by default, so on Ampere the fp32
runs are leaving throughput on the table. Three settings:

  fp32   as shipped.
  tf32   two backend flags. Ampere+ only, 10-bit mantissa, tensors stay fp32 --
         so it changes speed and not one byte of memory.
  bf16   autocast. Weights stay fp32 and the norms with them; what changes is the
         matmuls and, because the cache IS a pair of projections, the cache: 128.8
         MiB per reference frame instead of 257.6. That is the difference between
         a 70-frame cache fitting a 24 GB card and not. The checkpoint was trained
         in bf16 (housecat_default.yaml:250), so this is arguably the faithful
         setting and fp32 the deviation.

  bf16 needs sm_80+. On Turing (passau, the 12g partition) it runs but is
  emulated, and the timing means nothing.

Precision cost and cache fidelity are DIFFERENT comparisons and are kept apart,
because a control that differs from the test in two ways is not a control:

  PRECISION  readout at --dtype vs readout in fp32, on the query frame. What
             precision alone costs, with no cache involved.
  FIDELITY   cached vs readout AT THE SAME DTYPE. What the cache costs, within
             whatever precision it is running.

FIDELITY is gated against PRECISION, not against a chosen constant: the cache
passes when it moves the output no more than switching precision does. That is a
MEASURED floor, which is the same correction Step 0.5 had to make when LEAK was
being judged against zero instead of against a repeat run.

The first attempt did invent a constant, and it was wrong within the hour. tf32
was given fp32's 1e-4 on the reasoning that "tensors stay fp32, so the floor is
unchanged" -- but tf32 truncates mantissas inside the matmul, so its floor is
~1e-3, and the tf32 run failed 10 of 10 while REPEAT sat at exactly zero and
FIDELITY came in BELOW PRECISION on every sequence. The numbers were fine; the
threshold was fiction. --tol_fidelity survives as an absolute floor for fp32,
where there is no PRECISION to compare against.

WHAT IS MEASURED. Milliseconds per query frame for the cached path against
recomputing the whole readout, and the cache's size in bytes, both against
--num_ref.

--cache_only drops the readout, both controls and FIDELITY, leaving the build,
the query and their cost. It exists because the READOUT baseline is what runs out
of memory first: it holds 24 intermediates of [B, S, P, 2C], which is 2.4 GB at
n=8 and 8.9 GB at n=32 in fp32, on top of 5.4 GB of weights -- so the comparison
cannot reach the cache sizes worth measuring, while the cached path, which needs
the cache plus one frame of activations, can. That asymmetry is not an
inconvenience to work around; it is the argument for the cache stated in memory
rather than in time. Establish fidelity at a small n, then sweep with this. Note that the timing sweep may go far past the checkpoint's
img_nums: [2, 4] training window: n > 3 says nothing about accuracy, but a
kernel does not care what a checkpoint was trained on, and the cost curve is the
quantitative form of the redundancy objection -- dense per-patch K/V for
overlapping keyframes is exactly what a 70-frame cache cannot afford.

Usage (see tools/opt_pose_kvcache.sh):

    python test_kvcache_housecat6d.py \
        --data_root <HouseCat6D root> --checkpoint <abs_pose_housecat.pt> \
        --num_ref 3 --num_seqs 10 --out <results.json>
"""

import argparse
import contextlib
import json
import sys

import numpy as np
import torch

sys.path.append("opt/")
sys.path.append("utils/")

from test_abs_housecat6d import load_opt_model
from test_causal_housecat6d import build_common_conf, build_inputs, compare, select_query
from training.data.datasets.housecat import HouseCat6DPoseDataset


def set_precision(dtype: str):
    """TF32 is a global backend switch, not a context; bf16 is a context, not a switch."""
    allow_tf32 = dtype == "tf32"
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32


def amp_ctx(dtype: str):
    if dtype == "bf16":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def readout_fwd(agg, images, n, dtype="fp32"):
    """Step 0.5's pass: [0, n] with the reference block masked off from the query.

    Returns the GPU tensors. The timing calls use this directly, so no device
    transfer lands inside a measurement.
    """
    with torch.inference_mode(), amp_ctx(dtype):
        out, _ = agg(images=images, num_ref_frames=n)
        return out


def query_fwd(agg, images_query, cache, dtype="fp32"):
    """Tracking phase: one frame, attending to the stored block."""
    with torch.inference_mode(), amp_ctx(dtype):
        out, _ = agg(images=images_query, kv_cache=cache)
        return out


def build_cache(agg, images_ref, dtype="fp32"):
    """Mapping phase: a plain forward over the reference frames, keeping their k/v."""
    cache = []
    with torch.inference_mode(), amp_ctx(dtype):
        agg(images=images_ref, collect_kv=cache)
    return cache


def run_readout(agg, images, n, dtype="fp32"):
    """readout_fwd, query frame only, on the CPU for comparison."""
    with torch.inference_mode():
        return [t[:, n : n + 1].float().cpu() for t in readout_fwd(agg, images, n, dtype)]


def run_query(agg, images_query, cache, dtype="fp32"):
    """query_fwd on the CPU for comparison."""
    with torch.inference_mode():
        return [t.float().cpu() for t in query_fwd(agg, images_query, cache, dtype)]


def compare_lists(left, right):
    """Max absolute and relative difference over all aggregator blocks."""
    assert len(left) == len(right), f"{len(left)} vs {len(right)} blocks"
    worst_abs, worst_rel = 0.0, 0.0
    for a, b in zip(left, right):
        d, r = compare(a, b)
        worst_abs, worst_rel = max(worst_abs, d), max(worst_rel, r)
    return worst_abs, worst_rel


def cache_bytes(cache):
    return sum(k.numel() * k.element_size() + v.numel() * v.element_size() for k, v in cache)


def time_ms(fn, warmup, iters):
    """Median wall time of fn, in ms. Median, not mean: the first timed call after
    warmup still catches allocator growth, and one outlier should not set the number."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return float(np.median(times))


def main():
    parser = argparse.ArgumentParser(description="Causal readout Step 1a: the KV cache")
    parser.add_argument("--data_root", type=str, required=True, help="HouseCat6D root")
    parser.add_argument("--checkpoint", type=str, required=True, help="abs_pose_housecat.pt")
    parser.add_argument("--num_ref", type=int, default=3,
                        help="cache size in frames. Accuracy is only interpretable up to 3 "
                             "(img_nums: [2, 4]); larger values still give a valid fidelity "
                             "check and a valid cost measurement")
    parser.add_argument("--num_seqs", type=int, default=10, help="object sequences to evaluate")
    parser.add_argument("--start", type=int, default=0, help="first frame index of the window")
    parser.add_argument("--query_gap", type=int, default=1,
                        help="frames from the last reference frame to the query frame, as in "
                             "test_causal_housecat6d.py")
    parser.add_argument("--dtype", choices=("fp32", "tf32", "bf16"), default="fp32",
                        help="fp32 as shipped; tf32 flips the two Ampere backend flags "
                             "(torch 2.x defaults them off, tensors stay fp32); bf16 "
                             "autocasts, which also halves the cache. sm_80+ for both")
    parser.add_argument("--tol_fidelity", type=float, default=None,
                        help="absolute floor on the RELATIVE difference between the cached query "
                             "pass and the readout, used only in fp32. At any other dtype the gate "
                             "is FIDELITY <= PRECISION -- a measured floor rather than a chosen "
                             "one, since the arithmetic perturbation of the precision itself is "
                             "the right yardstick for an arithmetic perturbation of the cache")
    parser.add_argument("--zeroed_margin", type=float, default=100.0,
                        help="the zeroed-cache control must be at least this many times worse "
                             "than FIDELITY, or the cache is not being read")
    parser.add_argument("--seed", type=int, default=0, help="numpy seed, set before every pass")
    parser.add_argument("--cache_only", action="store_true",
                        help="skip the readout baseline, FIDELITY, REPEAT and ZEROED; measure "
                             "only what the cached path costs. For the large-n cost curve, "
                             "where the baseline no longer fits in memory")
    parser.add_argument("--warmup", type=int, default=2, help="untimed calls before timing")
    parser.add_argument("--iters", type=int, default=10, help="timed calls per measurement")
    parser.add_argument("--use_gt_intrinsics", action="store_true")
    parser.add_argument("--out", type=str, required=True, help="results json")
    args = parser.parse_args()

    assert args.query_gap >= 1, "--query_gap 1 is the adjacent case; 0 would query a reference frame"
    if args.tol_fidelity is None:
        # Only used where PRECISION does not exist, i.e. fp32 against itself.
        args.tol_fidelity = 1e-4
    set_precision(args.dtype)
    cap = torch.cuda.get_device_capability()
    assert args.dtype == "fp32" or cap >= (8, 0), (
        f"--dtype {args.dtype} needs sm_80+, this is sm_{cap[0]}{cap[1]}. On Turing bf16 is "
        "emulated and tf32 does not exist, so the timing would be meaningless"
    )
    n = args.num_ref
    q_idx = n - 1 + args.query_gap
    window = q_idx + 1
    model, device = load_opt_model(checkpoint_path=args.checkpoint)
    agg = model.aggregator

    common_conf = build_common_conf(img_size=518, patch_size=14)
    ds = HouseCat6DPoseDataset(
        common_conf, data_root=args.data_root, split="test",
        min_num_images=window, sample_num=1024,
    )
    assert len(ds.seq_names) > 0, f"no object sequences under {args.data_root}"

    names = sorted(ds.seq_names)[: args.num_seqs]
    print(f"{len(ds.seq_names)} sequences available, evaluating {len(names)}, "
          f"dtype = {args.dtype}, num_ref = {n}, query_gap = {args.query_gap}, "
          f"window = frames [{args.start}, {args.start + window}), query = frame {args.start + q_idx}")

    records = []
    failures = []
    for name in names:
        entries = ds.chunks[name]
        if len(entries) < args.start + window:
            print(f"[skip] {name}: {len(entries)} frames < {args.start + window}")
            continue

        ids = list(range(args.start, args.start + window))
        batch = ds.get_data(seq_name=name, ids=ids, aspect_ratio=1.0)
        if len(batch["images"]) != window:
            print(f"[skip] {name}: loader returned {len(batch['images'])} of {window} frames")
            continue

        inp = build_inputs(batch, window, device, args.use_gt_intrinsics)
        # Built exactly the way the readout builds it, so the readout and the cached
        # path are handed the same frames in the same layout -- Step 0's lesson about
        # controls that differ in two ways applies to this comparison too.
        readout_inp = select_query(inp, n, q_idx)
        images = readout_inp["images"]
        images_ref, images_query = images[:, :n], images[:, n : n + 1]

        np.random.seed(args.seed)
        torch.cuda.reset_peak_memory_stats()
        cache = build_cache(agg, images_ref, args.dtype)
        assert len(cache) == len(agg.global_blocks)
        n_bytes = cache_bytes(cache)
        peak_build = torch.cuda.max_memory_allocated()
        # What the cache's own numel claims, against what is actually resident once the
        # build's transients are gone. They agree only if nothing in the cache is a view
        # into a larger allocation.
        resident = torch.cuda.memory_allocated()

        fidelity = repeat = control = precision = None
        if not args.cache_only:
            query = run_query(agg, images_query, cache, args.dtype)

            np.random.seed(args.seed)
            readout = run_readout(agg, images, n, args.dtype)
            fidelity = compare_lists(query, readout)

            # What precision alone costs: the same readout, one in fp32. No cache is
            # involved, so this and FIDELITY move for different reasons and are never
            # summed. fp32 compares against itself and is trivially zero, so skip it.
            if args.dtype != "fp32":
                set_precision("fp32")
                np.random.seed(args.seed)
                readout_fp32 = run_readout(agg, images, n, "fp32")
                precision = compare_lists(readout, readout_fp32)
                del readout_fp32
                set_precision(args.dtype)
                torch.cuda.empty_cache()

            # A second, independently built cache: the floor of the cached path itself.
            # Freed before the control allocates a third one -- each cache is 258 MiB
            # per reference frame in fp32, and three of them plus the weights and a
            # readout pass are what put n=8 over a 16 GB card.
            cache2 = build_cache(agg, images_ref, args.dtype)
            query2 = run_query(agg, images_query, cache2, args.dtype)
            repeat = compare_lists(query, query2)
            del cache2, query2
            torch.cuda.empty_cache()

            with torch.inference_mode():
                zeroed = [(torch.zeros_like(k), torch.zeros_like(v)) for k, v in cache]
            query_zeroed = run_query(agg, images_query, zeroed, args.dtype)
            control = compare_lists(query_zeroed, readout)
            del zeroed, query_zeroed, query, readout
            torch.cuda.empty_cache()

        # Timed on the GPU tensors: a device transfer is not part of what a tracker
        # would pay per frame, and it is the same for both paths anyway.
        ms_cached = time_ms(lambda: query_fwd(agg, images_query, cache, args.dtype),
                            args.warmup, args.iters)
        ms_readout = None if args.cache_only else time_ms(
            lambda: readout_fwd(agg, images, n, args.dtype), args.warmup, args.iters)

        print(f"--- {name}")
        print(f"    cache {n_bytes / 2**20:.1f} MiB claimed   {resident / 2**20:.1f} MiB "
              f"resident after build   peak {peak_build / 2**20:.1f} MiB")
        if args.cache_only:
            print(f"    cached {ms_cached:.1f} ms   (readout baseline skipped)")
        else:
            # Against the measured floor where one exists, the constant otherwise.
            floor = args.tol_fidelity if precision is None else max(args.tol_fidelity, precision[1])
            ok_fidelity = fidelity[1] <= floor
            ok_control = control[0] >= args.zeroed_margin * max(fidelity[0], 1e-12)
            if not (ok_fidelity and ok_control):
                failures.append((name, fidelity, control, ok_fidelity, ok_control))
            print(f"    FIDELITY {fidelity[0]:.3e} ({fidelity[1]:.2%})   "
                  f"REPEAT {repeat[0]:.3e}   ZEROED {control[0]:.3e} ({control[1]:.2%})")
            if precision is not None:
                print(f"    PRECISION vs fp32 {precision[0]:.3e} ({precision[1]:.2%})")
            print(f"    cached {ms_cached:.1f} ms   readout {ms_readout:.1f} ms   "
                  f"speedup {ms_readout / ms_cached:.2f}x")

        records.append({
            "seq_name": name, "ids": ids,
            "fidelity": fidelity, "repeat": repeat, "zeroed": control,
            "precision_vs_fp32": precision,
            "cache_bytes": n_bytes, "peak_build_bytes": peak_build,
            "resident_after_build_bytes": resident,
            "ms_cached": ms_cached, "ms_readout": ms_readout,
            "speedup": None if ms_readout is None else ms_readout / ms_cached,
        })
        del cache
        torch.cuda.empty_cache()

    assert records, "every sequence was skipped"

    summary = {
        "cache_bytes": records[0]["cache_bytes"],
        "peak_build_bytes": max(r["peak_build_bytes"] for r in records),
        "resident_after_build_bytes": max(r["resident_after_build_bytes"] for r in records),
        "ms_cached_median": float(np.median([r["ms_cached"] for r in records])),
    }
    if not args.cache_only:
        summary.update({
            "fidelity_max_abs": max(r["fidelity"][0] for r in records),
            "fidelity_max_rel": max(r["fidelity"][1] for r in records),
            "repeat_max_abs": max(r["repeat"][0] for r in records),
            "zeroed_min_abs": min(r["zeroed"][0] for r in records),
            "ms_readout_median": float(np.median([r["ms_readout"] for r in records])),
            "speedup_median": float(np.median([r["speedup"] for r in records])),
        })
        if records[0]["precision_vs_fp32"] is not None:
            summary["precision_max_abs"] = max(r["precision_vs_fp32"][0] for r in records)
            summary["precision_max_rel"] = max(r["precision_vs_fp32"][1] for r in records)
    result = {
        "dtype": args.dtype,
        "num_ref": n, "start": args.start, "query_gap": args.query_gap,
        "query_index": args.start + q_idx, "num_seqs": len(records),
        "tol_fidelity": args.tol_fidelity, "zeroed_margin": args.zeroed_margin,
        "warmup": args.warmup, "iters": args.iters,
        "summary": summary, "per_sequence": records,
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n=== summary over {len(records)} sequences, num_ref = {n}, "
          f"dtype = {args.dtype} ===")
    if "precision_max_rel" in summary:
        print(f"PRECISION max {summary['precision_max_abs']:.3e} "
              f"({summary['precision_max_rel']:.2%}) vs fp32 -- the cost of {args.dtype} "
              "alone, no cache involved")
    if not args.cache_only:
        print(f"FIDELITY  max {summary['fidelity_max_abs']:.3e} ({summary['fidelity_max_rel']:.2%})  "
              f"-- expected at the GEMM floor, tol {args.tol_fidelity:.0e} relative")
        print(f"REPEAT    max {summary['repeat_max_abs']:.3e}  -- the cached path's own floor")
        print(f"ZEROED    min {summary['zeroed_min_abs']:.3e}  -- must be >= "
              f"{args.zeroed_margin:g}x FIDELITY")
    print(f"cache     {summary['cache_bytes'] / 2**20:.1f} MiB claimed "
          f"({summary['cache_bytes'] / 2**20 / n:.1f} MiB per reference frame)")
    print(f"memory    {summary['resident_after_build_bytes'] / 2**20:.1f} MiB resident "
          f"after build, peak {summary['peak_build_bytes'] / 2**20:.1f} MiB. Resident "
          "minus weights should equal the claim; double it and something in the cache "
          "is a view into a larger allocation.")
    if args.cache_only:
        print(f"time      cached {summary['ms_cached_median']:.1f} ms  (no baseline)")
    else:
        print(f"time      cached {summary['ms_cached_median']:.1f} ms  vs  "
              f"readout {summary['ms_readout_median']:.1f} ms  =  "
              f"{summary['speedup_median']:.2f}x")
    print(f"\nwrote {args.out}")

    if args.cache_only:
        print("\ncache_only: cost measured, correctness NOT. FIDELITY, REPEAT and ZEROED come "
              "from the runs that keep the baseline; this run made no claim about them.")
        return
    if failures:
        print(f"\nFAILED on {len(failures)} of {len(records)} sequences:")
        for name, fid, ctl, ok_f, ok_c in failures:
            if not ok_f:
                print(f"  {name}: FIDELITY {fid[1]:.2%} over its floor -- the cached pass moved "
                      "the output MORE than the precision it is running in does, so this is not "
                      "arithmetic. Suspect, in order: the query frame's special tokens (a cached "
                      "pass must NOT get frame 0's), the cache being stale or in the wrong block "
                      "order, RoPE applied twice. Check REPEAT first: if it is also non-zero the "
                      "problem is not the cache at all.")
            if not ok_c:
                print(f"  {name}: ZEROED {ctl[0]:.3e} is not {args.zeroed_margin:g}x worse than "
                      f"FIDELITY {fid[0]:.3e} -- zeroing the cache barely changed the output, "
                      "so the cache is not being read and FIDELITY proves nothing.")
        sys.exit(1)
    print(f"\nOK on all {len(records)} sequences: the cached query pass reproduces the readout "
          "to the GEMM floor, and zeroing the cache destroys it.")


if __name__ == "__main__":
    main()
