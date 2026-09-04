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
  ZEROED    QUERY with a zeroed cache, against READOUT. A negative control: if
            this is not orders WORSE than FIDELITY, the cache is not being read
            and FIDELITY proves nothing. Step 0 shipped two controls that could
            not fail; this one can.

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
import json
import sys

import numpy as np
import torch

sys.path.append("opt/")
sys.path.append("utils/")

from test_abs_housecat6d import load_opt_model
from test_causal_housecat6d import build_common_conf, build_inputs, compare, select_query
from training.data.datasets.housecat import HouseCat6DPoseDataset


def readout_fwd(agg, images, n):
    """Step 0.5's pass: [0, n] with the reference block masked off from the query.

    Returns the GPU tensors. The timing calls use this directly, so no device
    transfer lands inside a measurement.
    """
    with torch.inference_mode():
        out, _ = agg(images=images, num_ref_frames=n)
        return out


def query_fwd(agg, images_query, cache):
    """Tracking phase: one frame, attending to the stored block."""
    with torch.inference_mode():
        out, _ = agg(images=images_query, kv_cache=cache)
        return out


def build_cache(agg, images_ref):
    """Mapping phase: a plain forward over the reference frames, keeping their k/v."""
    cache = []
    with torch.inference_mode():
        agg(images=images_ref, collect_kv=cache)
    return cache


def run_readout(agg, images, n):
    """readout_fwd, query frame only, on the CPU for comparison."""
    with torch.inference_mode():
        return [t[:, n : n + 1].float().cpu() for t in readout_fwd(agg, images, n)]


def run_query(agg, images_query, cache):
    """query_fwd on the CPU for comparison."""
    with torch.inference_mode():
        return [t.float().cpu() for t in query_fwd(agg, images_query, cache)]


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
    parser.add_argument("--tol_fidelity", type=float, default=1e-4,
                        help="ceiling on the RELATIVE difference between the cached query pass "
                             "and the readout. The expected magnitude is the ~1e-6 GEMM floor; "
                             "this is set two orders above it, because what would signal a real "
                             "error -- a wrong special token, an unapplied RoPE, an off-by-one "
                             "in the cache -- lands far higher than that, not just above it")
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
          f"num_ref = {n}, query_gap = {args.query_gap}, "
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
        cache = build_cache(agg, images_ref)
        assert len(cache) == len(agg.global_blocks)
        n_bytes = cache_bytes(cache)

        fidelity = repeat = control = None
        if not args.cache_only:
            query = run_query(agg, images_query, cache)

            np.random.seed(args.seed)
            readout = run_readout(agg, images, n)
            fidelity = compare_lists(query, readout)

            # A second, independently built cache: the floor of the cached path itself.
            # Freed before the control allocates a third one -- each cache is 258 MiB
            # per reference frame in fp32, and three of them plus the weights and a
            # readout pass are what put n=8 over a 16 GB card.
            cache2 = build_cache(agg, images_ref)
            query2 = run_query(agg, images_query, cache2)
            repeat = compare_lists(query, query2)
            del cache2, query2
            torch.cuda.empty_cache()

            with torch.inference_mode():
                zeroed = [(torch.zeros_like(k), torch.zeros_like(v)) for k, v in cache]
            query_zeroed = run_query(agg, images_query, zeroed)
            control = compare_lists(query_zeroed, readout)
            del zeroed, query_zeroed, query, readout
            torch.cuda.empty_cache()

        # Timed on the GPU tensors: a device transfer is not part of what a tracker
        # would pay per frame, and it is the same for both paths anyway.
        ms_cached = time_ms(lambda: query_fwd(agg, images_query, cache), args.warmup, args.iters)
        ms_readout = None if args.cache_only else time_ms(
            lambda: readout_fwd(agg, images, n), args.warmup, args.iters)

        print(f"--- {name}")
        if args.cache_only:
            print(f"    cache {n_bytes / 2**20:.1f} MiB   cached {ms_cached:.1f} ms   "
                  "(readout baseline skipped)")
        else:
            ok_fidelity = fidelity[1] <= args.tol_fidelity
            ok_control = control[0] >= args.zeroed_margin * max(fidelity[0], 1e-12)
            if not (ok_fidelity and ok_control):
                failures.append((name, fidelity, control, ok_fidelity, ok_control))
            print(f"    FIDELITY {fidelity[0]:.3e} ({fidelity[1]:.2%})   "
                  f"REPEAT {repeat[0]:.3e}   ZEROED {control[0]:.3e} ({control[1]:.2%})")
            print(f"    cache {n_bytes / 2**20:.1f} MiB   "
                  f"cached {ms_cached:.1f} ms   readout {ms_readout:.1f} ms   "
                  f"speedup {ms_readout / ms_cached:.2f}x")

        records.append({
            "seq_name": name, "ids": ids,
            "fidelity": fidelity, "repeat": repeat, "zeroed": control,
            "cache_bytes": n_bytes, "ms_cached": ms_cached, "ms_readout": ms_readout,
            "speedup": None if ms_readout is None else ms_readout / ms_cached,
        })
        del cache
        torch.cuda.empty_cache()

    assert records, "every sequence was skipped"

    summary = {
        "cache_bytes": records[0]["cache_bytes"],
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
    result = {
        "num_ref": n, "start": args.start, "query_gap": args.query_gap,
        "query_index": args.start + q_idx, "num_seqs": len(records),
        "tol_fidelity": args.tol_fidelity, "zeroed_margin": args.zeroed_margin,
        "warmup": args.warmup, "iters": args.iters,
        "summary": summary, "per_sequence": records,
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n=== summary over {len(records)} sequences, num_ref = {n} ===")
    if not args.cache_only:
        print(f"FIDELITY  max {summary['fidelity_max_abs']:.3e} ({summary['fidelity_max_rel']:.2%})  "
              f"-- expected at the GEMM floor, tol {args.tol_fidelity:.0e} relative")
        print(f"REPEAT    max {summary['repeat_max_abs']:.3e}  -- the cached path's own floor")
        print(f"ZEROED    min {summary['zeroed_min_abs']:.3e}  -- must be >= "
              f"{args.zeroed_margin:g}x FIDELITY")
    print(f"cache     {summary['cache_bytes'] / 2**20:.1f} MiB "
          f"({summary['cache_bytes'] / 2**20 / n:.1f} MiB per reference frame)")
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
                print(f"  {name}: FIDELITY {fid[1]:.2%} over tol -- the cached pass is not "
                      "computing what the readout computes. Suspect, in order: the query "
                      "frame's special tokens (a cached pass must NOT get frame 0's), the "
                      "cache being stale or in the wrong block order, RoPE applied twice.")
            if not ok_c:
                print(f"  {name}: ZEROED {ctl[0]:.3e} is not {args.zeroed_margin:g}x worse than "
                      f"FIDELITY {fid[0]:.3e} -- zeroing the cache barely changed the output, "
                      "so the cache is not being read and FIDELITY proves nothing.")
        sys.exit(1)
    print(f"\nOK on all {len(records)} sequences: the cached query pass reproduces the readout "
          "to the GEMM floor, and zeroing the cache destroys it.")


if __name__ == "__main__":
    main()
