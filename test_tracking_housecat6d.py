# SPDX-License-Identifier: CC-BY-NC-SA-4.0
"""Bounded frozen-reference OPT tracking experiment. See tools/OPT_TRACKING.md.

Methods run in separate processes; no baseline competes with a resident cache.
Inputs and outputs are streamed to disk, never accumulated on the GPU or in RAM.
"""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import time

import numpy as np
import torch

from test_abs_housecat6d import load_opt_model
from test_causal_housecat6d import build_common_conf, build_inputs
from test_kvcache_housecat6d import amp_ctx, set_precision, model_fwd
from training.data.datasets.housecat import HouseCat6DPoseDataset


def read_input(path, device):
    with np.load(path) as data:
        return {key: torch.from_numpy(data[key]).to(device) for key in
                ("images", "intrinsics", "choose_indices", "nocs_gt", "depth_sensor", "cat_labels")}


def invoke(model, inp, args, cache=None, collect=None, method=None, dtype=None):
    method = method or args.method
    with torch.inference_mode(), amp_ctx(dtype or args.dtype):
        return model.forward_tracking(
            inp["images"], mode=args.mode,
            num_ref_frames=args.num_ref if method == "readout" else None,
            opt_cache=cache, collect_cache=collect,
        )


def cpu_predictions(preds):
    return {k: v[:, -1:].float().cpu().numpy() for k, v in preds.items()}


def cache_digest(cache):
    digest = hashlib.sha256()
    for tensor in [t for pair in cache["agg_kv"] for t in pair] + [cache["pose_tokens"]]:
        # One layer at a time on CPU, including bf16 bytes without a float conversion.
        digest.update(tensor.contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


def measure(fn):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    result = fn()
    torch.cuda.synchronize()
    return result, {
        "wall_ms": (time.perf_counter() - start) * 1000,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "resident_bytes": torch.cuda.memory_allocated(),
        "host_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    }


def differences(left, right):
    result = {}
    for key in left:
        a, b = left[key], right[key]
        assert a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all(), key
        delta = np.abs(a - b)
        scale = float(np.abs(b).max())
        result[key] = {"max_abs": float(delta.max()), "mean_abs": float(delta.mean()),
                       "max_rel": float(delta.max()) / max(scale, 1e-12)}
    return result


def viewpoint_arc(entries, run):
    """Max pairwise angle between view directions, in degrees.

    HouseCat6D annotations are object-centred, so the camera centre -R.T @ t read
    off source_extrinsics is already a direction from the object. This is the same
    quantity tools/kvt_run.sh scan --arc-only reports for the KV-Tracker line.
    """
    groups = {"references": [], "queries": []}
    for entry in entries:
        with np.load(run / entry["input"]) as data:
            extrinsics = data["source_extrinsics"].astype(np.float64)
        key = "references" if entry["input"].endswith("references.npz") else "queries"
        for e in extrinsics:
            u, _, vt = np.linalg.svd(e[:3, :3])
            sign = np.ones(3)
            sign[-1] = np.linalg.det(u @ vt)
            rotation = ((u * sign) @ vt).T
            groups[key].append(-(rotation @ e[:3, 3]))
    groups["all"] = groups["references"] + groups["queries"]
    out = {}
    for key, points in groups.items():
        directions = np.stack(points)
        directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
        cosine = np.clip(directions @ directions.T, -1, 1)
        out[key] = float(np.degrees(np.arccos(cosine)).max())
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", choices=("prepare", "verify", "cached", "original", "readout"), required=True)
    p.add_argument("--opt_commit", required=True, help="Model commit recorded by the host launcher")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--data_root", default="/tmp/data/housecat6d")
    p.add_argument("--checkpoint", default="/mnt/projects/gr/3DRecon/opt_pose_ckpt/abs_pose_housecat.pt")
    p.add_argument("--mode", choices=("camera", "geometry"), default="camera")
    p.add_argument("--dtype", choices=("fp32", "bf16", "tf32"), default="bf16")
    p.add_argument("--num_ref", type=int, default=3)
    p.add_argument("--num_seqs", type=int, default=1)
    p.add_argument("--queries", type=int, default=50)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--stride", type=int, default=1)
    # References were consecutive, so a 3-frame cache held one viewpoint sampled
    # three times (0.36 deg apart on bottle-v8_small). Spread them over real baseline.
    p.add_argument("--ref_stride", type=int, default=1)
    # Sequences were taken in sorted order, which always yielded bottle-v8_small.
    # Name them explicitly to choose an object by its viewpoint arc instead.
    p.add_argument("--seq_names", nargs="*", default=None)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tol", type=float, default=1e-4)
    args = p.parse_args()
    # 3 was the in-distribution cap: OPT trained on img_nums [2, 4], so references +
    # query <= 4 is all the checkpoint has seen. Nothing in the model enforces it --
    # the aggregator has no temporal position encoding, so frames are a set -- and
    # Step 1a measured a 48-reference ceiling on 24 GB. Larger caches are an
    # experiment, not a default: query cost grows ~8.5 ms per reference at bf16, so
    # the speedup falls to ~1.2x at 24 references and inverts past ~40.
    assert 1 <= args.num_ref <= 48, "Step 1a measured a 48-reference ceiling on 24 GB"
    assert args.num_seqs > 0 and args.queries > 0 and args.stride > 0 and args.start >= 0
    assert args.ref_stride > 0
    assert args.warmup >= 0 and args.tol > 0
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    set_precision(args.dtype)
    assert args.dtype == "fp32" or torch.cuda.get_device_capability() >= (8, 0)
    manifest_path = args.run / "manifest.json"

    if args.method == "prepare":
        args.run.mkdir(parents=True, exist_ok=False)
        ds = HouseCat6DPoseDataset(build_common_conf(518, 14), data_root=args.data_root,
                                  split="test", min_num_images=args.num_ref + 1, sample_num=1024)
        refs = list(range(args.start, args.start + args.num_ref * args.ref_stride, args.ref_stride))
        queries = list(range(refs[-1] + 1, refs[-1] + 1 + args.queries * args.stride, args.stride))
        sequences = []
        names = sorted(ds.seq_names)
        if args.seq_names:
            unknown = [n for n in args.seq_names if n not in names]
            assert not unknown, f"Unknown sequences {unknown}; available: {names}"
            names = list(args.seq_names)
        for name in names:
            if len(ds.chunks[name]) <= queries[-1]:
                continue
            seqdir = args.run / f"seq{len(sequences):03d}"
            seqdir.mkdir()
            entries = []
            for label, ids in [("references", refs)] + [(f"query{i:05d}", [q]) for i, q in enumerate(queries)]:
                batch = ds.get_data(seq_name=name, ids=ids, aspect_ratio=1.0)
                assert list(batch["ids"]) == ids and len(batch["images"]) == len(ids), (
                    f"{name}: loader substituted or dropped requested frames {ids}: {batch['ids']}")
                inp = build_inputs(batch, len(ids), "cpu", False)
                arrays = {k: v.numpy() for k, v in inp.items() if torch.is_tensor(v)}
                # Exact crop pixels, masks, camera calibration and labels accompany the model tensors.
                for key in ("images", "inst_masks", "extrinsics", "extrinsics_sym", "crop_boxes", "depths"):
                    arrays["source_" + key] = np.stack(batch[key])
                target = seqdir / f"{label}.npz"
                np.savez_compressed(target, **arrays)
                entries.append({"input": str(target.relative_to(args.run)), "ids": ids,
                                "source_paths": batch["filepaths"],
                                "sha256": hashlib.sha256(target.read_bytes()).hexdigest()})
                del batch, inp, arrays
            # Report the viewpoint spread. Nothing printed this before, and a clip
            # whose cache holds one viewpoint will pass every fidelity gate while
            # measuring nothing: bottle-v8_small ran with references 0.36 deg apart.
            arc = viewpoint_arc(entries, args.run)
            print(f"ARC {name}: references {arc['references']:.2f} deg, "
                  f"queries {arc['queries']:.2f} deg, references+queries {arc['all']:.2f} deg",
                  flush=True)
            sequences.append({"name": name, "directory": seqdir.name,
                              "viewpoint_arc_deg": arc, "frames": entries})
            if len(sequences) == args.num_seqs:
                break
        assert len(sequences) == args.num_seqs, "Not enough sequences long enough for the requested clip"
        ckpt_hash = hashlib.sha256()
        with open(args.checkpoint, "rb") as f:
            for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
                ckpt_hash.update(chunk)
        manifest = {"arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    "sequences": sequences, "checkpoint_sha256": ckpt_hash.hexdigest(),
                    "opt_commit": args.opt_commit,
                    "torch": torch.__version__, "cuda": torch.version.cuda,
                    "gpu": torch.cuda.get_device_name(), "protocol": "fixed references; no query insertion; no ground-truth accuracy claim"}
        manifest_path.write_text(json.dumps(manifest, indent=2))
        print("PREPARE OK", flush=True)
        return

    manifest = json.loads(manifest_path.read_text())
    for key in ("num_ref", "dtype", "mode", "checkpoint", "seed"):
        assert getattr(args, key) == manifest["arguments"][key], f"Mismatch with prepared {key}"
    report_dir = args.run / args.method
    report_dir.mkdir(exist_ok=False)
    model, device = load_opt_model(args.checkpoint)
    records = []
    for sequence in manifest["sequences"]:
        seqdir = report_dir / sequence["directory"]
        seqdir.mkdir()
        ref = read_input(args.run / sequence["frames"][0]["input"], device)
        query_frames = sequence["frames"][1:]
        if args.method == "verify":
            q = read_input(args.run / query_frames[0]["input"], device)
            combined = {k: torch.cat([ref[k], q[k]], dim=1) if k != "cat_labels" else ref[k] for k in ref}
            # Matched original and readout baselines first, with no live cache.
            original = cpu_predictions(invoke(model, combined, args, method="original"))
            readout = cpu_predictions(invoke(model, combined, args, method="readout"))
            # Retention compares forward_tracking's ordinary path against the STOCK
            # forward. That stock path runs the NOCS/DPT branch over every frame and
            # allocates ~6.4 GiB at 25 frames, so it OOMs on 24 GB past ~12 references.
            # The claim it supports -- that the refactor left the ordinary path
            # unchanged -- needs one reference, not the whole set, so pin it to a pair
            # and keep the check's cost independent of num_ref.
            pair = {k: torch.cat([ref[k][:, -1:], q[k]], dim=1) if k != "cat_labels" else ref[k]
                    for k in ref}
            retention_base = cpu_predictions(invoke(model, pair, args, method="original"))
            legacy_input = dict(pair, use_gt_intrinsics=False)
            legacy_preds = model_fwd(model, legacy_input, args.seed, args.dtype)
            legacy = cpu_predictions({k: legacy_preds[k] for k in retention_base})
            del legacy_preds, pair
            precision = {k: {"max_rel": 0.0} for k in original}
            if args.dtype != "fp32":
                set_precision("fp32")
                fp32 = cpu_predictions(invoke(model, combined, args, method="readout", dtype="fp32"))
                precision = differences(readout, fp32)
                del fp32
                set_precision(args.dtype)
            del combined, legacy_input
            cache = {}
            invoke(model, ref, args, collect=cache, method="cached")
            fingerprint = cache_digest(cache)
            cached = cpu_predictions(invoke(model, q, args, cache=cache, method="cached"))
            q_last = read_input(args.run / query_frames[-1]["input"], device)
            invoke(model, q_last, args, cache=cache, method="cached")
            assert cache_digest(cache) == fingerprint, "Query modified reference memory"
            del cache, q_last
            # Real independent rebuild; the previous cache is no longer resident.
            cache = {}
            invoke(model, ref, args, collect=cache, method="cached")
            repeat = cpu_predictions(invoke(model, q, args, cache=cache, method="cached"))
            with torch.inference_mode():
                for k, v in cache["agg_kv"]:
                    k.zero_()
                    v.zero_()
                cache["pose_tokens"].zero_()
            zeroed = cpu_predictions(invoke(model, q, args, cache=cache, method="cached"))
            del cache, q
            fidelity = differences(cached, readout)
            retention = differences(retention_base, legacy)
            repeat_diff = differences(cached, repeat)
            control = differences(zeroed, readout)
            floors = {k: max(args.tol, precision[k]["max_rel"]) for k in original}
            checks = {
                "retention": all(retention[k]["max_rel"] <= floors[k] for k in original),
                "fidelity": all(fidelity[k]["max_rel"] <= floors[k] for k in original),
                "repeat": all(repeat_diff[k]["max_rel"] <= args.tol for k in original),
                "zeroed": all(control[k]["max_abs"] >= 100 * max(fidelity[k]["max_abs"], 1e-12) for k in original),
            }
            record = {"sequence": sequence["name"], "checks": checks, "fidelity": fidelity,
                      "retention": retention, "repeat": repeat_diff, "zeroed": control,
                      "precision": precision, "floors": floors,
                      "causal_difference": differences(cached, original)}
            (seqdir / "checks.json").write_text(json.dumps(record, indent=2))
            print(json.dumps(record), flush=True)
            assert all(checks.values()), "VERIFY FAILED; see saved per-key checks"
            records.append(record)
        else:
            cache = None
            build_stats = None
            cache_size = 0
            if args.method == "cached":
                # Warm up construction, release it, then measure one independent build.
                cold_build = None
                for warmup_index in range(args.warmup):
                    cache = {}
                    _, warm_stats = measure(lambda: invoke(model, ref, args, collect=cache))
                    if warmup_index == 0:
                        cold_build = warm_stats
                    del cache
                cache = {}
                _, build_stats = measure(lambda: invoke(model, ref, args, collect=cache))
                build_stats["cold_build"] = cold_build
                tensors = [t for pair in cache["agg_kv"] for t in pair] + [cache["pose_tokens"]]
                cache_size = sum(t.numel() * t.element_size() for t in tensors)
                # Storage accounting catches the historical fused-qkv view bug.
                assert all(t.untyped_storage().nbytes() == t.numel() * t.element_size() for t in tensors)
                del tensors
                fingerprint = cache_digest(cache)
            measurements = []
            with (seqdir / "frames.jsonl").open("w") as log:
                for index, frame in enumerate(query_frames):
                    inp = read_input(args.run / frame["input"], device)
                    if args.method != "cached":
                        inp = {k: torch.cat([ref[k], inp[k]], dim=1) if k != "cat_labels" else ref[k] for k in ref}
                    if index == 0:
                        for _ in range(args.warmup):
                            invoke(model, inp, args, cache=cache)
                    preds, stats = measure(lambda: invoke(model, inp, args, cache=cache))
                    arrays = cpu_predictions(preds)
                    assert all(np.isfinite(v).all() for v in arrays.values()), "Nonfinite model prediction"
                    np.savez_compressed(seqdir / f"query{index:05d}.npz", **arrays)
                    stats.update({"query_index": index, "ids": frame["ids"]})
                    log.write(json.dumps(stats) + "\n")
                    log.flush()
                    measurements.append(stats)
                    del preds, arrays, inp
            if cache is not None:
                assert cache_digest(cache) == fingerprint, "Streaming queries modified the cache"
            records.append({"sequence": sequence["name"], "directory": sequence["directory"],
                            "build": build_stats, "cache_bytes": cache_size, "frames": measurements})
            del cache
        del ref
        torch.cuda.empty_cache()
    report = {"method": args.method, "mode": args.mode, "dtype": args.dtype,
              "gpu": torch.cuda.get_device_name(), "warmup": args.warmup, "records": records}
    (report_dir / "results.json").write_text(json.dumps(report, indent=2))
    print(f"{args.method.upper()} OK", flush=True)


if __name__ == "__main__":
    main()
