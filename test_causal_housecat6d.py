# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Project-owned portions of this file are licensed under CC-BY-NC-SA-4.0.
# See LICENSE and NOTICE for details. Third-party notices remain applicable.

"""
Step 0 of the causal-readout experiment: does OPT-Pose survive one-directional
(KV-Tracker style) attention without retraining?

Two measurements per object sequence, three forwards over the SAME preloaded
frames (the batch is built once and reused, so dataset sampling randomness --
choose_indices, resize augmentation -- cannot leak into the comparison):

  A  bidirectional over frames [0, n)                          reference values
  B  causal readout over frames [0, n], num_ref_frames = n      the thing under test
  C  bidirectional over frames [0, n]                           drift baseline

  SELFCHECK   A  vs  B[:, :n]     must be ~0. This is the plumbing test: every
              path that writes query-frame information back into a reference
              frame's output has to be gated, otherwise reference outputs move
              when a query frame is appended. It covers all five gated paths at
              once (global attention, camera-head trunk, pointmap center,
              cross-frame keypoint attention, z_obj pooling) and needs no
              ground truth. Non-zero here means a bug, not a research result.

  DRIFT       C[:, n]  vs  B[:, n]  on the query frame. This is the actual
              quantity of interest: how far the causal readout moves the
              prediction away from the bidirectional model it was trained as.

Not measured here: accuracy against ground truth, and the pose head. The pose
head is left ungated on purpose (see OPT.forward's num_ref_frames docstring), so
its output is not comparable between A and B by construction. nocs and
pred_kpt_3d, which are what the pose head consumes, are compared instead.

Residual non-determinism, two sources. SDPA/cuDNN kernel selection can differ
between the one-call and two-call attention paths, so the selfcheck threshold is
a tolerance (--tol, fp32 default 1e-4), not bit equality. Larger by three orders:
sonata's GridSample runs in mode="train", which keeps one RANDOM point per voxel
(np.random.randint), so every call to SonataBackbone.extract resamples the point
cloud. Everything downstream of it -- nocs, pred_kpt_3d, z_obj -- is affected.
Each forward therefore reseeds numpy: extract transforms frames in order, so
frames [0, n) draw identically across passes and the query frame's draws land
after them. This changes no model behaviour, only the RNG state at entry.

Usage (see tools/opt_pose_causal.sbatch, this is not meant to be run by hand):

    python test_causal_housecat6d.py \
        --data_root <HouseCat6D root> \
        --checkpoint <abs_pose_housecat.pt> \
        --num_ref 3 --num_seqs 10 \
        --out <results.json>
"""

import argparse
import json
import sys

import numpy as np
import torch

from types import SimpleNamespace

sys.path.append("opt/")
sys.path.append("utils/")

from test_abs_housecat6d import cat_name_2_id, load_opt_model
from training.data.datasets.housecat import HouseCat6DPoseDataset

# Outputs that are defined identically in both modes and are indexed [B, S, ...].
COMPARE_KEYS = [
    "pose_enc",
    "depth",
    "world_points",
    "nocs",
    "pred_kpt_3d",
    "z_obj",
    "abs_translation",
    "abs_size",
]


def build_common_conf(img_size: int, patch_size: int) -> SimpleNamespace:
    """The fields BaseDataset.__init__ and HouseCat6DPoseDataset.__init__ read.

    rescale_aug and landscape_check are off so get_data is deterministic; with
    training=False the scale augmentation is skipped regardless.
    """
    return SimpleNamespace(
        img_size=img_size,
        patch_size=patch_size,
        augs=SimpleNamespace(scales=None),
        rescale=True,
        rescale_aug=False,
        landscape_check=False,
        debug=False,
        training=False,
        get_nearby=False,
        inside_random=False,
        allow_duplicate_img=False,
    )


def build_inputs(batch: dict, seq_len: int, device: str, use_gt_intrinsics: bool) -> dict:
    """Pack one object sequence into the [B=1, S, ...] tensors OPT.forward wants.

    Shapes and dtypes follow test_abs_housecat6d.py's batch construction, which is
    the path the HouseCat6D numbers in tools/FINDINGS.md came from.
    """
    images = torch.from_numpy(np.stack(batch["images"]).astype(np.float32))  # [S,H,W,3]
    images = images.permute(0, 3, 1, 2).to(torch.get_default_dtype()).div(255)
    assert images.shape[0] == seq_len, f"got {images.shape[0]} frames, want {seq_len}"

    cat_name = batch["cat_name"]
    assert cat_name in cat_name_2_id, f"unknown category {cat_name!r}"

    def stack(key, dtype):
        return torch.from_numpy(np.stack(batch[key]).astype(dtype)).unsqueeze(0).to(device)

    return {
        "images": images.unsqueeze(0).to(device),                    # [1,S,3,H,W]
        "choose_indices": stack("choose_indices", np.int64),         # [1,S,K]
        "nocs_gt": stack("nocs", np.float32),                        # [1,S,H,W,3]
        "depth_sensor": stack("depths_sensor", np.float32),          # [1,S,H,W]
        "intrinsics": stack("intrinsics", np.float32),               # [1,S,3,3]
        "cat_labels": torch.tensor([cat_name_2_id[cat_name]], dtype=torch.long, device=device),
        "use_gt_intrinsics": use_gt_intrinsics,
    }


def forward(model: torch.nn.Module, inp: dict, num_frames: int, num_ref_frames, seed: int):
    """One forward over the first num_frames frames, keeping only COMPARE_KEYS on CPU.

    Reseeds numpy first; see the module docstring on sonata's GridSample.
    """
    np.random.seed(seed)
    with torch.inference_mode():
        preds = model(
            images=inp["images"][:, :num_frames],
            cat_labels=inp["cat_labels"],
            choose_indices=inp["choose_indices"][:, :num_frames],
            nocs_gt=inp["nocs_gt"][:, :num_frames],
            depth_sensor=inp["depth_sensor"][:, :num_frames],
            intrinsics=inp["intrinsics"][:, :num_frames],
            use_gt_intrinsics=inp["use_gt_intrinsics"],
            num_ref_frames=num_ref_frames,
        )
    kept = {k: preds[k].detach().float().cpu() for k in COMPARE_KEYS if k in preds}
    del preds
    torch.cuda.empty_cache()
    return kept


def compare(a: torch.Tensor, b: torch.Tensor):
    """Max absolute and relative difference over the entries finite in both."""
    finite = torch.isfinite(a)
    assert torch.equal(finite, torch.isfinite(b)), "NaN/Inf pattern differs between the two runs"
    if finite.sum() == 0:
        return float("nan"), float("nan")
    d = (a[finite] - b[finite]).abs().max().item()
    scale = a[finite].abs().max().item()
    return d, (d / scale if scale > 0 else float("nan"))


def compare_frames(left: dict, right: dict, frame_slice: slice) -> dict:
    out = {}
    keys = sorted(set(left) & set(right))
    assert keys, "no comparable keys in the predictions"
    for k in keys:
        out[k] = compare(left[k][:, frame_slice], right[k][:, frame_slice])
    return out


def fmt(table: dict) -> str:
    return "  ".join(f"{k}={abs_d:.2e}({rel:.1%})" for k, (abs_d, rel) in sorted(table.items()))


def main():
    parser = argparse.ArgumentParser(description="Causal-readout Step 0 on HouseCat6D")
    parser.add_argument("--data_root", type=str, required=True, help="HouseCat6D root")
    parser.add_argument("--checkpoint", type=str, required=True, help="abs_pose_housecat.pt")
    parser.add_argument("--num_ref", type=int, default=3,
                        help="reference (cache) frames; training saw img_nums [2,4], so "
                             "num_ref+1 should stay <= 4 or frame count becomes a second shift")
    parser.add_argument("--num_seqs", type=int, default=10, help="object sequences to evaluate")
    parser.add_argument("--start", type=int, default=0, help="first frame index of the window")
    parser.add_argument("--tol", type=float, default=1e-4, help="selfcheck pass threshold")
    parser.add_argument("--seed", type=int, default=0,
                        help="numpy seed set before every forward, so sonata's train-mode "
                             "GridSample picks the same voxel representatives in each pass")
    parser.add_argument("--use_gt_intrinsics", action="store_true")
    parser.add_argument("--out", type=str, required=True, help="results json")
    args = parser.parse_args()

    seq_len = args.num_ref + 1
    model, device = load_opt_model(checkpoint_path=args.checkpoint)

    common_conf = build_common_conf(img_size=518, patch_size=14)
    ds = HouseCat6DPoseDataset(
        common_conf,
        data_root=args.data_root,
        split="test",
        min_num_images=seq_len,
        sample_num=1024,
    )
    # Go/no-go, same shape as the other lines' loader traps: a wrong extraction
    # layout yields zero sequences rather than an error.
    assert len(ds.seq_names) > 0, f"no object sequences under {args.data_root}"

    names = sorted(ds.seq_names)[: args.num_seqs]
    print(f"{len(ds.seq_names)} sequences available, evaluating {len(names)}, "
          f"window = frames [{args.start}, {args.start + seq_len}), num_ref = {args.num_ref}")

    records = []
    worst_selfcheck = 0.0
    for name in names:
        entries = ds.chunks[name]
        if len(entries) < args.start + seq_len:
            print(f"[skip] {name}: {len(entries)} frames < {args.start + seq_len}")
            continue

        ids = list(range(args.start, args.start + seq_len))
        batch = ds.get_data(seq_name=name, ids=ids, aspect_ratio=1.0)
        if len(batch["images"]) != seq_len:
            # process_record drops frames whose instance mask is too small, and then
            # resamples at random -- which would break the shared-input premise.
            print(f"[skip] {name}: loader returned {len(batch['images'])} of {seq_len} frames")
            continue

        inp = build_inputs(batch, seq_len, device, args.use_gt_intrinsics)

        a = forward(model, inp, num_frames=args.num_ref, num_ref_frames=None, seed=args.seed)
        b = forward(model, inp, num_frames=seq_len, num_ref_frames=args.num_ref, seed=args.seed)
        c = forward(model, inp, num_frames=seq_len, num_ref_frames=None, seed=args.seed)

        selfcheck = compare_frames(a, b, slice(0, args.num_ref))
        drift = compare_frames(c, b, slice(args.num_ref, seq_len))

        seq_worst = max(v[0] for v in selfcheck.values())
        worst_selfcheck = max(worst_selfcheck, seq_worst)

        print(f"--- {name}")
        print(f"    SELFCHECK {fmt(selfcheck)}")
        print(f"    DRIFT     {fmt(drift)}")
        records.append({"seq_name": name, "ids": ids, "selfcheck": selfcheck, "drift": drift})

    assert records, "every sequence was skipped"

    summary = {}
    for stage in ("selfcheck", "drift"):
        keys = sorted(records[0][stage])
        summary[stage] = {
            k: {
                "max_abs": max(r[stage][k][0] for r in records),
                "max_rel": max(r[stage][k][1] for r in records),
            }
            for k in keys
        }

    result = {
        "num_ref": args.num_ref,
        "seq_len": seq_len,
        "start": args.start,
        "num_seqs": len(records),
        "tol": args.tol,
        "summary": summary,
        "per_sequence": records,
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    print("\n=== summary over", len(records), "sequences ===")
    for stage in ("selfcheck", "drift"):
        for k, v in summary[stage].items():
            print(f"{stage:9s} {k:18s} max_abs={v['max_abs']:.3e}  max_rel={v['max_rel']:.2%}")
    print(f"\nwrote {args.out}")

    if worst_selfcheck > args.tol:
        print(f"SELFCHECK FAILED: {worst_selfcheck:.3e} > tol {args.tol:.1e} -- "
              "a cross-frame path is still ungated; the DRIFT numbers are meaningless")
        sys.exit(1)
    print(f"SELFCHECK OK: {worst_selfcheck:.3e} <= tol {args.tol:.1e}")


if __name__ == "__main__":
    main()
