# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Project-owned portions of this file are licensed under CC-BY-NC-SA-4.0.
# See LICENSE and NOTICE for details. Third-party notices remain applicable.

"""
Step 0 of the causal-readout experiment: does OPT-Pose survive one-directional
(KV-Tracker style) attention without retraining?

Four measurements per object sequence, five forwards over the SAME preloaded
window of n+2 frames (the batch is built once and reused, so dataset sampling
randomness -- choose_indices, resize augmentation -- cannot leak into the
comparison):

  A   bidirectional over frames [0, n)                         reference values
  B   causal readout over frames [0, n], num_ref_frames = n    the thing under test
  B*  B again, byte-identical inputs                           the noise floor
  B'  causal readout over frames [0, n) + {n+1}                same, other query
  C   bidirectional over frames [0, n]                         drift baseline

  REPEAT      B[:, :n]  vs  B*[:, :n]. Same values and same shapes, but built as
              a SEPARATE tensor, exactly the way B' is -- which is what makes it
              a floor LEAK can be read against. Handing B and B* the identical
              tensor instead measures nothing at all: it returned exactly
              0.000e+00 on every deterministic key, while B', running on a
              freshly built tensor of the same shape, sat at ~3e-6. The gate then
              failed on all ten sequences against a floor that no pass could
              reach, because REPEAT and LEAK differed in two things (memory and
              content) while being compared as if they differed in one.

  LEAK        B[:, :n]  vs  B'[:, :n], where B' is the same readout with a
              DIFFERENT query frame appended. This is the gate, and it passes
              when LEAK <= REPEAT: shapes are identical between B and B', so the
              only thing that differs beyond allocator noise is the query
              frame's content. LEAK above REPEAT is information flowing from the
              query frame into the cache, i.e. an ungated cross-frame path.

  SELFCHECK   A  vs  B[:, :n]. Reported, not gated. It cannot be zero: pass B
              runs the projections over (n+1)*P tokens instead of n*P, which
              changes the GEMM shapes and leaves ~1e-6 in the reference frames'
              geometry no gating can remove. Downstream of sonata's GridSample
              that is amplified by up to three orders -- np.floor(coord/0.004)
              flips a near-boundary point into another voxel, count.size
              changes, and idx_select is keyed on the voxel counts, so one flip
              re-draws the representative of every voxel. The result is chaotic
              in the object and worthless as a pass/fail criterion, which is why
              LEAK exists. Read it as a sensitivity measurement instead.

  DRIFT       C[:, n]  vs  B[:, n]  on the query frame. This is the actual
              quantity of interest: how far the causal readout moves the
              prediction away from the bidirectional model it was trained as.

Not measured here: accuracy against ground truth, and the pose head. The pose
head is left ungated on purpose (see OPT.forward's num_ref_frames docstring), so
its output is not comparable between A and B by construction. nocs and
pred_kpt_3d, which are what the pose head consumes, are compared instead.

Residual non-determinism, three sources. SDPA/cuDNN kernel selection and GEMM
tiling differ between an n-frame and an (n+1)-frame forward, which is what keeps
SELFCHECK off zero; LEAK is not exposed to that one, since B and B' share every
shape. Allocator addresses differ between any two calls and kernel selection is
alignment sensitive, which is what REPEAT measures and LEAK is judged against.
Larger by three orders: sonata's GridSample runs in mode="train", keeping one RANDOM point per voxel
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


def select_query(inp: dict, n: int, q_idx: int) -> dict:
    """Frames [0, n) followed by frame q_idx, as a freshly allocated tensor.

    Every readout pass is built through this, including the ones whose query frame
    is frame n, so that B, B* and B' differ in the query frame's CONTENT and in
    nothing else. Slicing `inp` for B while B' came out of a torch.cat was enough
    to break the gate: B and B* then shared one tensor at one address and REPEAT
    was a replay of the identical computation, exactly 0, while B' ran on separate
    memory. Only entries carrying a frame axis are reindexed; cat_labels is [B]
    and use_gt_intrinsics is a bool.
    """
    return {
        k: torch.cat([v[:, :n], v[:, q_idx:q_idx + 1]], dim=1)
        if torch.is_tensor(v) and v.dim() >= 2 and v.shape[1] >= q_idx + 1 else v
        for k, v in inp.items()
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
    parser.add_argument("--tol", type=float, default=1e-4,
                        help="reporting threshold for SELFCHECK; not a gate (see module docstring)")
    parser.add_argument("--tol_leak", type=float, default=0.0,
                        help="LEAK pass threshold. Zero on purpose: B and B' differ only in the "
                             "query frame's content, at identical shapes, so a correctly gated "
                             "readout reproduces the reference frames bit for bit")
    parser.add_argument("--seed", type=int, default=0,
                        help="numpy seed set before every forward, so sonata's train-mode "
                             "GridSample picks the same voxel representatives in each pass")
    parser.add_argument("--use_gt_intrinsics", action="store_true")
    parser.add_argument("--out", type=str, required=True, help="results json")
    args = parser.parse_args()

    seq_len = args.num_ref + 1
    # One frame beyond the readout window, used only as the alternative query in
    # pass B'. It never enters the reference block.
    window = seq_len + 1
    model, device = load_opt_model(checkpoint_path=args.checkpoint)

    common_conf = build_common_conf(img_size=518, patch_size=14)
    ds = HouseCat6DPoseDataset(
        common_conf,
        data_root=args.data_root,
        split="test",
        min_num_images=window,
        sample_num=1024,
    )
    # Go/no-go, same shape as the other lines' loader traps: a wrong extraction
    # layout yields zero sequences rather than an error.
    assert len(ds.seq_names) > 0, f"no object sequences under {args.data_root}"

    names = sorted(ds.seq_names)[: args.num_seqs]
    print(f"{len(ds.seq_names)} sequences available, evaluating {len(names)}, "
          f"window = frames [{args.start}, {args.start + window}), num_ref = {args.num_ref}, "
          f"alternative query = frame {args.start + seq_len}")

    records = []
    leaking = []
    for name in names:
        entries = ds.chunks[name]
        if len(entries) < args.start + window:
            print(f"[skip] {name}: {len(entries)} frames < {args.start + window}")
            continue

        ids = list(range(args.start, args.start + window))
        batch = ds.get_data(seq_name=name, ids=ids, aspect_ratio=1.0)
        if len(batch["images"]) != window:
            # process_record drops frames whose instance mask is too small, and then
            # resamples at random -- which would break the shared-input premise.
            print(f"[skip] {name}: loader returned {len(batch['images'])} of {window} frames")
            continue

        inp = build_inputs(batch, window, device, args.use_gt_intrinsics)
        # All three readout passes are constructed identically, so the only thing
        # that varies across them is the query frame's content. See select_query.
        inp_b = select_query(inp, args.num_ref, args.num_ref)
        inp_rep = select_query(inp, args.num_ref, args.num_ref)
        inp_alt = select_query(inp, args.num_ref, args.num_ref + 1)
        # The premise the whole gate rests on: B and B' agree on the reference block.
        assert torch.equal(inp_b["images"][:, : args.num_ref],
                           inp_alt["images"][:, : args.num_ref]), "reference frames differ"

        a = forward(model, inp, num_frames=args.num_ref, num_ref_frames=None, seed=args.seed)
        b = forward(model, inp_b, num_frames=seq_len, num_ref_frames=args.num_ref, seed=args.seed)
        # B* before B', so the floor is measured one call downstream of B -- the
        # same position B' occupies relative to the allocator.
        b_rep = forward(model, inp_rep, num_frames=seq_len, num_ref_frames=args.num_ref, seed=args.seed)
        b_alt = forward(model, inp_alt, num_frames=seq_len, num_ref_frames=args.num_ref, seed=args.seed)
        c = forward(model, inp_b, num_frames=seq_len, num_ref_frames=None, seed=args.seed)

        repeat = compare_frames(b, b_rep, slice(0, args.num_ref))
        leak = compare_frames(b, b_alt, slice(0, args.num_ref))
        selfcheck = compare_frames(a, b, slice(0, args.num_ref))
        drift = compare_frames(c, b, slice(args.num_ref, seq_len))

        # Per key, against that key's own floor: the amplification through
        # GridSample is three orders larger for nocs than for depth, so one
        # scalar threshold across keys would be meaningless.
        over = {k: (leak[k][0], repeat[k][0]) for k in leak if leak[k][0] > repeat[k][0]}
        if over:
            leaking.append((name, over))

        print(f"--- {name}")
        print(f"    REPEAT    {fmt(repeat)}")
        print(f"    LEAK      {fmt(leak)}")
        print(f"    SELFCHECK {fmt(selfcheck)}")
        print(f"    DRIFT     {fmt(drift)}")
        if over:
            print(f"    ^^ LEAK over REPEAT on: {', '.join(sorted(over))}")
        records.append({"seq_name": name, "ids": ids, "repeat": repeat,
                        "leak": leak, "selfcheck": selfcheck, "drift": drift})

    assert records, "every sequence was skipped"

    summary = {}
    for stage in ("repeat", "leak", "selfcheck", "drift"):
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
        "tol_leak": args.tol_leak,
        "summary": summary,
        "per_sequence": records,
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    print("\n=== summary over", len(records), "sequences ===")
    for stage in ("repeat", "leak", "selfcheck", "drift"):
        for k, v in summary[stage].items():
            print(f"{stage:9s} {k:18s} max_abs={v['max_abs']:.3e}  max_rel={v['max_rel']:.2%}")
    print(f"\nwrote {args.out}")

    if leaking:
        print(f"LEAK FAILED on {len(leaking)} of {len(records)} sequences -- changing only the "
              "query frame's content moved the reference frames further than re-running the "
              "identical computation does, so a cross-frame path is still ungated and the "
              "DRIFT numbers are meaningless:")
        for name, over in leaking:
            detail = "  ".join(f"{k}={l:.2e} vs floor {r:.2e}" for k, (l, r) in sorted(over.items()))
            print(f"  {name}: {detail}")
        sys.exit(1)
    print(f"LEAK OK on all {len(records)} sequences: every key is at or below its own "
          "repeat-run floor, so no query-frame information reaches the reference block. "
          "SELFCHECK above that floor is the GEMM-shape difference between an n-frame and "
          "an (n+1)-frame forward, amplified by GridSample -- not a gating failure.")


if __name__ == "__main__":
    main()
