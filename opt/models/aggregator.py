# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Project-owned portions of this file are licensed under CC-BY-NC-SA-4.0.
# See LICENSE and NOTICE for details. Third-party notices remain applicable.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Optional, Tuple, Union, List, Dict, Any

from opt.layers import PatchEmbed
from opt.layers.block import Block
from opt.layers.rope import RotaryPositionEmbedding2D, PositionGetter
from opt.layers.vision_transformer import vit_small, vit_base, vit_large, vit_giant2

logger = logging.getLogger(__name__)

_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


class Aggregator(nn.Module):
    """
    The Aggregator applies alternating-attention over input frames,
    as described in VGGT: Visual Geometry Grounded Transformer.

    Remember to set model.train() to enable gradient checkpointing to reduce memory usage.

    Args:
        img_size (int): Image size in pixels.
        patch_size (int): Size of each patch for PatchEmbed.
        embed_dim (int): Dimension of the token embeddings.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of MLP hidden dim to embedding dim.
        num_register_tokens (int): Number of register tokens.
        block_fn (nn.Module): The block type used for attention (Block by default).
        qkv_bias (bool): Whether to include bias in QKV projections.
        proj_bias (bool): Whether to include bias in the output projection.
        ffn_bias (bool): Whether to include bias in MLP layers.
        patch_embed (str): Type of patch embed. e.g., "conv" or "dinov2_vitl14_reg".
        aa_order (list[str]): The order of alternating attention, e.g. ["frame", "global"].
        aa_block_size (int): How many blocks to group under each attention type before switching. If not necessary, set to 1.
        qk_norm (bool): Whether to apply QK normalization.
        rope_freq (int): Base frequency for rotary embedding. -1 to disable.
        init_values (float): Init scale for layer scale.
    """

    def __init__(
        self,
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        num_register_tokens=4,
        block_fn=Block,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        patch_embed="dinov2_vitl14_reg",
        aa_order=["frame", "global"],
        aa_block_size=1,
        qk_norm=True,
        rope_freq=100,
        init_values=0.01,
        use_mask_token=True,
        load_pretrained_patch_embed=False,
        patch_embed_pretrained_path=None,
    ):
        super().__init__()

        self.__build_patch_embed__(
            patch_embed,
            img_size,
            patch_size,
            num_register_tokens,
            embed_dim=embed_dim,
            load_pretrained_patch_embed=load_pretrained_patch_embed,
            patch_embed_pretrained_path=patch_embed_pretrained_path,
        )

        self.use_mask_token = use_mask_token
        if self.use_mask_token:
            self.mask_token = nn.Parameter(torch.randn(1, 1, 3))
            nn.init.normal_(self.mask_token, std=1e-6)

        # Initialize rotary position embedding if frequency > 0
        self.rope = RotaryPositionEmbedding2D(frequency=rope_freq) if rope_freq > 0 else None
        self.position_getter = PositionGetter() if self.rope is not None else None

        self.frame_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(depth)
            ]
        )

        self.global_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(depth)
            ]
        )

        self.depth = depth
        self.aa_order = aa_order
        self.patch_size = patch_size
        self.aa_block_size = aa_block_size

        # Validate that depth is divisible by aa_block_size
        if self.depth % self.aa_block_size != 0:
            raise ValueError(f"depth ({depth}) must be divisible by aa_block_size ({aa_block_size})")

        self.aa_block_num = self.depth // self.aa_block_size

        # Note: We have two camera tokens, one for the first frame and one for the rest
        # The same applies for register tokens
        self.camera_token = nn.Parameter(torch.randn(1, 2, 1, embed_dim))
        self.register_token = nn.Parameter(torch.randn(1, 2, num_register_tokens, embed_dim))

        # The patch tokens start after the camera and register tokens
        self.patch_start_idx = 1 + num_register_tokens

        # Initialize parameters with small values
        nn.init.normal_(self.camera_token, std=1e-6)
        nn.init.normal_(self.register_token, std=1e-6)

        # Register normalization constants as buffers
        for name, value in (("_resnet_mean", _RESNET_MEAN), ("_resnet_std", _RESNET_STD)):
            self.register_buffer(name, torch.FloatTensor(value).view(1, 1, 3, 1, 1), persistent=False)

        self.use_reentrant = False # hardcoded to False

    def __build_patch_embed__(
        self,
        patch_embed,
        img_size,
        patch_size,
        num_register_tokens,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        block_chunks=0,
        init_values=1.0,
        embed_dim=1024,
        load_pretrained_patch_embed=False,
        patch_embed_pretrained_path=None,
    ):
        """
        Build the patch embed layer. If 'conv', we use a
        simple PatchEmbed conv layer. Otherwise, we use a vision transformer.
        """

        if "conv" in patch_embed:
            self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=3, embed_dim=embed_dim)
        else:
            vit_models = {
                "dinov2_vitl14_reg": vit_large,
                "dinov2_vitb14_reg": vit_base,
                "dinov2_vits14_reg": vit_small,
                "dinov2_vitg2_reg": vit_giant2,
            }

            self.patch_embed = vit_models[patch_embed](
                img_size=img_size,
                patch_size=patch_size,
                num_register_tokens=num_register_tokens,
                interpolate_antialias=interpolate_antialias,
                interpolate_offset=interpolate_offset,
                block_chunks=block_chunks,
                init_values=init_values,
            )

            if load_pretrained_patch_embed:
                if patch_embed_pretrained_path is None:
                    logger.warning(
                        "load_pretrained_patch_embed=True but patch_embed_pretrained_path is None; "
                        "skipping patch embed weight loading."
                    )
                elif not os.path.exists(patch_embed_pretrained_path):
                    logger.warning(
                        "Patch embed pretrained path does not exist: %s. Skipping weight loading.",
                        patch_embed_pretrained_path,
                    )
                else:
                    self.patch_embed.load_state_dict(
                        torch.load(patch_embed_pretrained_path, map_location="cpu"),
                        strict=True,
                    )
                    print("Loaded pretrained patch_embed from %s" % patch_embed_pretrained_path)

            # Disable gradient updates for mask token
            if hasattr(self.patch_embed, "mask_token"):
                self.patch_embed.mask_token.requires_grad_(False)

    def forward(
        self,
        images: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
        num_ref_frames: Optional[int] = None,
        kv_cache: Optional[List] = None,
        collect_kv: Optional[List] = None,
    ) -> Tuple[List[torch.Tensor], int]:
        """
        Args:
            images (torch.Tensor): Input images with shape [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width
            num_ref_frames (int, optional): If set, global attention becomes a causal
                readout: frames [0, num_ref_frames) form the reference (cache) block and
                do not see the query frame, while the query frame attends to everything.
                Must equal S - 1 -- exactly one query frame, and it must be last.
                Frame attention is untouched; it is per-frame already. Inference only.
            collect_kv (list, optional): appended with one (k, v) per global block, in
                block order. Run this over the reference frames alone to build a cache.
                A cache build returns an EMPTY output list. The intermediates are what
                a forward is read for, and this forward is not read -- it is run to
                capture k/v. They are not free: each of the 48 blocks retains a view of
                its output, and the 24 concatenations that follow are [B, S, P, 2C]
                each, together about 540 MB per reference frame in fp32. That is more
                than the 258 MB/frame the cache itself costs, and holding it is what
                put n=16 over a 16 GB card. A tracker's mapping phase reads none of it.
            kv_cache (list, optional): a cache from collect_kv. `images` is then the
                query frame ALONE (S=1) and the reference block is never recomputed --
                the same arithmetic num_ref_frames performs, minus the recomputation.
                Its special tokens are taken from the non-first-frame slice, since a
                query frame is by construction not frame 0.

                Frame attention still runs on the query frame, and the DPT heads still
                decode it; what this removes is the reference block's share of the 24
                global blocks, which is the only cost that grows with cache size.

        Returns:
            (list[torch.Tensor], int):
                The list of outputs from the attention blocks,
                and the patch_start_idx indicating where patch tokens begin.
        """
        B, S, C_in, H, W = images.shape

        if C_in != 3:
            raise ValueError(f"Expected 3 input channels, got {C_in}")

        if kv_cache is not None:
            assert not self.training, "kv_cache is an inference-only readout"
            assert num_ref_frames is None, "kv_cache IS the reference block"
            assert collect_kv is None, "collect_kv builds a cache; kv_cache consumes one"
            assert S == 1, f"a cached readout takes exactly one query frame, got S={S}"
            assert len(kv_cache) == len(self.global_blocks), (
                f"cache has {len(kv_cache)} entries, expected one per global block "
                f"({len(self.global_blocks)})"
            )

        if collect_kv is not None:
            assert not self.training, "collect_kv is an inference-only cache build"
            assert num_ref_frames is None, "build the cache from a plain forward over the references"
            assert len(collect_kv) == 0, "pass an empty list; entries are appended in block order"

        if num_ref_frames is not None:
            assert not self.training, "num_ref_frames is an inference-only readout"
            assert num_ref_frames == S - 1, (
                f"num_ref_frames={num_ref_frames} requires S={num_ref_frames + 1}, got S={S}. "
                "The readout supports exactly one query frame, placed last."
            )

        # Normalize images and reshape for patch embed
        images = (images - self._resnet_mean) / self._resnet_std

        if self.use_mask_token and masks is not None:
            if masks.dim() == 5 and masks.shape[2] == 1:
                masks = masks[:, :, 0, ...]
            elif masks.dim() == 5 and masks.shape[-1] == 1:
                masks = masks[..., 0]
            if masks.dim() != 4:
                raise ValueError(f"Expected masks with shape [B, S, H, W], got {tuple(masks.shape)}")
            if masks.shape[:2] != images.shape[:2] or masks.shape[2:] != images.shape[3:]:
                raise ValueError(
                    f"Masks shape {tuple(masks.shape)} must match images [B,S,H,W] = {images.shape[0], images.shape[1], images.shape[3], images.shape[4]}"
                )
            images = images.permute(0, 1, 3, 4, 2)
            mask_bool = masks.to(device=images.device).bool()
            images[~mask_bool] = self.mask_token.to(images.dtype).to(images.device)
            images = images.permute(0, 1, 4, 2, 3)

        # Reshape to [B*S, C, H, W] for patch embedding
        images = images.view(B * S, C_in, H, W)
        patch_tokens = self.patch_embed(images)

        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]

        _, P, C = patch_tokens.shape

        # Expand camera and register tokens to match batch size and sequence length
        # A cached query pass is handed one frame that is NOT frame 0, so it must not
        # be given frame 0's special tokens. Every other pass keeps the original
        # behaviour: first frame gets slice 0, the rest slice 1.
        first_frame = kv_cache is None
        camera_token = slice_expand_and_flatten(self.camera_token, B, S, first_frame=first_frame)
        register_token = slice_expand_and_flatten(self.register_token, B, S, first_frame=first_frame)

        # Concatenate special tokens with patch tokens
        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)

        pos = None
        if self.rope is not None:
            pos = self.position_getter(B * S, H // self.patch_size, W // self.patch_size, device=images.device)

        if self.patch_start_idx > 0:
            # do not use position embedding for special tokens (camera and register tokens)
            # so set pos to 0 for the special tokens
            pos = pos + 1
            pos_special = torch.zeros(B * S, self.patch_start_idx, 2).to(images.device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)

        # update P because we added special tokens
        _, P, C = tokens.shape

        # Global attention flattens to (B, S*P, C), so frames are contiguous blocks and
        # the reference block is simply the first num_ref_frames * P tokens.
        num_ref_tokens = None if num_ref_frames is None else num_ref_frames * P

        # A cache build is run for its k/v, not for its outputs. Keeping the
        # intermediates would cost more memory than the cache does; see the docstring.
        keep_intermediates = collect_kv is None

        frame_idx = 0
        global_idx = 0
        output_list = []

        for _ in range(self.aa_block_num):
            for attn_type in self.aa_order:
                if attn_type == "frame":
                    tokens, frame_idx, frame_intermediates = self._process_frame_attention(
                        tokens, B, S, P, C, frame_idx, pos=pos,
                        keep_intermediates=keep_intermediates,
                    )
                elif attn_type == "global":
                    tokens, global_idx, global_intermediates = self._process_global_attention(
                        tokens, B, S, P, C, global_idx, pos=pos, num_ref_tokens=num_ref_tokens,
                        kv_cache=kv_cache, collect_kv=collect_kv,
                        keep_intermediates=keep_intermediates,
                    )
                else:
                    raise ValueError(f"Unknown attention type: {attn_type}")

            for i in range(len(frame_intermediates)):
                # concat frame and global intermediates, [B x S x P x 2C]
                concat_inter = torch.cat([frame_intermediates[i], global_intermediates[i]], dim=-1)
                output_list.append(concat_inter)
                del concat_inter

        del frame_intermediates
        del global_intermediates
        return output_list, self.patch_start_idx

    def _process_frame_attention(self, tokens, B, S, P, C, frame_idx, pos=None,
                                 keep_intermediates=True):
        """
        Process frame attention blocks. We keep tokens in shape (B*S, P, C).
        """
        # If needed, reshape tokens or positions:
        if tokens.shape != (B * S, P, C):
            tokens = tokens.view(B, S, P, C).view(B * S, P, C)

        if pos is not None and pos.shape != (B * S, P, 2):
            pos = pos.view(B, S, P, 2).view(B * S, P, 2)

        intermediates = []

        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            if self.training:
                tokens = checkpoint(self.frame_blocks[frame_idx], tokens, pos, use_reentrant=self.use_reentrant)
            else:
                tokens = self.frame_blocks[frame_idx](tokens, pos=pos)
            frame_idx += 1
            # A view, not a copy: appending it keeps this block's output alive.
            if keep_intermediates:
                intermediates.append(tokens.view(B, S, P, C))

        return tokens, frame_idx, intermediates

    def _process_global_attention(self, tokens, B, S, P, C, global_idx, pos=None, num_ref_tokens=None,
                                  kv_cache=None, collect_kv=None, keep_intermediates=True):
        """
        Process global attention blocks. We keep tokens in shape (B, S*P, C).
        """
        if tokens.shape != (B, S * P, C):
            tokens = tokens.view(B, S, P, C).view(B, S * P, C)

        if pos is not None and pos.shape != (B, S * P, 2):
            pos = pos.view(B, S, P, 2).view(B, S * P, 2)

        intermediates = []

        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            if self.training:
                assert num_ref_tokens is None, "num_ref_tokens is an inference-only readout"
                tokens = checkpoint(self.global_blocks[global_idx], tokens, pos, use_reentrant=self.use_reentrant)
            else:
                # collect_kv is handed the same list to every block, and blocks run in
                # order, so it comes back indexed by global_idx.
                tokens = self.global_blocks[global_idx](
                    tokens, pos=pos, num_ref_tokens=num_ref_tokens,
                    kv_cache=None if kv_cache is None else kv_cache[global_idx],
                    collect_kv=collect_kv,
                )
            global_idx += 1
            if keep_intermediates:
                intermediates.append(tokens.view(B, S, P, C))

        return tokens, global_idx, intermediates


def slice_expand_and_flatten(token_tensor, B, S, first_frame=True):
    """
    Processes specialized tokens with shape (1, 2, X, C) for multi-frame processing:
    1) Uses the first position (index=0) for the first frame only, unless
       first_frame=False, in which case every frame gets the second position. That is
       for a cached query pass, whose single frame is not frame 0 of the sequence.
    2) Uses the second position (index=1) for all remaining frames (S-1 frames)
    3) Expands both to match batch size B
    4) Concatenates to form (B, S, X, C) where each sequence has 1 first-position token
       followed by (S-1) second-position tokens
    5) Flattens to (B*S, X, C) for processing

    Returns:
        torch.Tensor: Processed tokens with shape (B*S, X, C)
    """

    if not first_frame:
        combined = token_tensor[:, 1:, ...].expand(B, S, *token_tensor.shape[2:])
        return combined.reshape(B * S, *combined.shape[2:])

    # Slice out the "query" tokens => shape (1, 1, ...)
    query = token_tensor[:, 0:1, ...].expand(B, 1, *token_tensor.shape[2:])
    # Slice out the "other" tokens => shape (1, S-1, ...)
    others = token_tensor[:, 1:, ...].expand(B, S - 1, *token_tensor.shape[2:])
    # Concatenate => shape (B, S, ...)
    combined = torch.cat([query, others], dim=1)

    # Finally flatten => shape (B*S, ...)
    combined = combined.view(B * S, *combined.shape[2:])
    return combined
