# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Project-owned portions of this file are licensed under CC-BY-NC-SA-4.0.
# See LICENSE and NOTICE for details. Third-party notices remain applicable.

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import logging
import os
import warnings

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F

XFORMERS_AVAILABLE = False


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        qk_norm: bool = False,
        fused_attn: bool = True,  # use F.scaled_dot_product_attention or not
        rope=None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = fused_attn

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope

    def forward(self, x: Tensor, pos=None, num_ref_tokens=None, kv_cache=None, collect_kv=None) -> Tensor:
        """
        num_ref_tokens: enables the KV-Tracker-style causal readout. The first
        `num_ref_tokens` positions are the reference (cache) block and attend only
        among themselves; the remaining positions are the single query block and
        attend to [reference, query]. Equivalent to a block attention mask, but
        keeps the fused kernels. Inference only.

        collect_kv: a list. The post-RoPE, post-q/k-norm k and v of this call are
            appended to it as one (k, v) tuple. This is how a cache is built: run a
            plain forward over the reference frames and keep what they projected.

        kv_cache: a (k_ref, v_ref) pair from an earlier collect_kv. `x` is then the
            query frame's tokens ALONE, and it attends to [k_ref, k_q] -- the same
            quantity the readout computes, minus recomputing the reference block.
            The cache is immutable: the query's own k/v are never written back, so
            the reference representation cannot drift, which is the property that
            makes KV-Tracker's cache reusable across frames.

        Both are inference-only, and mutually exclusive with each other's mode:
        num_ref_tokens splits one sequence, kv_cache replaces the reference half of
        that split with stored tensors.

        RoPE is what makes the stored k reusable. It encodes per-frame 2D patch
        coordinates only (aggregator.py's position_getter), identical for every
        frame, so a reference frame's k does not depend on how many frames follow
        it or on where the query sits in time.
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.rope is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)

        if collect_kv is not None:
            assert not self.training, "collect_kv is an inference-only cache build"
            assert kv_cache is None, "collect_kv builds a cache; kv_cache consumes one"
            collect_kv.append((k, v))

        if kv_cache is not None:
            assert not self.training, "kv_cache is an inference-only readout"
            assert num_ref_tokens is None, (
                "kv_cache already IS the reference block; num_ref_tokens would split "
                "the query frame against itself"
            )
            assert self.fused_attn, "kv_cache requires fused_attn"
            k_ref, v_ref = kv_cache
            assert k_ref.shape[0] == B and k_ref.shape[1] == self.num_heads, (
                f"cache {tuple(k_ref.shape)} does not match this attention's [B, H] = {(B, self.num_heads)}"
            )
            k = torch.cat([k_ref, k], dim=2)
            v = torch.cat([v_ref, v], dim=2)
            x = F.scaled_dot_product_attention(q, k, v)
        elif self.fused_attn:
            if num_ref_tokens is None:
                x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0)
            else:
                assert not self.training, "num_ref_tokens is an inference-only readout"
                assert 0 < num_ref_tokens < N, f"num_ref_tokens={num_ref_tokens} not in (0, {N})"
                n = num_ref_tokens
                # The reference slices are made contiguous so this call matches, layout and
                # all, the one a plain n-frame forward issues: a slice along dim 2 of
                # [B, H, N, D] is non-contiguous and SDPA can pick a different kernel for
                # it. The resulting ~1e-6 drift is not harmless downstream -- sonata's
                # GridSample does np.floor(coord / 0.004), so a point within 1e-6 of a
                # voxel boundary flips voxel, which changes count.size, which re-keys
                # np.random.randint(0, count.max(), count.size) % count for every voxel at
                # once. Three orders of amplification, and object-dependent.
                x_ref = F.scaled_dot_product_attention(
                    q[:, :, :n].contiguous(), k[:, :, :n].contiguous(), v[:, :, :n].contiguous()
                )
                x_qry = F.scaled_dot_product_attention(q[:, :, n:], k, v)
                x = torch.cat([x_ref, x_qry], dim=2)
        else:
            assert num_ref_tokens is None, "num_ref_tokens requires fused_attn"
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MemEffAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None, pos=None) -> Tensor:
        assert pos is None
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(x)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x
