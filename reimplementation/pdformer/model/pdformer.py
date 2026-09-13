"""PDFormer backbone matching ``reference/PDFormer/libcity/model/.../PDFormer.py``.

Public tensors stay ``[B, T, N, C]``. The output head is the original skip +
two 1x1 convolutions with ``output_window=1`` and ``output_dim=1`` configured
at the convolution channel sizes (not a post-hoc slice of a 12-step forecast).
"""

from __future__ import annotations

from functools import partial
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from reimplementation.common.errors import ReimplementationError
from reimplementation.pdformer.graph import geographic_mask, semantic_mask
from reimplementation.pdformer.model.attention import assert_head_partition
from reimplementation.pdformer.model.embedding import DataEmbedding
from reimplementation.pdformer.model.layers import STEncoderBlock, TokenEmbedding

CODE_VERSION = "0.1.0"


def extract_delay_patterns(
    x: torch.Tensor,
    s_attn_size: int,
    output_dim: int,
) -> torch.Tensor:
    """Sliding ``s_attn_size`` windows with front-zero pad.

    Original loop in ``PDFormer.forward``. At time ``t`` the pattern is the
    ``s_attn_size`` traffic steps ending at ``t``, padded with zeros if ``t``
    is near the start of the input window. The future target is never included.
    """
    if x.ndim != 4:
        raise ReimplementationError(f"pattern input must be [B,T,N,C], got {tuple(x.shape)}")
    _batch, time, _nodes, _channels = x.shape
    if int(s_attn_size) <= 0:
        raise ReimplementationError(f"s_attn_size must be positive, got {s_attn_size}")
    if int(s_attn_size) > int(time):
        raise ReimplementationError(
            f"s_attn_size={s_attn_size} exceeds input_window={time}"
        )
    if int(output_dim) > int(x.size(-1)):
        raise ReimplementationError("pattern output_dim exceeds input channels")
    pieces: list[torch.Tensor] = []
    for index in range(int(s_attn_size)):
        sliced = x[:, : time + index + 1 - int(s_attn_size), :, : int(output_dim)]
        padded = F.pad(
            sliced,
            (0, 0, 0, 0, int(s_attn_size) - 1 - index, 0),
            "constant",
            0,
        )
        pieces.append(padded.unsqueeze(-2))
    patterns = torch.cat(pieces, dim=-2)
    expected = (int(x.size(0)), int(time), int(x.size(2)), int(s_attn_size), int(output_dim))
    if tuple(patterns.shape) != expected:
        raise ReimplementationError(f"pattern tensor {tuple(patterns.shape)} != {expected}")
    return patterns


def _as_bool_mask(mask: np.ndarray | torch.Tensor) -> torch.Tensor:
    tensor = torch.as_tensor(mask)
    if tensor.dtype != torch.bool:
        tensor = tensor.bool()
    return tensor


class PDFormer(nn.Module):
    def __init__(
        self,
        *,
        num_nodes: int,
        input_window: int,
        output_window: int,
        output_dim: int,
        feature_dim: int,
        ext_dim: int,
        embed_dim: int,
        skip_dim: int,
        lape_dim: int,
        geo_num_heads: int,
        sem_num_heads: int,
        t_num_heads: int,
        mlp_ratio: float,
        qkv_bias: bool,
        drop: float,
        attn_drop: float,
        drop_path: float,
        s_attn_size: int,
        t_attn_size: int,
        enc_depth: int,
        type_ln: str,
        add_time_in_day: bool,
        add_day_in_week: bool,
        far_mask_delta: int,
        dtw_delta: int,
        hop_matrix: np.ndarray,
        dtw_matrix: np.ndarray,
        laplacian_pe: np.ndarray,
        pattern_keys: np.ndarray,
        random_flip: bool = False,
        relations_are_test_only: bool = False,
    ) -> None:
        super().__init__()
        if int(output_window) != 1:
            raise ReimplementationError("R-only PDFormer output_window must be 1")
        if int(output_dim) != 1:
            raise ReimplementationError("R-only PDFormer output_dim must be 1")
        if int(s_attn_size) > int(input_window):
            raise ReimplementationError("s_attn_size cannot exceed input_window")
        total_heads, head_dim = assert_head_partition(
            int(embed_dim), int(geo_num_heads), int(sem_num_heads), int(t_num_heads)
        )
        self.num_nodes = int(num_nodes)
        self.input_window = int(input_window)
        self.output_window = int(output_window)
        self.output_dim = int(output_dim)
        self.feature_dim = int(feature_dim)
        self.ext_dim = int(ext_dim)
        self.embed_dim = int(embed_dim)
        self.skip_dim = int(skip_dim)
        self.lape_dim = int(lape_dim)
        self.geo_num_heads = int(geo_num_heads)
        self.sem_num_heads = int(sem_num_heads)
        self.t_num_heads = int(t_num_heads)
        self.num_heads = total_heads
        self.head_dim = head_dim
        self.s_attn_size = int(s_attn_size)
        self.t_attn_size = int(t_attn_size)
        self.enc_depth = int(enc_depth)
        self.type_ln = str(type_ln)
        self.add_time_in_day = bool(add_time_in_day)
        self.add_day_in_week = bool(add_day_in_week)
        self.far_mask_delta = int(far_mask_delta)
        self.dtw_delta = int(dtw_delta)
        self.random_flip = bool(random_flip)
        self.relations_are_test_only = bool(relations_are_test_only)
        self.mlp_hidden_dim = int(embed_dim * mlp_ratio)

        geo = geographic_mask(hop_matrix, far_mask_delta=self.far_mask_delta, transpose=True)
        sem = semantic_mask(dtw_matrix, dtw_delta=self.dtw_delta)
        if geo.shape != (self.num_nodes, self.num_nodes):
            raise ReimplementationError(f"geo mask {geo.shape} != ({self.num_nodes}, {self.num_nodes})")
        if sem.shape != (self.num_nodes, self.num_nodes):
            raise ReimplementationError(f"semantic mask {sem.shape} != ({self.num_nodes}, {self.num_nodes})")
        lap = np.asarray(laplacian_pe, dtype=np.float32)
        if lap.shape != (self.num_nodes, self.lape_dim):
            raise ReimplementationError(
                f"laplacian PE {lap.shape} != ({self.num_nodes}, {self.lape_dim})"
            )
        keys = np.asarray(pattern_keys, dtype=np.float32)
        if keys.ndim != 3 or int(keys.shape[1]) != self.s_attn_size or int(keys.shape[2]) != self.output_dim:
            raise ReimplementationError(
                f"pattern_keys shape {keys.shape} != (n_cluster, {self.s_attn_size}, {self.output_dim})"
            )

        self.register_buffer("geo_mask", _as_bool_mask(geo))
        self.register_buffer("sem_mask", _as_bool_mask(sem))
        self.register_buffer("laplacian_pe", torch.from_numpy(np.array(lap, copy=True)))
        self.register_buffer("pattern_keys", torch.from_numpy(np.array(keys, copy=True)))
        self.register_buffer("hop_matrix", torch.from_numpy(np.asarray(hop_matrix, dtype=np.float32)))
        self.register_buffer("dtw_matrix", torch.from_numpy(np.asarray(dtw_matrix, dtype=np.float32)))

        traffic_dim = self.feature_dim - self.ext_dim
        if traffic_dim != self.output_dim:
            raise ReimplementationError(
                f"traffic feature_dim-ext_dim={traffic_dim} != output_dim={self.output_dim}"
            )
        self.pattern_embeddings = nn.ModuleList(
            [TokenEmbedding(self.s_attn_size, self.embed_dim) for _ in range(self.output_dim)]
        )
        self.enc_embed_layer = DataEmbedding(
            traffic_dim,
            self.embed_dim,
            self.lape_dim,
            drop=drop,
            add_time_in_day=self.add_time_in_day,
            add_day_in_week=self.add_day_in_week,
        )
        enc_dpr = [value.item() for value in torch.linspace(0, drop_path, enc_depth)]
        self.encoder_blocks = nn.ModuleList(
            [
                STEncoderBlock(
                    dim=self.embed_dim,
                    s_attn_size=self.s_attn_size,
                    t_attn_size=self.t_attn_size,
                    geo_num_heads=self.geo_num_heads,
                    sem_num_heads=self.sem_num_heads,
                    t_num_heads=self.t_num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=enc_dpr[index],
                    act_layer=nn.GELU,
                    norm_layer=partial(nn.LayerNorm, eps=1e-6),
                    type_ln=self.type_ln,
                    output_dim=self.output_dim,
                )
                for index in range(enc_depth)
            ]
        )
        self.skip_convs = nn.ModuleList(
            [
                nn.Conv2d(in_channels=self.embed_dim, out_channels=self.skip_dim, kernel_size=1)
                for _ in range(enc_depth)
            ]
        )
        self.end_conv1 = nn.Conv2d(
            in_channels=self.input_window,
            out_channels=self.output_window,
            kernel_size=1,
            bias=True,
        )
        self.end_conv2 = nn.Conv2d(
            in_channels=self.skip_dim,
            out_channels=self.output_dim,
            kernel_size=1,
            bias=True,
        )

    def parameter_count(self) -> int:
        return int(sum(param.numel() for param in self.parameters()))

    def _flipped_laplacian(self) -> torch.Tensor:
        lap = self.laplacian_pe
        if self.training and self.random_flip:
            signs = torch.empty(lap.size(1), device=lap.device, dtype=lap.dtype)
            signs.bernoulli_(0.5)
            signs = signs * 2.0 - 1.0
            return lap * signs.unsqueeze(0)
        return lap

    def encode_patterns(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        patterns = extract_delay_patterns(x, self.s_attn_size, self.output_dim)
        pattern_list = []
        key_list = []
        for channel in range(self.output_dim):
            pattern_list.append(self.pattern_embeddings[channel](patterns[..., channel]).unsqueeze(-1))
            key_list.append(self.pattern_embeddings[channel](self.pattern_keys[..., channel]).unsqueeze(-1))
        return torch.cat(pattern_list, dim=-1), torch.cat(key_list, dim=-1)

    def forward(self, x: torch.Tensor, return_trace: bool = False):
        if x.ndim != 4:
            raise ReimplementationError(f"PDFormer input must be [B,T,N,C], got {tuple(x.shape)}")
        batch, time, nodes, _channels = x.shape
        if int(time) != self.input_window:
            raise ReimplementationError(f"input time {time} != input_window {self.input_window}")
        if int(nodes) != self.num_nodes:
            raise ReimplementationError(f"input nodes {nodes} != {self.num_nodes}")
        x_patterns, pattern_keys = self.encode_patterns(x)
        lap_mx = self._flipped_laplacian()
        enc = self.enc_embed_layer(x, lap_mx)
        skip: torch.Tensor | int = 0
        attn_trace = None
        for index, encoder_block in enumerate(self.encoder_blocks):
            want_attn = bool(return_trace and index == 0)
            encoded = encoder_block(
                enc,
                x_patterns,
                pattern_keys,
                self.geo_mask,
                self.sem_mask,
                return_attention=want_attn,
            )
            if want_attn:
                enc, attn_trace = encoded
            else:
                enc = encoded
            skip = skip + self.skip_convs[index](enc.permute(0, 3, 2, 1))
        skip = self.end_conv1(F.relu(skip.permute(0, 3, 2, 1)))
        skip = self.end_conv2(F.relu(skip.permute(0, 3, 2, 1)))
        prediction = skip.permute(0, 3, 2, 1)
        expected = (batch, self.output_window, self.num_nodes, self.output_dim)
        if tuple(prediction.shape) != expected:
            raise ReimplementationError(f"prediction {tuple(prediction.shape)} != {expected}")
        if not return_trace:
            return prediction
        trace = {
            "input_shape": list(x.shape),
            "pattern_shape": list(x_patterns.shape),
            "pattern_key_shape": list(pattern_keys.shape),
            "encoder_hidden_shape": list(enc.shape),
            "prediction_shape": list(prediction.shape),
            "geo_mask_true_means": "mask_out",
            "sem_mask_true_means": "mask_out",
            "concat_order": ["temporal", "geographic", "semantic"],
            "temporal_used": True,
            "geographic_used": True,
            "semantic_used": True,
            "pattern_used": True,
            "output_window": self.output_window,
            "relations_are_test_only": self.relations_are_test_only,
        }
        if attn_trace is not None:
            trace.update(
                {
                    "temporal_attention_shape": attn_trace["temporal_attention_shape"],
                    "geographic_attention_shape": attn_trace["geographic_attention_shape"],
                    "semantic_attention_shape": attn_trace["semantic_attention_shape"],
                    "first_block_attention": {
                        key: value
                        for key, value in attn_trace.items()
                        if key.endswith("_used") or key.endswith("_shape") or key == "concat_order"
                    },
                }
            )
        return prediction, trace


def build_pdformer_from_config(
    config: Mapping[str, Any],
    *,
    hop_matrix: np.ndarray,
    dtw_matrix: np.ndarray,
    laplacian_pe: np.ndarray,
    pattern_keys: np.ndarray,
    relations_are_test_only: bool,
) -> PDFormer:
    feature_dim = int(config.get("feature_dim", 1))
    ext_dim = int(config.get("ext_dim", 0))
    add_time = bool(config.get("add_time_in_day", True))
    add_week = bool(config.get("add_day_in_week", False))
    if add_time:
        ext_dim += 1
        feature_dim += 1
    if add_week:
        ext_dim += 7
        feature_dim += 7
    return PDFormer(
        num_nodes=int(config["num_nodes"]),
        input_window=int(config.get("input_window", config.get("n_his", 12))),
        output_window=int(config.get("output_window", config.get("horizon", 1))),
        output_dim=int(config.get("output_dim", 1)),
        feature_dim=feature_dim,
        ext_dim=ext_dim,
        embed_dim=int(config.get("embed_dim", 64)),
        skip_dim=int(config.get("skip_dim", 256)),
        lape_dim=int(config.get("lape_dim", 8)),
        geo_num_heads=int(config.get("geo_num_heads", 4)),
        sem_num_heads=int(config.get("sem_num_heads", 2)),
        t_num_heads=int(config.get("t_num_heads", config.get("temporal_num_heads", 2))),
        mlp_ratio=float(config.get("mlp_ratio", 4)),
        qkv_bias=bool(config.get("qkv_bias", True)),
        drop=float(config.get("drop", 0.0)),
        attn_drop=float(config.get("attn_drop", 0.0)),
        drop_path=float(config.get("drop_path", 0.3)),
        s_attn_size=int(config.get("s_attn_size", 3)),
        t_attn_size=int(config.get("t_attn_size", 1)),
        enc_depth=int(config.get("enc_depth", 6)),
        type_ln=str(config.get("type_ln", "pre")),
        add_time_in_day=add_time,
        add_day_in_week=add_week,
        far_mask_delta=int(config.get("far_mask_delta", 7)),
        dtw_delta=int(config.get("dtw_delta", 5)),
        hop_matrix=hop_matrix,
        dtw_matrix=dtw_matrix,
        laplacian_pe=laplacian_pe,
        pattern_keys=pattern_keys,
        random_flip=bool(config.get("random_flip", False)),
        relations_are_test_only=relations_are_test_only,
    )


def in_memory_relations_from_topology(
    topology: np.ndarray,
    *,
    far_mask_delta: int,
    dtw_delta: int,
    lape_dim: int,
    n_cluster: int,
    s_attn_size: int,
    output_dim: int,
    bidir: bool = True,
    seed: int = 42,
) -> dict[str, Any]:
    """Test/smoke relations. Never written as official PDFormer artifacts."""
    from reimplementation.pdformer.graph import (
        TEST_ONLY_RELATIONS,
        hop_shortest_path,
        laplacian_positional_encoding,
        test_only_dtw_matrix,
        test_only_pattern_keys,
    )

    hops = hop_shortest_path(topology, bidir=bidir)
    dtw = test_only_dtw_matrix(int(topology.shape[0]))
    pe = laplacian_positional_encoding(topology, lape_dim)["laplacian_pe"]
    keys = test_only_pattern_keys(
        n_cluster=n_cluster,
        s_attn_size=s_attn_size,
        output_dim=output_dim,
        seed=seed,
    )
    return {
        "kind": TEST_ONLY_RELATIONS,
        "hop_matrix": hops,
        "dtw_matrix": dtw,
        "laplacian_pe": pe,
        "pattern_keys": keys,
        "geo_mask": geographic_mask(hops, far_mask_delta=far_mask_delta, transpose=True),
        "sem_mask": semantic_mask(dtw, dtw_delta=dtw_delta),
        "relations_are_test_only": True,
    }
