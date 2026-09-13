"""Input embeddings from original ``DataEmbedding``.

Enabled components (PeMS-style config, R-only adaptations noted in config)::

    value Linear(feature_dim, embed_dim)
    + sinusoidal positional encoding on time (max_len=100)
    + time-of-day Embedding(1440, D) if add_time_in_day
    + day-of-week Embedding(7, D) if add_day_in_week
    + Laplacian PE Linear(lape_dim, D)
    + Dropout

``feature_dim`` here is the traffic-channel count only. Time channels are
extra axes on ``x``, not projected by ``TokenEmbedding``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from reimplementation.common.errors import ReimplementationError
from reimplementation.pdformer.model.layers import TokenEmbedding


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, max_len: int = 100) -> None:
        super().__init__()
        pe = torch.zeros(max_len, embed_dim).float()
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pe[:, : x.size(1)].unsqueeze(2).expand_as(x).detach()


class LaplacianPE(nn.Module):
    def __init__(self, lape_dim: int, embed_dim: int) -> None:
        super().__init__()
        self.embedding_lap_pos_enc = nn.Linear(lape_dim, embed_dim)

    def forward(self, lap_mx: torch.Tensor) -> torch.Tensor:
        lap_pos_enc = self.embedding_lap_pos_enc(lap_mx).unsqueeze(0).unsqueeze(0)
        return lap_pos_enc


class DataEmbedding(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        embed_dim: int,
        lape_dim: int,
        drop: float = 0.0,
        add_time_in_day: bool = False,
        add_day_in_week: bool = False,
    ) -> None:
        super().__init__()
        self.add_time_in_day = bool(add_time_in_day)
        self.add_day_in_week = bool(add_day_in_week)
        self.embed_dim = int(embed_dim)
        self.feature_dim = int(feature_dim)
        self.value_embedding = TokenEmbedding(feature_dim, embed_dim)
        self.position_encoding = PositionalEncoding(embed_dim)
        if self.add_time_in_day:
            self.minute_size = 1440
            self.daytime_embedding = nn.Embedding(self.minute_size, embed_dim)
        if self.add_day_in_week:
            self.weekday_embedding = nn.Embedding(7, embed_dim)
        self.spatial_embedding = LaplacianPE(lape_dim, embed_dim)
        self.dropout = nn.Dropout(drop)

    def expected_input_channels(self) -> int:
        channels = self.feature_dim
        if self.add_time_in_day:
            channels += 1
        if self.add_day_in_week:
            channels += 7
        return channels

    def forward(self, x: torch.Tensor, lap_mx: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ReimplementationError(f"DataEmbedding expects [B,T,N,C], got {tuple(x.shape)}")
        expected = self.expected_input_channels()
        if int(x.size(-1)) != expected:
            raise ReimplementationError(
                f"input channels {int(x.size(-1))} != embedding layout {expected} "
                f"(traffic={self.feature_dim}, time_in_day={self.add_time_in_day}, "
                f"day_in_week={self.add_day_in_week})"
            )
        origin_x = x
        embedded = self.value_embedding(origin_x[:, :, :, : self.feature_dim])
        embedded = embedded + self.position_encoding(embedded)
        if self.add_time_in_day:
            minute_index = (origin_x[:, :, :, self.feature_dim] * self.minute_size).round().long()
            if torch.any(minute_index < 0) or torch.any(minute_index >= self.minute_size):
                raise ReimplementationError("time-of-day embedding index is outside [0, 1440)")
            embedded = embedded + self.daytime_embedding(minute_index)
        if self.add_day_in_week:
            if not self.add_time_in_day:
                raise ReimplementationError(
                    "original DataEmbedding reads weekday at channel feature_dim+1, "
                    "which assumes time-of-day occupies feature_dim. Enable both or neither."
                )
            weekday_onehot = origin_x[:, :, :, self.feature_dim + 1 : self.feature_dim + 8]
            embedded = embedded + self.weekday_embedding(weekday_onehot.argmax(dim=3))
        embedded = embedded + self.spatial_embedding(lap_mx)
        embedded = self.dropout(embedded)
        return embedded
