"""Seq2seq GCN-LSTM for 12-to-1 full-flow reconstruction.

Label: Seq2seq GCN-LSTM (Reimplemented, 12-to-1 Adaptation).
This is not an official implementation. Abdelraouf, Abdel-Aty, and Mahmoud
(IEEE T-ITS 2023, DOI 10.1109/TITS.2022.3168865) do not release code, and the
paper does not fully specify the LSTM layout, decoder state, or tuned widths.

The decoder is the fixed-context reading of Fig. 4 requested for this port:
encoder context goes through the decoder GCN once, then the decoder LSTM
steps from a zero state. Predicted values are not fed back.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from reimplementation.common.errors import ReimplementationError
from reimplementation.s2s_gcn_lstm.graph import symmetric_normalized_adjacency

CODE_VERSION = "0.1.0"
MODEL_NAME = "s2s_gcn_lstm"
MODEL_LABEL = "Seq2seq GCN-LSTM (Reimplemented, 12-to-1 Adaptation)"
PAPER_DOI = "10.1109/TITS.2022.3168865"


class KipfGCN(nn.Module):
    """One Kipf & Welling layer, ``ReLU(Â H W)``, without a bias term."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        if int(in_features) < 1 or int(out_features) < 1:
            raise ReimplementationError("GCN widths must be positive")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.weight = nn.Parameter(torch.empty(self.in_features, self.out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features: torch.Tensor, a_hat: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3:
            raise ReimplementationError(f"GCN features must be [B, N, C], got {tuple(features.shape)}")
        if a_hat.ndim != 2 or a_hat.shape[0] != a_hat.shape[1]:
            raise ReimplementationError(f"Â must be square, got {tuple(a_hat.shape)}")
        if int(features.shape[1]) != int(a_hat.shape[0]):
            raise ReimplementationError(
                f"GCN node count {features.shape[1]} != adjacency {a_hat.shape[0]}"
            )
        if int(features.shape[-1]) != self.in_features:
            raise ReimplementationError(
                f"GCN input channels {features.shape[-1]} != {self.in_features}"
            )
        support = torch.matmul(features, self.weight)
        batch, nodes, width = support.shape
        mixed = torch.matmul(a_hat, support.permute(1, 0, 2).reshape(nodes, batch * width))
        output = mixed.reshape(nodes, batch, width).permute(1, 0, 2).contiguous()
        return torch.relu(output)


class GraphConvolutionStack(nn.Module):
    """Independent GCN stack. Encoder and decoder each own one instance."""

    def __init__(self, in_features: int, hidden_features: int, n_layers: int) -> None:
        super().__init__()
        if int(n_layers) < 1:
            raise ReimplementationError("GCN stack needs at least one layer")
        layers: list[KipfGCN] = []
        channels = int(in_features)
        for _ in range(int(n_layers)):
            layers.append(KipfGCN(channels, int(hidden_features)))
            channels = int(hidden_features)
        self.layers = nn.ModuleList(layers)
        self.out_features = int(hidden_features)

    def forward(self, features: torch.Tensor, a_hat: torch.Tensor) -> torch.Tensor:
        hidden = features
        for layer in self.layers:
            hidden = layer(hidden, a_hat)
        return hidden


class S2SGCNLSTM(nn.Module):
    """Encoder GCN → node-wise LSTM, then decoder GCN → LSTM → Dropout → linear FC.

    Public layout is ``[B, T, N, C]`` in and ``[B, T_out, N, C_out]`` out.
    Nodes share each LSTM's weights. Spatial mixing happens only in the GCNs.
    """

    def __init__(
        self,
        adjacency: np.ndarray,
        *,
        num_nodes: int,
        input_channels: int = 1,
        output_channels: int = 1,
        input_window: int = 12,
        output_window: int = 1,
        gcn_hidden_dim: int = 64,
        lstm_hidden_dim: int = 64,
        encoder_gcn_layers: int = 1,
        decoder_gcn_layers: int = 1,
        lstm_layers: int = 1,
        encoder_lstm_layers: int | None = None,
        decoder_lstm_layers: int | None = None,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if int(input_window) < 1 or int(output_window) < 1:
            raise ReimplementationError("input_window and output_window must be positive")
        encoder_lstm = int(lstm_layers if encoder_lstm_layers is None else encoder_lstm_layers)
        decoder_lstm = int(lstm_layers if decoder_lstm_layers is None else decoder_lstm_layers)
        if int(encoder_gcn_layers) < 1 or int(decoder_gcn_layers) < 1:
            raise ReimplementationError("GCN layer counts must be positive")
        if encoder_lstm < 1 or decoder_lstm < 1:
            raise ReimplementationError("LSTM layer counts must be positive")
        if not 0.0 <= float(dropout) < 1.0:
            raise ReimplementationError(f"dropout must be in [0, 1), got {dropout}")
        self.num_nodes = int(num_nodes)
        self.input_channels = int(input_channels)
        self.output_channels = int(output_channels)
        self.input_window = int(input_window)
        self.output_window = int(output_window)
        self.gcn_hidden_dim = int(gcn_hidden_dim)
        self.lstm_hidden_dim = int(lstm_hidden_dim)
        self.encoder_gcn_layers = int(encoder_gcn_layers)
        self.decoder_gcn_layers = int(decoder_gcn_layers)
        self.encoder_lstm_layers = encoder_lstm
        self.decoder_lstm_layers = decoder_lstm
        self.lstm_layers = encoder_lstm
        self.dropout_p = float(dropout)
        self.output_activation = "linear"
        self.label = MODEL_LABEL
        a_hat = symmetric_normalized_adjacency(adjacency, expected_nodes=self.num_nodes)
        if not np.isfinite(a_hat).all():
            raise ReimplementationError("normalized adjacency contains NaN or inf")
        buffer = torch.from_numpy(np.ascontiguousarray(a_hat, dtype=np.float32))
        if not torch.isfinite(buffer).all():
            raise ReimplementationError("float32 Â contains NaN or inf")
        self.register_buffer("a_hat", buffer)
        self.encoder_gcn = GraphConvolutionStack(
            self.input_channels, self.gcn_hidden_dim, self.encoder_gcn_layers
        )
        self.decoder_gcn = GraphConvolutionStack(
            self.lstm_hidden_dim, self.gcn_hidden_dim, self.decoder_gcn_layers
        )
        self.encoder_lstm = nn.LSTM(
            input_size=self.gcn_hidden_dim,
            hidden_size=self.lstm_hidden_dim,
            num_layers=self.encoder_lstm_layers,
            batch_first=True,
        )
        self.decoder_lstm = nn.LSTM(
            input_size=self.gcn_hidden_dim,
            hidden_size=self.lstm_hidden_dim,
            num_layers=self.decoder_lstm_layers,
            batch_first=True,
        )
        self.dropout = nn.Dropout(self.dropout_p)
        self.fc = nn.Linear(self.lstm_hidden_dim, self.output_channels)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    @classmethod
    def from_config(cls, config: Mapping[str, Any], adjacency: np.ndarray) -> "S2SGCNLSTM":
        return cls(
            adjacency,
            num_nodes=int(config["num_nodes"]),
            input_channels=int(config["input_channels"]),
            output_channels=int(config["output_channels"]),
            input_window=int(config.get("n_his", config.get("input_window", 12))),
            output_window=int(config.get("output_steps", config.get("output_window", 1))),
            gcn_hidden_dim=int(config.get("gcn_hidden_dim", 64)),
            lstm_hidden_dim=int(config.get("lstm_hidden_dim", 64)),
            encoder_gcn_layers=int(config.get("encoder_gcn_layers", 1)),
            decoder_gcn_layers=int(config.get("decoder_gcn_layers", 1)),
            lstm_layers=int(config.get("lstm_layers", 1)),
            encoder_lstm_layers=int(config["encoder_lstm_layers"]) if "encoder_lstm_layers" in config else None,
            decoder_lstm_layers=int(config["decoder_lstm_layers"]) if "decoder_lstm_layers" in config else None,
            dropout=float(config.get("dropout", 0.2)),
        )

    def parameter_count(self) -> int:
        return int(sum(param.numel() for param in self.parameters() if param.requires_grad))

    def forward(self, x: torch.Tensor, return_trace: bool = False) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
        if not torch.is_tensor(x) or x.ndim != 4:
            raise ReimplementationError(f"input must be [B, T, N, C], got {None if not torch.is_tensor(x) else tuple(x.shape)}")
        batch, steps, nodes, channels = (int(x.shape[0]), int(x.shape[1]), int(x.shape[2]), int(x.shape[3]))
        if steps != self.input_window or nodes != self.num_nodes or channels != self.input_channels:
            raise ReimplementationError(
                f"input {tuple(x.shape)} != [B, {self.input_window}, {self.num_nodes}, {self.input_channels}]"
            )
        encoded: list[torch.Tensor] = []
        for step in range(steps):
            encoded.append(self.encoder_gcn(x[:, step], self.a_hat))
        spatial = torch.stack(encoded, dim=1)
        sequence = spatial.permute(0, 2, 1, 3).reshape(batch * nodes, steps, self.gcn_hidden_dim)
        _output, (encoder_h, _encoder_c) = self.encoder_lstm(sequence)
        context = encoder_h[-1].reshape(batch, nodes, self.lstm_hidden_dim)
        # Fixed context: the decoder GCN reads E once. It does not read predictions.
        decoded = self.decoder_gcn(context, self.a_hat)
        decoder_gcn_calls = 1
        step_input = decoded.reshape(batch * nodes, 1, self.gcn_hidden_dim)
        # Assumption: decoder state starts at zero and is not copied from the encoder.
        # With one decoder step, weight_hh is multiplied by that zero state.
        state_h = step_input.new_zeros(self.decoder_lstm_layers, batch * nodes, self.lstm_hidden_dim)
        state_c = step_input.new_zeros(self.decoder_lstm_layers, batch * nodes, self.lstm_hidden_dim)
        frames: list[torch.Tensor] = []
        for _ in range(self.output_window):
            lstm_out, (state_h, state_c) = self.decoder_lstm(step_input, (state_h, state_c))
            hidden = self.dropout(lstm_out[:, 0, :])
            frames.append(self.fc(hidden).reshape(batch, nodes, self.output_channels))
        prediction = torch.stack(frames, dim=1)
        if not return_trace:
            return prediction
        trace = {
            "label": self.label,
            "input_shape": [batch, steps, nodes, channels],
            "encoder_gcn_layers": self.encoder_gcn_layers,
            "decoder_gcn_layers": self.decoder_gcn_layers,
            "encoder_lstm_layers": self.encoder_lstm_layers,
            "decoder_lstm_layers": self.decoder_lstm_layers,
            "encoder_gcn_steps": len(encoded),
            "encoder_spatial_shape": list(spatial.shape),
            "encoder_context_shape": list(context.shape),
            "decoder_gcn_calls": decoder_gcn_calls,
            "decoder_spatial_shape": list(decoded.shape),
            "decoder_lstm_steps": len(frames),
            "decoder_initial_state": "zeros",
            "prediction_feedback": False,
            "teacher_forcing": False,
            "output_activation": self.output_activation,
            "head_shape": [batch, nodes, self.output_channels],
            "output_shape": list(prediction.shape),
        }
        return prediction, trace
