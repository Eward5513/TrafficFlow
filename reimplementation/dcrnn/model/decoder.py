"""Stacked DCGRU decoder with GO symbol and original rnn_decoder loop.

Original ``legacy_seq2seq.rnn_decoder``:

* first input is decoder_inputs[0] = GO (loop_function is skipped because prev is None)
* later inputs are replaced by loop_function(prev, i)
* it unrolls len(decoder_inputs) = horizon + 1 steps
* DCRNNModel then keeps outputs[:-1], i.e. horizon predictions

With horizon=1 the kept prediction is always the GO step. Scheduled sampling
only affects the discarded second step.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from reimplementation.dcrnn.model.dcgru_cell import DCGRUCell


def sampling_threshold(global_step: int | torch.Tensor, cl_decay_steps: float) -> torch.Tensor:
    """Original ``k / (k + exp(global_step / k))``."""
    step = torch.as_tensor(global_step, dtype=torch.float32)
    decay = float(cl_decay_steps)
    return decay / (decay + torch.exp(step / decay))


class DCRNNDecoder(nn.Module):
    def __init__(
        self,
        cells: nn.ModuleList,
        *,
        horizon: int,
        num_nodes: int,
        output_dim: int,
        use_curriculum_learning: bool,
        cl_decay_steps: int,
    ) -> None:
        super().__init__()
        if len(cells) == 0:
            raise ValueError("decoder needs at least one DCGRU cell")
        self.cells = cells
        self.horizon = int(horizon)
        self.num_nodes = int(num_nodes)
        self.output_dim = int(output_dim)
        self.use_curriculum_learning = bool(use_curriculum_learning)
        self.cl_decay_steps = int(cl_decay_steps)

    def _loop_input(
        self,
        prev: torch.Tensor,
        labels_flat: torch.Tensor | None,
        step_index: int,
        *,
        is_training: bool,
        global_step: int,
    ) -> torch.Tensor:
        if not is_training:
            return prev
        if labels_flat is None:
            return prev
        teacher = labels_flat[:, step_index]
        if not self.use_curriculum_learning:
            return teacher
        threshold = sampling_threshold(global_step, self.cl_decay_steps).to(
            device=prev.device, dtype=prev.dtype
        )
        coin = torch.rand((), device=prev.device, dtype=prev.dtype)
        return teacher if bool(coin < threshold) else prev

    def forward(
        self,
        encoder_states: tuple[torch.Tensor, ...],
        *,
        supports: list[torch.Tensor],
        labels: torch.Tensor | None,
        global_step: int,
        is_training: bool,
    ) -> tuple[torch.Tensor, dict[str, object]]:
        if len(encoder_states) != len(self.cells):
            raise ValueError("encoder/decoder layer counts differ")
        batch = encoder_states[0].shape[0]
        go = encoder_states[0].new_zeros(batch, self.num_nodes * self.output_dim)
        labels_flat = None
        if labels is not None:
            if labels.shape != (batch, self.horizon, self.num_nodes, self.output_dim):
                raise ValueError(
                    f"decoder labels {tuple(labels.shape)} != "
                    f"[{batch}, {self.horizon}, {self.num_nodes}, {self.output_dim}]"
                )
            labels_flat = labels.reshape(batch, self.horizon, self.num_nodes * self.output_dim)
            # Original inserts GO at index 0, so ground-truth frame t is labels_list[t+1].
            # rnn_decoder calls loop_function(prev, i) with i starting at 1.

        states = list(encoder_states)
        outputs: list[torch.Tensor] = []
        current = go
        used_label_on_first_step = False
        scheduled_sampling_called = False
        discarded_step_ran = False
        decoder_steps = self.horizon + 1
        prev: torch.Tensor | None = None
        for step_index in range(decoder_steps):
            if prev is not None:
                scheduled_sampling_called = True
                current = self._loop_input(
                    prev,
                    labels_flat,
                    step_index - 1,
                    is_training=is_training,
                    global_step=global_step,
                )
            elif not torch.equal(current, go):
                used_label_on_first_step = True
            layer_input = current
            for layer_index, cell in enumerate(self.cells):
                if not isinstance(cell, DCGRUCell):
                    raise TypeError("decoder cells must be DCGRUCell")
                layer_input, states[layer_index] = cell(layer_input, states[layer_index], supports)
            outputs.append(layer_input)
            prev = layer_input
            if step_index == 1:
                discarded_step_ran = True

        kept = torch.stack(outputs[: self.horizon], dim=1)
        kept = kept.view(batch, self.horizon, self.num_nodes, self.output_dim)
        trace = {
            "go_shape": list(go.shape),
            "decoder_unroll_steps": decoder_steps,
            "kept_steps": self.horizon,
            "used_label_on_first_step": used_label_on_first_step,
            "scheduled_sampling_called": scheduled_sampling_called,
            "discarded_step_ran": discarded_step_ran,
            "curriculum_active": bool(is_training and self.use_curriculum_learning),
            "sampling_threshold": float(
                sampling_threshold(global_step, self.cl_decay_steps).detach().cpu()
            )
            if is_training
            else None,
            "horizon_one_ss_does_not_affect_kept_output": self.horizon == 1,
        }
        return kept, trace
