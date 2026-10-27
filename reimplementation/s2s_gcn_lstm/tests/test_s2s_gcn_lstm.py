"""Shape, graph, gradient, and checkpoint checks for Seq2seq GCN-LSTM."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.optim import Adam

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_root import xuancheng26_training_root
from reimplementation.common.data.r_only_npz_dataset import ROnlyNPZDataset, build_dataloader
from reimplementation.common.errors import ReimplementationError
from reimplementation.common.evaluation import evaluate_loader
from reimplementation.common.utils.checkpoint import load_checkpoint, save_checkpoint
from reimplementation.common.utils.reproducibility import seed_everything
from reimplementation.s2s_gcn_lstm.graph import symmetric_normalized_adjacency
from reimplementation.s2s_gcn_lstm.losses import MAERawLoss
from reimplementation.s2s_gcn_lstm.model.s2s_gcn_lstm import MODEL_LABEL, S2SGCNLSTM
from reimplementation.s2s_gcn_lstm.search_hyperparameters import (
    BATCH_SIZES,
    DROPOUTS,
    ENCODER_GCN_LAYERS,
    LEARNING_RATES,
    LSTM_HIDDEN_SIZES,
    sample_trial,
)
from reimplementation.s2s_gcn_lstm.validation import assert_task_config

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "r_only_s2s_gcn_lstm.json"


def _chain_adjacency() -> np.ndarray:
    return np.array(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )


def _positive_model(**overrides: object) -> S2SGCNLSTM:
    seed_everything(0)
    kwargs = {
        "num_nodes": 3,
        "input_channels": 1,
        "output_channels": 1,
        "input_window": 12,
        "output_window": 1,
        "gcn_hidden_dim": 4,
        "lstm_hidden_dim": 4,
        "encoder_gcn_layers": 1,
        "decoder_gcn_layers": 1,
        "lstm_layers": 1,
        "dropout": 0.0,
    }
    kwargs.update(overrides)
    model = S2SGCNLSTM(_chain_adjacency(), **kwargs)
    with torch.no_grad():
        for layer in list(model.encoder_gcn.layers) + list(model.decoder_gcn.layers):
            layer.weight.fill_(0.1)
        for lstm in (model.encoder_lstm, model.decoder_lstm):
            # Keep gate pre-activations positive so ReLU does not hide the GCN path.
            # Official training still uses the PyTorch LSTM initialization.
            lstm.weight_ih_l0.fill_(0.05)
            lstm.weight_hh_l0.fill_(0.05)
            lstm.bias_ih_l0.fill_(1.0)
            lstm.bias_hh_l0.fill_(1.0)
        model.fc.weight.fill_(0.1)
        model.fc.bias.zero_()
    return model


class GraphTests(unittest.TestCase):
    def test_symmetric_normalization_matches_hand_formula(self) -> None:
        adjacency = _chain_adjacency()
        normalized = symmetric_normalized_adjacency(adjacency)
        inv0 = 1.0 / np.sqrt(2.0)
        expected = np.array(
            [
                [inv0 * inv0, inv0 * inv0, 0.0],
                [inv0 * inv0, inv0 * inv0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        self.assertTrue(np.allclose(normalized, expected))
        self.assertTrue(np.isfinite(normalized).all())

    def test_isolated_node_does_not_mix_and_relu_clips(self) -> None:
        model = _positive_model()
        layer = model.encoder_gcn.layers[0]
        with torch.no_grad():
            layer.weight.fill_(1.0)
        features = torch.tensor([[[4.0], [0.0], [-3.0]]], dtype=torch.float32)
        output = layer(features, model.a_hat)
        expected = torch.tensor([[[2.0], [2.0], [0.0]]], dtype=torch.float32)
        self.assertTrue(torch.allclose(output, expected, atol=1e-5))

    def test_distance_weights_are_not_replaced_by_ones(self) -> None:
        distance = np.array(
            [
                [0.0, 4.0, 0.0],
                [4.0, 0.0, 2.0],
                [0.0, 2.0, 0.0],
            ],
            dtype=np.float64,
        )
        normalized = symmetric_normalized_adjacency(distance)
        binary = symmetric_normalized_adjacency((distance > 0.0).astype(np.float64))
        self.assertFalse(np.allclose(normalized, binary))
        degree = np.array([5.0, 7.0, 3.0])
        inv = 1.0 / np.sqrt(degree)
        self.assertAlmostEqual(float(normalized[0, 1]), inv[0] * 4.0 * inv[1])

    def test_bad_adjacency_is_rejected(self) -> None:
        with self.assertRaises(ReimplementationError):
            symmetric_normalized_adjacency(np.array([[0.0, 1.0], [0.0, 0.0]]))
        with self.assertRaises(ReimplementationError):
            symmetric_normalized_adjacency(np.array([[0.0, -1.0], [-1.0, 0.0]]))
        with self.assertRaises(ReimplementationError):
            symmetric_normalized_adjacency(np.array([[np.nan, 0.0], [0.0, 0.0]]))
        zero = symmetric_normalized_adjacency(np.zeros((2, 2)))
        self.assertTrue(np.allclose(zero, np.eye(2)))

    def test_xuancheng_graph_is_aligned_and_finite(self) -> None:
        root = xuancheng26_training_root()
        topology = root / "graph" / "stsgcn_undirected_topology.npy"
        if not topology.is_file():
            self.skipTest("Xuancheng26 graph is not on this machine")
        from reimplementation.s2s_gcn_lstm.graph import load_s2s_graph

        info = load_s2s_graph(
            topology_path=topology,
            r_graph_path=root / "graph" / "r_graph.npz",
            r_nodes_path=root / "graph" / "r_nodes.csv",
            node_mapping_path=root / "window_data" / "node_mapping.csv",
            expected_nodes=1406,
            edge_weight="binary_connectivity",
        )
        self.assertEqual(info["edge_weight"], "binary_connectivity")
        self.assertTrue(info["matches_symmetrized_r_graph"])
        self.assertTrue(np.isfinite(info["a_hat"]).all())
        self.assertTrue(np.allclose(info["a_hat"], info["a_hat"].T))
        self.assertEqual(info["undirected_neighbor_count"] * 2, info["topology_nnz"])
        gaussian = np.load(root / "graph" / "stgcn_weighted_adjacency.npy")
        self.assertFalse(np.allclose(info["adjacency"], gaussian))
        distance_info = load_s2s_graph(
            topology_path=topology,
            r_graph_path=root / "graph" / "r_graph.npz",
            r_nodes_path=root / "graph" / "r_nodes.csv",
            node_mapping_path=root / "window_data" / "node_mapping.csv",
            expected_nodes=1406,
            edge_weight="distance",
            distance_path=root / "graph" / "undirected_center_distance.npy",
        )
        self.assertEqual(distance_info["edge_weight"], "distance")
        self.assertGreater(float(distance_info["adjacency"].max()), 1.0)
        self.assertTrue(np.isfinite(distance_info["a_hat"]).all())
        self.assertEqual(len(info["node_ids"]), 1406)
        self.assertTrue(all(isinstance(edge_id, str) and edge_id for edge_id in info["node_ids"]))


class ForwardTests(unittest.TestCase):
    def test_encoder_steps_are_independent_and_decoder_runs_once(self) -> None:
        model = _positive_model()
        model.eval()
        encoder_calls = {"n": 0}
        decoder_calls = {"n": 0}
        lstm_lengths: list[int] = []
        decoder_steps: list[int] = []
        encoder_gcn = model.encoder_gcn.forward
        decoder_gcn = model.decoder_gcn.forward
        encoder_lstm = model.encoder_lstm.forward
        decoder_lstm = model.decoder_lstm.forward

        def enc_gcn(features, a_hat):
            encoder_calls["n"] += 1
            return encoder_gcn(features, a_hat)

        def dec_gcn(features, a_hat):
            decoder_calls["n"] += 1
            return decoder_gcn(features, a_hat)

        def enc_lstm(seq, hx=None):
            lstm_lengths.append(int(seq.shape[1]))
            return encoder_lstm(seq, hx)

        def dec_lstm(seq, hx=None):
            decoder_steps.append(int(seq.shape[1]))
            self.assertIsNotNone(hx)
            self.assertTrue(torch.all(hx[0] == 0))
            self.assertTrue(torch.all(hx[1] == 0))
            return decoder_lstm(seq, hx)

        model.encoder_gcn.forward = enc_gcn
        model.decoder_gcn.forward = dec_gcn
        model.encoder_lstm.forward = enc_lstm
        model.decoder_lstm.forward = dec_lstm
        features = torch.rand(2, 12, 3, 1)
        first, trace = model(features, return_trace=True)
        self.assertEqual(tuple(first.shape), (2, 1, 3, 1))
        self.assertEqual(encoder_calls["n"], 12)
        self.assertEqual(decoder_calls["n"], 1)
        self.assertEqual(lstm_lengths, [12])
        self.assertEqual(decoder_steps, [1])
        self.assertEqual(trace["encoder_gcn_steps"], 12)
        self.assertEqual(trace["decoder_lstm_steps"], 1)
        self.assertFalse(trace["prediction_feedback"])
        self.assertEqual(trace["output_activation"], "linear")

        captured: dict[str, torch.Tensor] = {}

        def capture_lstm(seq, hx=None):
            captured["seq"] = seq.detach().clone()
            return encoder_lstm(seq, hx)

        model.encoder_lstm.forward = capture_lstm
        model(features)
        base = captured["seq"].reshape(2, 3, 12, -1)
        shifted = features.clone()
        shifted[:, 0] += 3.0
        model(shifted)
        changed = captured["seq"].reshape(2, 3, 12, -1)
        self.assertFalse(torch.allclose(base[:, :, 0], changed[:, :, 0]))
        self.assertTrue(torch.allclose(base[:, :, 1:], changed[:, :, 1:]))

    def test_multi_step_decoder_reuses_one_gcn(self) -> None:
        model = _positive_model(output_window=3)
        calls = {"n": 0}
        original = model.decoder_gcn.forward

        def wrapped(features, a_hat):
            calls["n"] += 1
            return original(features, a_hat)

        model.decoder_gcn.forward = wrapped
        prediction, trace = model(torch.rand(2, 12, 3, 1), return_trace=True)
        self.assertEqual(tuple(prediction.shape), (2, 3, 3, 1))
        self.assertEqual(calls["n"], 1)
        self.assertEqual(trace["decoder_lstm_steps"], 3)
        prediction.sum().backward()
        recurrent = model.decoder_lstm.weight_hh_l0.grad
        self.assertIsNotNone(recurrent)
        self.assertGreater(float(recurrent.abs().sum()), 0.0)

    def test_parameters_are_separate_and_receive_gradients(self) -> None:
        model = _positive_model()
        encoder_ptr = {param.data_ptr() for param in model.encoder_gcn.parameters()}
        encoder_ptr.update(param.data_ptr() for param in model.encoder_lstm.parameters())
        decoder_ptr = {param.data_ptr() for param in model.decoder_gcn.parameters()}
        decoder_ptr.update(param.data_ptr() for param in model.decoder_lstm.parameters())
        decoder_ptr.update(param.data_ptr() for param in model.fc.parameters())
        self.assertTrue(encoder_ptr.isdisjoint(decoder_ptr))
        self.assertFalse(any(isinstance(module, torch.nn.Sigmoid) for module in model.modules()))
        prediction = model(torch.rand(2, 12, 3, 1))
        prediction.sum().backward()
        groups = {"encoder_gcn": 0.0, "encoder_lstm": 0.0, "decoder_gcn": 0.0, "decoder_lstm": 0.0, "fc": 0.0}
        for name, param in model.named_parameters():
            self.assertIsNotNone(param.grad)
            self.assertTrue(torch.isfinite(param.grad).all())
            prefix = name.split(".", 1)[0]
            groups[prefix] += float(param.grad.abs().sum())
        for name, value in groups.items():
            self.assertGreater(value, 0.0, name)

    def test_linear_head_is_not_squashed_to_unit_interval(self) -> None:
        model = _positive_model()
        with torch.no_grad():
            model.fc.bias.fill_(-5.0)
        model.eval()
        output = model(torch.rand(2, 12, 3, 1))
        self.assertTrue(torch.all(output < 0))
        self.assertEqual(model.label, MODEL_LABEL)
        self.assertNotIn("Official", model.label)


class DepthTests(unittest.TestCase):
    def test_each_stack_owns_distinct_layers(self) -> None:
        model = S2SGCNLSTM(
            _chain_adjacency(),
            num_nodes=3,
            gcn_hidden_dim=4,
            lstm_hidden_dim=4,
            encoder_gcn_layers=2,
            decoder_gcn_layers=3,
            encoder_lstm_layers=2,
            decoder_lstm_layers=3,
            dropout=0.0,
        )
        with torch.no_grad():
            for layer in list(model.encoder_gcn.layers) + list(model.decoder_gcn.layers):
                layer.weight.fill_(0.1)
            for lstm in (model.encoder_lstm, model.decoder_lstm):
                for name, param in lstm.named_parameters():
                    param.fill_(0.05 if "weight" in name else 1.0)
        self.assertEqual(len(model.encoder_gcn.layers), 2)
        self.assertEqual(len(model.decoder_gcn.layers), 3)
        self.assertEqual(tuple(model.encoder_gcn.layers[0].weight.shape), (1, 4))
        self.assertEqual(tuple(model.encoder_gcn.layers[1].weight.shape), (4, 4))
        self.assertNotEqual(
            model.encoder_gcn.layers[0].weight.data_ptr(),
            model.encoder_gcn.layers[1].weight.data_ptr(),
        )
        self.assertIn("weight_ih_l1", dict(model.encoder_lstm.named_parameters()))
        self.assertNotIn("weight_ih_l2", dict(model.encoder_lstm.named_parameters()))
        self.assertIn("weight_ih_l2", dict(model.decoder_lstm.named_parameters()))
        prediction, trace = model(torch.rand(2, 12, 3, 1), return_trace=True)
        self.assertEqual(tuple(prediction.shape), (2, 1, 3, 1))
        self.assertEqual(trace["encoder_lstm_layers"], 2)
        self.assertEqual(trace["decoder_lstm_layers"], 3)
        self.assertEqual(trace["decoder_gcn_calls"], 1)
        prediction.sum().backward()
        self.assertGreater(float(model.encoder_gcn.layers[1].weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(model.encoder_lstm.weight_ih_l1.grad.abs().sum()), 0.0)
        self.assertGreater(float(model.decoder_gcn.layers[2].weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(model.decoder_lstm.weight_ih_l2.grad.abs().sum()), 0.0)


class SearchSpaceTests(unittest.TestCase):
    def test_learning_rate_grid_is_half_decade(self) -> None:
        self.assertEqual(len(LEARNING_RATES), 5)
        self.assertAlmostEqual(LEARNING_RATES[0], 1e-4)
        self.assertAlmostEqual(LEARNING_RATES[1], 10 ** -3.5)
        self.assertAlmostEqual(LEARNING_RATES[-1], 1e-2)
        for value in LEARNING_RATES:
            self.assertGreaterEqual(value, 1e-4 - 1e-15)
            self.assertLessEqual(value, 1e-2 + 1e-15)

    def test_samples_stay_inside_table_ii_and_are_reproducible(self) -> None:
        first = sample_trial(42, 0)
        self.assertEqual(first, sample_trial(42, 0))
        self.assertNotEqual(first, sample_trial(42, 1))
        self.assertIn(first["encoder_gcn_layers"], ENCODER_GCN_LAYERS)
        self.assertIn(first["gcn_hidden_size"], (4, 8, 12, 16, 20, 24, 28, 32))
        self.assertIn(first["lstm_hidden_size"], LSTM_HIDDEN_SIZES)
        self.assertIn(first["dropout"], DROPOUTS)
        self.assertIn(first["batch_size"], BATCH_SIZES)
        self.assertTrue(any(abs(first["learning_rate"] - value) < 1e-15 for value in LEARNING_RATES))


class PipelineTests(unittest.TestCase):
    def test_train_save_load_and_evaluate(self) -> None:
        mean = 11.115971648068362
        std = 17.049848173652574
        rng = np.random.default_rng(0)
        y_raw = rng.random((4, 1, 3, 1)).astype(np.float32) * 20.0
        y = ((y_raw - mean) / std).astype(np.float32)
        x_raw = rng.random((4, 12, 3, 1)).astype(np.float32)
        x = ((x_raw - 0.5) / 1.2).astype(np.float32)
        start = np.arange(4, dtype=np.int64)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            npz = root / "train.npz"
            np.savez(
                npz,
                x=x,
                y=y,
                x_raw=x_raw,
                y_raw=y_raw,
                day_index=np.zeros(4, dtype=np.int64),
                window_start_slot=start,
                target_slot=start + 11,
            )
            dataset = ROnlyNPZDataset(npz, n_his=12, num_nodes=3, output_steps=1)
            loader = build_dataloader(dataset, batch_size=2, shuffle=False, seed=0)
            seed_everything(0)
            model = _positive_model()
            optimizer = Adam(model.parameters(), lr=1e-2)
            criterion = MAERawLoss(mean, std)
            model.train()
            batch = next(iter(loader))
            before = {name: param.detach().clone() for name, param in model.named_parameters()}
            loss = criterion(model(batch["x"]), batch["y_raw"])
            loss.backward()
            optimizer.step()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(any(not torch.equal(param.detach(), before[name]) for name, param in model.named_parameters()))
            payload = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": None,
                "epoch": 0,
                "best_metric": float(loss.detach()),
                "config": {"num_nodes": 3, "n_his": 12, "output_steps": 1, "input_channels": 1, "output_channels": 1},
                "penetration_rate": "p05",
                "seed": 0,
                "model_parameter_count": model.parameter_count(),
                "graph_sha256": "synthetic",
                "data_file_sha256": "synthetic",
                "normalization_sha256": "synthetic",
                "code_version": "0.1.0",
            }
            save_checkpoint(root / "best_checkpoint.pt", payload)
            restored = load_checkpoint(root / "best_checkpoint.pt")
            seed_everything(1)
            other = S2SGCNLSTM(_chain_adjacency(), num_nodes=3, gcn_hidden_dim=4, lstm_hidden_dim=4, dropout=0.0)
            self.assertFalse(torch.allclose(other.fc.weight, model.fc.weight))
            other.load_state_dict(restored["model_state_dict"])
            other.eval()
            model.eval()
            with torch.no_grad():
                left = model(batch["x"])
                right = other(batch["x"])
            self.assertTrue(torch.allclose(left, right))
            report = evaluate_loader(
                loader,
                lambda item: other(item["x"]),
                mean_y=mean,
                std_y=std,
                device=torch.device("cpu"),
                output_dir=root,
                split="test",
                expected_count=4,
                n_his=12,
            )
            self.assertEqual(report["sample_count"], 4)
            self.assertTrue(np.isfinite(report["metrics"]["mae"]))
            self.assertTrue((root / "test_predictions.npz").is_file())

    def test_official_config_stays_on_the_reconstruction_task(self) -> None:
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        assert_task_config(config)
        self.assertEqual(config["model_label"], MODEL_LABEL)
        self.assertFalse(config["official_implementation"])
        self.assertEqual(config["edge_weight"], "gaussian")
        self.assertTrue(str(config["distance_matrix"]).endswith("stgcn_weighted_adjacency.npy"))
        self.assertEqual(config["input_channels"], 1)
        self.assertEqual(config["rates"][:2], ["p05", "p10"])
        disk = xuancheng26_training_root() / "config" / "s2s_gcn_lstm" / "r_only_s2s_gcn_lstm.json"
        if disk.is_file():
            self.assertEqual(json.loads(disk.read_text(encoding="utf-8"))["model_name"], "s2s_gcn_lstm")


if __name__ == "__main__":
    unittest.main()
