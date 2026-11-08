"""Checkpoint evaluation adapters; model-specific mathematics remain in each engine."""
from __future__ import annotations
from pathlib import Path
import torch
from data_root import CODE_ROOT as PROJECT_ROOT
from reimplementation.common.data.r_only_npz_dataset import build_dataloader, loss_scale
from reimplementation.common.data.state_windows import open_split_dataset, read_state_target_scaler
from reimplementation.common.evaluation import collect_split_predictions, compute_raw_metrics
from reimplementation.dcrnn.engine import build_model as build_dcrnn
from reimplementation.dcrnn.validation import validate_dcrnn_graph
from reimplementation.graph_wavenet.engine import build_model as build_gwn
from reimplementation.graph_wavenet.validation import validate_gwn_graph
from reimplementation.s2s_gcn_lstm.engine import build_model as build_s2s
from reimplementation.s2s_gcn_lstm.validation import validate_s2s_graph
from reimplementation.stfgnn.engine import build_model as build_stfgnn
from reimplementation.stfgnn.validation import load_node_ids, validate_stfgnn_graph
from reimplementation.stgcn.engine import build_model as build_stgcn
from reimplementation.stgcn.validation import validate_graph
from reimplementation.stid.engine import assemble_batch_history, build_model as build_stid, identity_kwargs
from reimplementation.stsgcn.engine import build_model as build_stsgcn
from reimplementation.stsgcn.validation import validate_stsgcn_graph
from reimplementation.tlast.engine import build_model as build_tlast, predict_batch


def _node_ids(config: dict) -> list[str]:
    return load_node_ids(Path(config["node_mapping"]))


def build_predictor(name: str, config: dict, device: torch.device):
    nodes = _node_ids(config)
    expected = int(config["num_nodes"])
    if name == "stgcn":
        graph = validate_graph(
            Path(config["weighted_adjacency"]),
            int(config["Ks"]),
            expected,
            nodes,
            Path(config["r_nodes"]),
        )
        model = build_stgcn(config, graph["cheb_kernel"]).to(device)

        def predict(batch, _model=model):
            return _model(batch["x"])

    elif name == "dcrnn":
        graph = validate_dcrnn_graph(
            adjacency_path=Path(config["adjacency"]),
            pickle_path=Path(config["adjacency_pickle"]),
            r_nodes_path=Path(config["r_nodes"]),
            node_ids=nodes,
            expected_nodes=expected,
            filter_type=str(config.get("filter_type", "dual_random_walk")),
        )
        model = build_dcrnn(config, graph["supports"]).to(device)

        def predict(batch, _model=model):
            return _model(batch["x"], global_step=0)

    elif name == "graph_wavenet":
        graph = validate_gwn_graph(
            adjacency_path=Path(config["adjacency"]),
            pickle_path=Path(config["adjacency_pickle"]),
            r_nodes_path=Path(config["r_nodes"]),
            node_ids=nodes,
            expected_nodes=expected,
            adjtype=str(config.get("adjtype", "doubletransition")),
        )
        model = build_gwn(config, graph["supports"]).to(device)

        def predict(batch, _model=model):
            return _model(batch["x"])

    elif name == "stsgcn":
        graph = validate_stsgcn_graph(
            topology_path=Path(config["spatial_adjacency"]),
            r_nodes_path=Path(config["r_nodes"]),
            node_ids=nodes,
            expected_nodes=expected,
        )
        model = build_stsgcn(config, graph["spatial_topology"]).to(device)

        def predict(batch, _model=model):
            return _model(batch["x"])

    elif name == "stfgnn":
        graph = validate_stfgnn_graph(
            topology_path=Path(config["spatial_adjacency"]),
            r_nodes_path=Path(config["r_nodes"]),
            node_ids=nodes,
            expected_nodes=expected,
            temporal_path=Path(config["temporal_adjacency"]),
        )
        model = build_stfgnn(
            config,
            graph["spatial_topology"],
            graph["temporal_adjacency"],
            temporal_graph_is_test_only=False,
        ).to(device)

        def predict(batch, _model=model):
            return _model(batch["x"])

    elif name == "s2s_gcn_lstm":
        graph = validate_s2s_graph(config, project_root=PROJECT_ROOT)
        model = build_s2s(config, graph["adjacency"]).to(device)

        def predict(batch, _model=model):
            return _model(batch["x"])

    elif name == "stid":
        model = build_stid(config).to(device)

        def predict(batch, _model=model, _config=config):
            manifest_days = batch.pop("_all_days", None)
            day = {
                "day_of_week_source": str(_config.get("day_of_week_source")),
                "weekday_mapping": None,
                "all_days": manifest_days,
            }
            history = assemble_batch_history(batch, _config, **day).to(device)
            return _model(history, **identity_kwargs(batch, _config, day))

    elif name == "tlast":
        model = build_tlast(config).to(device)

        def predict(batch, _model=model, _config=config, _mean=None, _std=None):
            mean, std = read_state_target_scaler(Path(_config["data_root"]), _config.get("target_type", "inflow"))
            return predict_batch(_model, batch, _config, mean, std)

    elif name == "staeformer":
        from reimplementation.staeformer.engine import build_model, assemble_batch_payload, model_kwargs_from_payload, _day_args
        model = build_model(config).to(device)
        day = _day_args(config, Path(config["data_root"]))
        def predict(batch):
            payload = assemble_batch_payload(batch, config, **day)
            return model(payload["history"].to(device), **model_kwargs_from_payload(payload, device))
    elif name == "pdformer":
        from reimplementation.pdformer.engine import build_model, collect_graph_info, assemble_batch_x, _weekday_args
        graph = collect_graph_info(config, allow_test_only=False, project_root=PROJECT_ROOT)
        model = build_model(config, graph).to(device)
        day = _weekday_args(config, Path(config["data_root"]))
        def predict(batch):
            return model(assemble_batch_x(batch, config, **day).to(device))
    else:
        raise RuntimeError(f"unknown model {name}")
    return model, predict


