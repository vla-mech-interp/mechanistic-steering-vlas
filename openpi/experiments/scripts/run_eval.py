import os
import argparse
import logging
from pathlib import Path

import datetime as _datetime

# Python 3.10 compatibility for openpi's datetime.UTC usage.
if not hasattr(_datetime, "UTC"):
    _datetime.UTC = _datetime.timezone.utc

import sys as _sys
import types as _types

import jax as _jax
import inspect as _inspect

# Older JAX builds may not expose api_util.debug_info, which Flax expects.
if not hasattr(_jax.api_util, "debug_info"):
    def _debug_info(*_args, **_kwargs):
        return None
    _jax.api_util.debug_info = _debug_info

# Older JAX builds may not accept `debug_info` in lu.wrap_init.
def _patch_wrap_init(module) -> None:
    if not hasattr(module, "wrap_init"):
        return
    _wrap_init_sig = _inspect.signature(module.wrap_init)
    if "debug_info" in _wrap_init_sig.parameters:
        return
    _orig_wrap_init = module.wrap_init

    def _wrap_init(fun, *args, **kwargs):
        kwargs.pop("debug_info", None)
        return _orig_wrap_init(fun, *args, **kwargs)

    module.wrap_init = _wrap_init


_patch_wrap_init(getattr(_jax, "lu", None))
_patch_wrap_init(getattr(_jax, "linear_util", None))
try:
    import jax._src.linear_util as _linear_util
except Exception:  # pragma: no cover - best effort compatibility
    _linear_util = None
_patch_wrap_init(_linear_util)

# Shim missing lerobot module path used by openpi training code.
if "lerobot.common.datasets.lerobot_dataset" not in _sys.modules:
    _lerobot_root = _types.ModuleType("lerobot")
    _lerobot_common = _types.ModuleType("lerobot.common")
    _lerobot_common_datasets = _types.ModuleType("lerobot.common.datasets")
    _stub = _types.ModuleType("lerobot.common.datasets.lerobot_dataset")

    class LeRobotDatasetMetadata:  # noqa: N801 - keep upstream name
        def __init__(self, *_args, **_kwargs):
            raise RuntimeError("LeRobotDatasetMetadata is unavailable in this lerobot version.")

    class LeRobotDataset:  # noqa: N801 - keep upstream name
        def __init__(self, *_args, **_kwargs):
            raise RuntimeError("LeRobotDataset is unavailable in this lerobot version.")

    _stub.LeRobotDatasetMetadata = LeRobotDatasetMetadata
    _stub.LeRobotDataset = LeRobotDataset
    _sys.modules["lerobot"] = _lerobot_root
    _sys.modules["lerobot.common"] = _lerobot_common
    _sys.modules["lerobot.common.datasets"] = _lerobot_common_datasets
    _sys.modules["lerobot.common.datasets.lerobot_dataset"] = _stub

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import yaml
from openpi.policies import policy_config as _policy_config
from openpi.policies import droid_policy as _droid_policy
from openpi.shared import download
from openpi.training import config as _config

_SCRIPT_DIR = Path(__file__).resolve().parent
_OPENPI_ROOT = _SCRIPT_DIR.parent.parent
_REPO_ROOT = _OPENPI_ROOT.parent
_sys.path.insert(0, str(_SCRIPT_DIR))
import utils as _utils


def _resolve_run_path(raw: str | Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    if path.exists():
        return path
    repo_path = _REPO_ROOT / path
    if repo_path.exists():
        return repo_path
    openpi_path = _OPENPI_ROOT / path
    if openpi_path.exists():
        return openpi_path
    return path


def _setup_logger(log_file: Path):
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging_handlers = [
        logging.StreamHandler(),
        logging.FileHandler(log_file, mode="w"),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=logging_handlers,
    )
    return logging.getLogger("openpi.experiments")


def _build_hyperactivate_mask(target_vectors, num_layers, hidden_dim):
    mask = jnp.zeros((num_layers, hidden_dim), dtype=jnp.bool)
    for target_vector in target_vectors:
        layer_idx = target_vector // hidden_dim
        neuron_idx = target_vector % hidden_dim
        mask = mask.at[layer_idx, neuron_idx].set(True)
    return mask


def _save_action_plot(action_no: np.ndarray, action_with: np.ndarray | None, diff: np.ndarray | None, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(action_no))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, action_no, marker="o", label="no_intervention")
    if action_with is not None:
        ax.plot(x, action_with, marker="o", label="with_intervention")
    if diff is not None:
        ax.bar(x, diff, alpha=0.3, label="diff")
    ax.set_xlabel("Action dimension")
    ax.set_ylabel("Value")
    ax.set_title("Action + Intervention Difference")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _write_jsonl(path: Path, row: dict):
    _utils.write_jsonl(path, row)


def _decode_instruction(raw) -> str:
    if hasattr(raw, "numpy"):
        raw = raw.numpy()
    if isinstance(raw, bytes):
        return raw.decode("utf-8")
    if isinstance(raw, np.ndarray):
        if raw.dtype.kind in {"S", "O"}:
            return str(raw.item())
        if raw.shape == ():
            return str(raw.item())
    return str(raw)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to run config YAML")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override config values (dot notation), e.g. env.show_every=10",
    )
    parser.add_argument(
        "--interventions",
        default=str(_OPENPI_ROOT / "experiments/configs/interventions/dictionaries.yaml"),
        help="Path to intervention dictionaries YAML",
    )
    args = parser.parse_args()

    overrides = _utils.parse_overrides(args.override)
    cfg_path = _resolve_run_path(args.config)
    cfg = _utils.load_config(cfg_path, overrides)
    intervention_dict = _utils.load_intervention_dict(
        cfg.get("intervention", {}).get("dict_name", "high_cluster"),
        _resolve_run_path(args.interventions),
    )

    logging_cfg = cfg.get("logging", {})
    intervention_cfg = cfg.get("intervention", {})
    env_cfg = cfg.get("env", {})
    model_cfg = cfg.get("model", {})

    run_name = logging_cfg.get("run_name") or (
        f"INTERVENTION-{intervention_cfg.get('dict_name', 'unknown')}-coef{intervention_cfg.get('coef', 1.0)}-"
        f"{_datetime.datetime.now().strftime('%Y_%m_%d-%H_%M_%S')}"
    )
    root_dir = _resolve_run_path(logging_cfg.get("root_dir", _OPENPI_ROOT / "experiments/logs"))
    run_dir = root_dir / run_name
    logger = _setup_logger(run_dir / "stdout.log")

    (run_dir / "artifacts").mkdir(parents=True, exist_ok=True)

    logger.info("Getting model")
    train_config = _config.get_config(model_cfg.get("config_name", "pi0_fast_hooked"))
    checkpoint_dir = download.maybe_download(
        f"s3://openpi-assets/checkpoints/{model_cfg.get('checkpoint_name', 'pi0_fast_droid')}"
    )
    logger.info("Creating policy")
    policy = _policy_config.create_trained_policy(train_config, checkpoint_dir)

    use_fake_data = env_cfg.get("use_fake_data", False) or env_cfg.get("dataset_name") == "fake"
    if use_fake_data:
        logger.info("Using fake dataset example")
        ds = [ {"steps": [ {"observation": _droid_policy.make_droid_example(), "language_instruction": "do something"} ]} ]
    else:
        logger.info("Loading dataset")
        ds = tfds.load(
            env_cfg.get("dataset_name", "droid_100"),
            data_dir=env_cfg.get("dataset_data_dir", "gs://gresearch/robotics/"),
            split=env_cfg.get("dataset_split", "train"),
        )

    hyperactivate_mask = None
    if intervention_cfg.get("enabled", True):
        hyperactivate_mask = _build_hyperactivate_mask(
            intervention_dict,
            intervention_cfg.get("num_layers", 18),
            intervention_cfg.get("hidden_dim", 16384),
        )

    logger.info("Running inference")
    for i, episode in enumerate(ds):
        if i < env_cfg.get("episode_index", 0):
            continue
        if i > env_cfg.get("episode_index", 0):
            break
        for step_idx, step in enumerate(episode["steps"]):
            max_steps = env_cfg.get("max_steps")
            if max_steps is not None and step_idx >= max_steps:
                break
            if step_idx % env_cfg.get("show_every", 50) != 0:
                continue

            obs = step["observation"]
            instruction = _decode_instruction(step["language_instruction"])
            if "prompt" in obs and instruction in {"", "None"}:
                instruction = _decode_instruction(obs["prompt"])
            static_example = {
                "observation/exterior_image_1_left": np.asarray(obs["observation/exterior_image_1_left"])
                if "observation/exterior_image_1_left" in obs
                else obs["exterior_image_1_left"].numpy(),
                "observation/wrist_image_left": np.asarray(obs["observation/wrist_image_left"])
                if "observation/wrist_image_left" in obs
                else obs["wrist_image_left"].numpy(),
                "observation/joint_position": np.asarray(obs["observation/joint_position"])
                if "observation/joint_position" in obs
                else obs["joint_position"].numpy(),
                "observation/gripper_position": np.asarray(obs["observation/gripper_position"])
                if "observation/gripper_position" in obs
                else obs["gripper_position"].numpy(),
                "prompt": instruction,
            }

            result_no = policy.infer(static_example)
            action_no = result_no["actions"][0]
            logger.info("No intervention: Step %s -> Action: %s", step_idx, action_no)

            action_with = None
            diff = None
            if intervention_cfg.get("enabled", True):
                result_with = policy.infer(
                    static_example,
                    hyperactivate_mask=hyperactivate_mask,
                    hyperactivate_coeff=intervention_cfg.get("coef", 1.0),
                )
                action_with = result_with["actions"][0]
                diff = action_with - action_no
                logger.info("With intervention: Step %s -> Action: %s", step_idx, action_with)
                logger.info("Intervention difference: %s", diff)

            if logging_cfg.get("save_actions", True):
                _write_jsonl(
                    run_dir / "artifacts" / "actions.jsonl",
                    {
                        "step": int(step_idx),
                        "prompt": instruction,
                        "action_no_intervention": np.asarray(action_no).tolist(),
                        "action_with_intervention": None if action_with is None else np.asarray(action_with).tolist(),
                        "diff": None if diff is None else np.asarray(diff).tolist(),
                    },
                )

            if logging_cfg.get("save_images", True):
                _save_action_plot(
                    np.asarray(action_no),
                    None if action_with is None else np.asarray(action_with),
                    None if diff is None else np.asarray(diff),
                    run_dir / "artifacts" / "images" / f"step_{step_idx:04d}.png",
                )

            break


if __name__ == "__main__":
    main()