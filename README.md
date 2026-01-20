# Mechanistic Interpretability for Steering Vision‑Language‑Action Models
CoRL 2025  
[Paper (arXiv:2509.00328)](https://arxiv.org/pdf/2509.00328) · [Project Site](https://vla-mech-interp.github.io)

We introduce an activation‑level interface for controlling Vision‑Language‑Action (VLA) policies without fine‑tuning, reward signals, or environment interaction. By interpreting FFN value vectors as vocabulary‑aligned directions, we identify causal semantic mechanisms (e.g., *slow*, *up*, *careful*) and steer real‑time robot behavior across OpenVLA (LIBERO) and π₀‑FAST.

This repository provides:
- **Value Vector Instrumentation** for OpenVLA and π₀
- **Intervention Infrastructure for Simulated Environments** to steer VLA behavior and evalaute results.

## VLA Steering Pipelines
| VLA | Function | Methodological role | Entry point |
| --- | --- | --- | --- |
| **OpenVLA** | **FFN value vectors → token projections** | Derive value vectors and map them into the vocabulary basis for semantic inspection. | `openvla/ffn_value_vectors/scripts/run_ffn_value_projection.py` |
| **OpenVLA** | **Activation steering in LIBERO** | Inject clustered, concept‑aligned vectors during LIBERO evaluation to test causal effects. | `openvla/libero_experiments/scripts/run_eval.py` |
| **π₀** | **FFN value vectors → token projections** | Derive value vectors for π₀/π₀‑FAST and decode their top tokens. | `openpi/ffn_value_vectors/scripts/run_ffn_value_projection.py` |
| **π₀** | **Activation steering with LeRobot** | Apply concept‑aligned interventions during π₀ evaluation runs. | `openpi/experiments/scripts/run_eval.py` |

## Recommended Workflow (Value Vectors → Clusters → Experiments)
1) **Characterize a model’s FFN value vectors** to generate a `top_tokens.txt` file:
   - OpenVLA: `openvla/ffn_value_vectors/...`
   - OpenPI: `openpi/ffn_value_vectors/...`
2) **Construct concept‑aligned clusters** by selecting neuron indices from the top‑token outputs.
3) **Register intervention clusters** in the appropriate dictionary:
   - OpenVLA: `openvla/libero_experiments/configs/interventions/dictionaries.yaml`
   - OpenPI: `openpi/experiments/configs/interventions/dictionaries.yaml`
4) **Run evaluations** to quantify intervention effects in simulation.

## Experiment Artifacts
- **OpenVLA / LIBERO**
  - Videos per episode (if enabled).
  - Full action sequence logs (JSON).
  - Task success rates (CSV) for `libero_10`.
- **OpenPI**
  - Plot contrasting **baseline vs intervention** trajectories and their difference.
  - Per‑run logs + action outputs.

## Setup
Create the environment that matches the target model:
```bash
# OpenVLA (LIBERO)
conda env create -f setup/openvla/environment.openvla.yml
conda activate openvla-interp
pip install -e .
./setup/openvla/setup.sh

# OpenPI (π₀)
conda env create -f setup/openpi/environment.openpi.yml
conda activate openpi-interp
pip install -e .
```

Notes:
- `flash-attn==2.5.5` is included for OpenVLA; if it is unavailable, OpenVLA automatically falls back to `sdpa`.
- OpenPI uses JAX/Flax; CPU fallback works out of the box. Use `JAX_PLATFORM_NAME=cpu` to avoid GPU for quick tests.
- Model checkpoints are downloaded and cached under `~/.cache/openpi` or `~/.cache/huggingface` as needed.
- `./setup/openvla/setup.sh` writes a `LIBERO_CONFIG_PATH` config under `utils/libero_config/` and downloads LIBERO assets to `utils/libero_assets/` plus `libero_10` datasets to `utils/libero_datasets/`. Set `LIBERO_CONFIG_PATH=utils/libero_config` when running OpenVLA LIBERO evaluations.

## How to Run: OpenVLA

### (1) Value Vectors
```bash
cd openvla
python ffn_value_vectors/scripts/run_ffn_value_projection.py \
  --model_name openvla/openvla-7b-finetuned-libero-10 \
  --output_dir ffn_value_vectors/artifacts/ffn_value_projection \
  --batch_size 2048 \
  --top_k 10
```

Key parameters:
- `--top_k`: number of top tokens per vector.
- `--batch_size`: projection batch size (GPU memory bound).
- `--no_action_tokens`: disable action‑token decoding.

Artifacts:
- `value_vectors.pkl`, `top_tokens_output.txt` under the output directory.

### (2) Experiments
```bash
cd openvla
LIBERO_CONFIG_PATH=../utils/libero_config \
python libero_experiments/scripts/run_eval.py \
  --config libero_experiments/configs/runs/example_run.yaml
```

Key parameters:
- `--config`: run spec (task suite, trials, logging).
- `--interventions`: intervention YAML (defaults to `libero_experiments/configs/interventions/dictionaries.yaml`).

Artifacts:
- `libero_experiments/logs/...` (logs, videos/actions, success CSV).

## How to Run: π₀

### (1) Value Vectors
```bash
cd openpi
JAX_PLATFORM_NAME=cpu python ffn_value_vectors/scripts/run_ffn_value_projection.py \
  --output_dir ffn_value_vectors/artifacts/ffn_value_projection \
  --batch_size 5000 \
  --top_k 10
```

Key parameters:
- `--max_batches 1` for a quick sanity‑check run.
- `--save_logits` / `--save_value_vectors` for intermediate artifacts.

Artifacts:
- `top_tokens_batches/top_tokens.txt` in the output directory.

### (2) Experiments
```bash
cd openpi
JAX_PLATFORM_NAME=cpu python experiments/scripts/run_eval.py \
  --config openpi/experiments/configs/runs/example_run.yaml
```

Key parameters:
- `--config`: run spec (dataset, episode, logging).
- `--interventions`: intervention YAML (defaults to `openpi/experiments/configs/interventions/dictionaries.yaml`).

Artifacts:
- `openpi/experiments/logs/...` (logs, action JSONL, plot images).

## License & Attribution
This repo integrates OpenVLA and π₀ workflows for mechanistic interpretability research. Please cite original model papers and repositories when publishing results:
- OpenVLA codebase: https://github.com/openvla/openvla
- OpenVLA paper: https://openvla.github.io
- π₀ (OpenPI) codebase: https://github.com/Physical-Intelligence/openpi
- π₀ paper: https://www.pi.website/blog/pi0
