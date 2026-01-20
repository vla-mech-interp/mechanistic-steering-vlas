#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
export REPO_ROOT
export LIBERO_CONFIG_PATH="${REPO_ROOT}/utils/libero_config"
export ASSETS_DIR="${REPO_ROOT}/utils/libero_assets"
export DATASETS_DIR="${REPO_ROOT}/utils/libero_datasets"

mkdir -p "${LIBERO_CONFIG_PATH}" "${ASSETS_DIR}" "${DATASETS_DIR}"

conda run -n openvla-interp python -c $'import os\nfrom pathlib import Path\nimport importlib.util\nimport yaml\n\nassets_dir = Path(os.environ["ASSETS_DIR"])\ndatasets_dir = Path(os.environ["DATASETS_DIR"])\nconfig_path = Path(os.environ["LIBERO_CONFIG_PATH"]) / "config.yaml"\n\nspec = importlib.util.find_spec("libero.libero")\nif spec is None or spec.origin is None:\n    raise RuntimeError("libero package not found")\nbenchmark_root = Path(spec.origin).parent\n\nbase = {\n    "benchmark_root": str(benchmark_root),\n    "bddl_files": str(benchmark_root / "bddl_files"),\n    "init_states": str(benchmark_root / "init_files"),\n    "datasets": str(datasets_dir),\n    "assets": str(assets_dir),\n}\n\nconfig_path.parent.mkdir(parents=True, exist_ok=True)\nwith config_path.open("w", encoding="utf-8") as f:\n    yaml.safe_dump(base, f)\n\nfrom libero.libero.utils.download_utils import download_assets_from_huggingface, download_from_huggingface\n\ndownload_assets_from_huggingface(download_dir=str(assets_dir))\nlibero_10_dir = datasets_dir / "libero_10"\nif not libero_10_dir.exists():\n    download_from_huggingface(dataset_name="libero_10", download_dir=str(datasets_dir), check_overwrite=False)\nelse:\n    print(f"Dataset already present at {libero_10_dir}")\n\nprint(f"Wrote LIBERO config to {config_path}")'
