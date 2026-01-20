"""CLI for Paligemma FFN value vector extraction and token projection."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from openpi_ffn_value_vectors.pi0_extract import build_config_from_args, run  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_name", default="pi0_fast_droid")
    parser.add_argument(
        "--output_dir",
        default="openpi/ffn_value_vectors/artifacts/ffn_value_projection",
    )
    parser.add_argument("--batch_size", type=int, default=5000)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--save_logits", action="store_true")
    parser.add_argument("--save_value_vectors", action="store_true")
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--save_params", action="store_true")
    args = parser.parse_args()

    cfg = build_config_from_args(args)
    run(cfg)


if __name__ == "__main__":
    main()
