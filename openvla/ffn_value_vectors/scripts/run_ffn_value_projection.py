"""CLI for FFN value vector extraction and token projection."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from ffn_value_vectors.extract import build_config_from_args, run  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="openvla/openvla-7b-finetuned-libero-10")
    parser.add_argument("--output_dir", default="ffn_value_vectors/artifacts")
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--no_action_tokens", action="store_true")
    parser.add_argument("--save_logits", action="store_true")
    parser.add_argument("--action_bins", type=int, default=256)
    parser.add_argument("--action_min", type=float, default=-1.0)
    parser.add_argument("--action_max", type=float, default=1.0)
    args = parser.parse_args()

    cfg = build_config_from_args(args)
    run(cfg)


if __name__ == "__main__":
    main()
