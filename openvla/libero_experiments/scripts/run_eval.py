"""CLI entrypoint for a single LIBERO eval run."""

import argparse

from libero_experiments.config import load_config, parse_overrides
from libero_experiments.eval_libero import eval_libero


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to run config YAML")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override config values (dot notation), e.g. env.num_trials_per_task=5",
    )
    parser.add_argument(
        "--interventions",
        default="libero_experiments/configs/interventions/dictionaries.yaml",
        help="Path to intervention dictionaries YAML",
    )
    args = parser.parse_args()

    overrides = parse_overrides(args.override)
    cfg = load_config(args.config, overrides)

    eval_libero(cfg, args.interventions)


if __name__ == "__main__":
    main()
