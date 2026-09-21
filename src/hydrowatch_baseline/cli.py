from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config, resolve_project_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HydroWatch STURM-Flood baseline")
    subcommands = parser.add_subparsers(dest="command", required=True)

    run = subcommands.add_parser("run", help="Run inference and create submission.csv")
    run.add_argument("--data-root", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, default=Path("outputs/sturm_baseline"))
    run.add_argument("--config", type=Path, default=Path("configs/sturm_baseline.toml"))
    run.add_argument("--trained-weights", type=Path)

    score = subcommands.add_parser("score", help="Calculate the official local score")
    score.add_argument("--data-root", type=Path, required=True)
    score.add_argument("--submission", type=Path, required=True)

    prepare = subcommands.add_parser("prepare", help="Audit data and build event-safe patch manifests")
    prepare.add_argument("--data-root", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, default=Path("outputs/preparation"))
    prepare.add_argument("--patch-size", type=int, default=128)
    prepare.add_argument("--stride", type=int, default=128)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "score":
        from .metric import score_submission

        print(json.dumps(score_submission(args.submission, args.data_root).as_dict(), indent=2))
        return

    if args.command == "prepare":
        from .dataset import write_preparation_manifests

        outputs = write_preparation_manifests(
            args.data_root, args.output_dir, args.patch_size, args.stride
        )
        print(json.dumps({name: str(path) for name, path in outputs.items()}, indent=2))
        return

    config = load_config(args.config)
    project_root = Path.cwd()
    sturm = config["sturm"]
    repository = resolve_project_path(sturm["repository"], project_root)
    weights = resolve_project_path(sturm["weights"], project_root)
    from .sturm import load_multimodal_model
    from .pipeline import run_dataset

    model = load_multimodal_model(repository, weights, sturm["patch_size"])
    if args.trained_weights is not None:
        model.load_weights(args.trained_weights)
    submission = run_dataset(args.data_root, args.output_dir, model, config)
    print(f"Created {args.output_dir / 'submission.csv'} with {len(submission)} pairs")


if __name__ == "__main__":
    main()
