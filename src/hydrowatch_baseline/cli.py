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
    run.add_argument("--trained-weights", type=Path, required=True)

    score = subcommands.add_parser("score", help="Calculate the official local score")
    score.add_argument("--data-root", type=Path, required=True)
    score.add_argument("--submission", type=Path, required=True)

    report = subcommands.add_parser("report", help="Write score summary and per-pair diagnostics")
    report.add_argument("--data-root", type=Path, required=True)
    report.add_argument("--submission", type=Path, required=True)
    report.add_argument("--output-dir", type=Path, default=Path("outputs/report"))

    prepare = subcommands.add_parser("prepare", help="Audit data and build event-safe patch manifests")
    prepare.add_argument("--data-root", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, default=Path("outputs/preparation"))
    prepare.add_argument("--patch-size", type=int, default=128)
    prepare.add_argument("--stride", type=int, default=128)

    validate = subcommands.add_parser(
        "validate-package", help="Validate submission.csv and all mandatory flood masks"
    )
    validate.add_argument("--data-root", type=Path, required=True)
    validate.add_argument("--package-dir", type=Path, required=True)
    validate.add_argument("--report", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "score":
        from .metric import score_submission

        print(json.dumps(score_submission(args.submission, args.data_root).as_dict(), indent=2))
        return

    if args.command == "report":
        from .metric import write_submission_report

        outputs = write_submission_report(args.submission, args.data_root, args.output_dir)
        print(json.dumps({name: str(path) for name, path in outputs.items()}, indent=2))
        return

    if args.command == "prepare":
        from .dataset import write_preparation_manifests

        outputs = write_preparation_manifests(
            args.data_root, args.output_dir, args.patch_size, args.stride
        )
        print(json.dumps({name: str(path) for name, path in outputs.items()}, indent=2))
        return

    if args.command == "validate-package":
        from .package import validate_submission_package, write_package_validation

        result = validate_submission_package(args.data_root, args.package_dir)
        if args.report is not None:
            write_package_validation(result, args.report)
        print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2))
        if not result.valid:
            raise SystemExit(1)
        return

    config = load_config(args.config)
    project_root = Path.cwd()
    sturm = config["sturm"]
    repository = resolve_project_path(sturm["repository"], project_root)
    from .sturm import build_multimodal_model
    from .pipeline import run_dataset

    model = build_multimodal_model(repository, sturm["patch_size"])
    model.load_weights(args.trained_weights)
    submission = run_dataset(args.data_root, args.output_dir, model, config)
    print(f"Created {args.output_dir / 'submission.csv'} with {len(submission)} pairs")


if __name__ == "__main__":
    main()
