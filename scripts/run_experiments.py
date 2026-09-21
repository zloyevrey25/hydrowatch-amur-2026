from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import pandas as pd

from hydrowatch_baseline.config import load_config, resolve_project_path
from hydrowatch_baseline.experiments import build_experiment_plan, fold_score_pair_ids


def write_summary(rows: list[dict], output_root: Path) -> None:
    summary = pd.DataFrame(rows)
    output_root.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_root / "experiment_summary.csv", index=False)
    (output_root / "experiment_summary.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def final_validation_metrics(output_dir: Path) -> dict[str, float]:
    history_path = output_dir / "training_history.csv"
    if not history_path.exists():
        return {}
    history = pd.read_csv(history_path)
    if history.empty:
        return {}
    best = history.loc[history["val_loss"].idxmin()] if "val_loss" in history else history.iloc[-1]
    return {
        column: float(best[column])
        for column in history.columns
        if column.startswith("val_iou_") or column.startswith("val_f1_")
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train event folds and collect comparable ablation scores"
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("outputs/preparation/patch_manifest.csv"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/experiments"))
    parser.add_argument("--config", type=Path, default=Path("configs/sturm_baseline.toml"))
    parser.add_argument(
        "--ablations", default="sar-only,sar-ndwi,sar-ndwi-mndwi",
        help="Comma-separated: sar-only, sar-ndwi, sar-ndwi-mndwi",
    )
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--finetune-epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    plan = build_experiment_plan(
        args.data_root, args.manifest, args.ablations.split(",")
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    plan.to_csv(args.output_root / "experiment_plan.csv", index=False)
    if args.dry_run:
        print(plan.to_string(index=False))
        return

    from hydrowatch_baseline.metric import score_submission, write_submission_report
    from hydrowatch_baseline.pipeline import run_dataset
    from hydrowatch_baseline.sturm import load_multimodal_model
    from hydrowatch_baseline.training import train_multimodal

    base_config = load_config(args.config)
    project_root = Path.cwd()
    repository = resolve_project_path(base_config["sturm"]["repository"], project_root)
    source_weights = resolve_project_path(base_config["sturm"]["weights"], project_root)
    rows: list[dict] = []
    for experiment in plan.itertuples(index=False):
        output_dir = args.output_root / experiment.ablation / experiment.validation_event
        config = copy.deepcopy(base_config)
        config["model"]["optical_features"] = (
            [] if not experiment.optical_features else experiment.optical_features.split(",")
        )
        final_weights = output_dir / "final.weights.h5"
        record = {
            "ablation": experiment.ablation,
            "validation_event": experiment.validation_event,
            "optical_features": experiment.optical_features,
            "output_dir": str(output_dir),
        }
        try:
            if not args.resume or not final_weights.exists():
                final_weights = train_multimodal(
                    args.data_root, args.manifest, output_dir, repository, source_weights,
                    config, experiment.validation_event, args.warmup_epochs,
                    args.finetune_epochs, args.batch_size, args.seed,
                )
            model = load_multimodal_model(repository, source_weights, config["sturm"]["patch_size"])
            model.load_weights(final_weights)
            fold_pair_ids = fold_score_pair_ids(args.data_root, experiment.validation_event)
            run_dataset(args.data_root, output_dir, model, config, pair_ids=fold_pair_ids)
            score = score_submission(
                output_dir / "submission.csv", args.data_root, fold_pair_ids
            )
            report = write_submission_report(
                output_dir / "submission.csv", args.data_root, output_dir / "report", fold_pair_ids
            )
            record.update(score.as_dict())
            record.update(final_validation_metrics(output_dir))
            record["submission"] = str(output_dir / "submission.csv")
            record.update({f"report_{name}": str(path) for name, path in report.items()})
            record["status"] = "complete"
            print(f"Completed {experiment.ablation} / {experiment.validation_event}: {score.score:.4f}")
        except Exception as exc:
            record["status"] = "failed"
            record["error"] = str(exc)
            rows.append(record)
            write_summary(rows, args.output_root)
            raise
        rows.append(record)
        write_summary(rows, args.output_root)


if __name__ == "__main__":
    main()
