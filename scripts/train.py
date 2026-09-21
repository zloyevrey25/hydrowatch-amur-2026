from __future__ import annotations

from pathlib import Path
import argparse

from hydrowatch_baseline.config import load_config, resolve_project_path
from hydrowatch_baseline.training import train_multimodal


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune the eight-channel HydroWatch model")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("outputs/preparation/patch_manifest.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/training"))
    parser.add_argument("--config", type=Path, default=Path("configs/sturm_baseline.toml"))
    parser.add_argument("--validation-event", required=True)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--finetune-epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = load_config(args.config)
    project_root = Path.cwd()
    repository = resolve_project_path(config["sturm"]["repository"], project_root)
    weights = resolve_project_path(config["sturm"]["weights"], project_root)
    final_weights = train_multimodal(
        args.data_root,
        args.manifest,
        args.output_dir,
        repository,
        weights,
        config,
        args.validation_event,
        args.warmup_epochs,
        args.finetune_epochs,
        args.batch_size,
        args.seed,
    )
    print(f"Saved final weights to {final_weights}")


if __name__ == "__main__":
    main()
