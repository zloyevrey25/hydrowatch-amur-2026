from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import json
import shutil
import subprocess
import time


PAIR_ID = "flood_2021_06_amur__poyarkovo"
PRE_PATTERN = f"{PAIR_ID}__S1_pre_2021-05-14*.tif"
PEAK_PATTERN = f"{PAIR_ID}__S1_peak_2021-07-01*.tif"


def largest_complete_file(directory: Path, pattern: str, minimum_bytes: int) -> Path | None:
    candidates = sorted(directory.glob(pattern), key=lambda path: path.stat().st_size, reverse=True)
    for candidate in candidates:
        if candidate.stat().st_size >= minimum_bytes:
            return candidate
    return None


def wait_for_export(
    directory: Path, pattern: str, minimum_bytes: int, wait_minutes: int
) -> Path:
    deadline = time.monotonic() + wait_minutes * 60
    while True:
        candidate = largest_complete_file(directory, pattern, minimum_bytes)
        if candidate is not None:
            print(f"Found {candidate.name}: {candidate.stat().st_size / 1024**2:.1f} MiB", flush=True)
            return candidate
        if time.monotonic() >= deadline:
            sizes = {path.name: path.stat().st_size for path in directory.glob(pattern)}
            raise TimeoutError(f"No complete export matching {pattern}; candidates={sizes}")
        print(f"Waiting for {pattern} in {directory}...", flush=True)
        time.sleep(30)


def run(command: list[str], *, env_prefix: bool = False) -> None:
    shown = " ".join(command)
    print(f"\n>>> {shown}", flush=True)
    if env_prefix:
        subprocess.run(["env", "PYTHONPATH=src", *command], check=True)
    else:
        subprocess.run(command, check=True)


def main() -> None:
    parser = ArgumentParser(description="Run the time-boxed HydroWatch MVP in Google Colab")
    parser.add_argument("--drive-source", type=Path, default=Path("/content/drive/MyDrive/hydrowatch_amur"))
    parser.add_argument(
        "--drive-output", type=Path,
        default=Path("/content/drive/MyDrive/hydrowatch_amur_training/mvp"),
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/hydrowatch_amur"))
    parser.add_argument("--wait-minutes", type=int, default=20)
    parser.add_argument("--minimum-export-mib", type=int, default=50)
    parser.add_argument("--train-patches", type=int, default=128)
    parser.add_argument("--validation-patches", type=int, default=128)
    args = parser.parse_args()

    minimum_bytes = args.minimum_export_mib * 1024**2
    pre = wait_for_export(args.drive_source, PRE_PATTERN, minimum_bytes, args.wait_minutes)
    peak = wait_for_export(args.drive_source, PEAK_PATTERN, minimum_bytes, args.wait_minutes)
    pair_dir = args.data_root / "rasters/flood_2021_06_amur/poyarkovo"
    pair_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pre, pair_dir / "S1_pre_2021-05-14.tif")
    shutil.copy2(peak, pair_dir / "S1_peak_2021-07-01.tif")

    preparation = Path("outputs/preparation")
    eda = Path("outputs/eda")
    training = Path("outputs/training/mvp")
    final = Path("outputs/final")
    run([
        "python", "-m", "hydrowatch_baseline.cli", "prepare",
        "--data-root", str(args.data_root), "--output-dir", str(preparation),
    ], env_prefix=True)
    summary = json.loads((preparation / "dataset_summary.json").read_text())
    if summary["ready_s1_pairs"] != 11:
        raise RuntimeError(f"Expected 11 ready Sentinel-1 pairs, got {summary['ready_s1_pairs']}")
    run([
        "python", "-m", "hydrowatch_baseline.cli", "eda",
        "--preparation-dir", str(preparation), "--output-dir", str(eda),
    ], env_prefix=True)
    run([
        "python", "scripts/train.py", "--data-root", str(args.data_root),
        "--manifest", str(preparation / "patch_manifest.csv"),
        "--validation-event", "flood_2019_07_amur", "--output-dir", str(training),
        "--warmup-epochs", "1", "--finetune-epochs", "0", "--batch-size", "2",
        "--max-train-patches", str(args.train_patches),
        "--max-validation-patches", str(args.validation_patches),
    ])
    weights = training / "final.weights.h5"
    run([
        "python", "-m", "hydrowatch_baseline.cli", "run",
        "--data-root", str(args.data_root), "--output-dir", str(final),
        "--trained-weights", str(weights),
    ], env_prefix=True)
    run([
        "python", "-m", "hydrowatch_baseline.cli", "score",
        "--data-root", str(args.data_root), "--submission", str(final / "submission.csv"),
    ], env_prefix=True)
    run([
        "python", "-m", "hydrowatch_baseline.cli", "report",
        "--data-root", str(args.data_root), "--submission", str(final / "submission.csv"),
        "--output-dir", str(final / "report"),
    ], env_prefix=True)
    run([
        "python", "-m", "hydrowatch_baseline.cli", "validate-package",
        "--data-root", str(args.data_root), "--package-dir", str(final),
        "--report", str(final / "package_validation.json"),
    ], env_prefix=True)

    args.drive_output.mkdir(parents=True, exist_ok=True)
    for name in ("final.weights.h5", "training_history.csv", "run_config.json"):
        shutil.copy2(training / name, args.drive_output / name)
    shutil.copytree(preparation, args.drive_output / "preparation", dirs_exist_ok=True)
    shutil.copytree(eda, args.drive_output / "eda", dirs_exist_ok=True)
    shutil.copytree(final, args.drive_output / "final", dirs_exist_ok=True)
    print(f"\nMVP_DONE: {args.drive_output}", flush=True)


if __name__ == "__main__":
    main()
