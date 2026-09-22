from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd


ABLATIONS: dict[str, tuple[str, ...]] = {
    "sar_only": (),
    "sar_ndwi": ("ndwi",),
    "sar_ndwi_mndwi": ("ndwi", "mndwi"),
}


def parse_ablations(values: Iterable[str]) -> list[tuple[str, tuple[str, ...]]]:
    requested = [value.strip().replace("-", "_") for value in values if value.strip()]
    if not requested:
        raise ValueError("At least one ablation is required")
    unknown = sorted(set(requested) - set(ABLATIONS))
    if unknown:
        choices = ", ".join(ABLATIONS)
        raise ValueError(f"Unknown ablation(s): {unknown}. Choose from: {choices}")
    return [(name, ABLATIONS[name]) for name in requested]


def validation_events(data_root: Path, manifest_path: Path) -> list[str]:
    pairs = pd.read_csv(data_root / "pairs.csv")
    required = {"event_id", "event_kind"}
    missing = required - set(pairs.columns)
    if missing:
        raise ValueError(f"pairs.csv is missing columns: {sorted(missing)}")
    events = sorted(pairs.loc[pairs["event_kind"] != "baseline", "event_id"].unique())
    if not events:
        raise ValueError("No non-baseline events are available for validation")
    manifest = pd.read_csv(manifest_path, usecols=["event_id"])
    absent = sorted(set(events) - set(manifest["event_id"]))
    if absent:
        raise ValueError(f"Manifest has no patches for event(s): {absent}")
    return events


def fold_score_pair_ids(data_root: Path, validation_event: str) -> list[str]:
    pairs = pd.read_csv(data_root / "pairs.csv")
    required = {"pair_id", "event_id", "event_kind"}
    missing = required - set(pairs.columns)
    if missing:
        raise ValueError(f"pairs.csv is missing columns: {sorted(missing)}")
    selected = pairs[
        (pairs["event_id"] == validation_event) | (pairs["event_kind"] == "baseline")
    ]
    if not (selected["event_id"] == validation_event).any():
        raise ValueError(f"Unknown validation event: {validation_event}")
    return selected["pair_id"].tolist()


def build_experiment_plan(
    data_root: Path,
    manifest_path: Path,
    ablations: Iterable[str],
) -> pd.DataFrame:
    rows = []
    for name, features in parse_ablations(ablations):
        for event_id in validation_events(data_root, manifest_path):
            rows.append({
                "ablation": name,
                "validation_event": event_id,
                "optical_features": ",".join(features),
            })
    return pd.DataFrame(rows)
