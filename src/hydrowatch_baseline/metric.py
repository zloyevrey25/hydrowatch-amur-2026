from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ScoreBreakdown:
    score: float
    q_flood: float
    q_water_peak: float
    q_water_pre: float
    spec_base: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def area_convergence(predicted: float, reference: float, threshold: float) -> float:
    denominator = max(float(reference), float(threshold))
    return max(0.0, 1.0 - abs(float(predicted) - float(reference)) / denominator)


def _read_reference_stats(data_root: Path, pair_id: str) -> dict:
    path = data_root / "reference_masks" / f"reference_{pair_id}.json"
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)["stats"]


def score_submission(submission_path: str | Path, data_root: str | Path) -> ScoreBreakdown:
    data_root = Path(data_root)
    pairs = pd.read_csv(data_root / "pairs.csv")
    submission = pd.read_csv(submission_path)
    required = {"pair_id", "flood_ha", "water_pre_ha", "water_peak_ha"}
    missing = required - set(submission.columns)
    if missing:
        raise ValueError(f"Submission is missing columns: {sorted(missing)}")
    if submission["pair_id"].duplicated().any():
        raise ValueError("Submission contains duplicate pair_id values")

    table = pairs.merge(submission, on="pair_id", how="left", validate="one_to_one")
    if table[list(required - {"pair_id"})].isna().any().any():
        raise ValueError("Submission is missing pairs or contains NaN values")

    references = [_read_reference_stats(data_root, pair_id) for pair_id in table["pair_id"]]
    table["ref_flood_ha"] = [x["flood_ha"] for x in references]
    table["ref_water_pre_ha"] = [x["water_pre_ha"] for x in references]
    table["ref_water_peak_ha"] = [x["water_peak_ha"] for x in references]

    events = table[table["event_kind"] != "baseline"]
    controls = table[table["event_kind"] == "baseline"]
    if events.empty or controls.empty:
        raise ValueError("Both event and baseline pairs are required")

    q_flood = np.mean([
        area_convergence(p, r, 50.0)
        for p, r in zip(events["flood_ha"], events["ref_flood_ha"], strict=True)
    ])
    q_water_peak = np.mean([
        area_convergence(p, r, 200.0)
        for p, r in zip(events["water_peak_ha"], events["ref_water_peak_ha"], strict=True)
    ])
    q_water_pre = np.mean([
        area_convergence(p, r, 200.0)
        for p, r in zip(events["water_pre_ha"], events["ref_water_pre_ha"], strict=True)
    ])

    control_scores = []
    for row in controls.itertuples(index=False):
        excess_share = max(0.0, row.flood_ha - row.ref_flood_ha) / (row.aoi_km2 * 100.0)
        control_scores.append(1.0 - min(1.0, excess_share / 0.005))
    spec_base = float(np.mean(control_scores))
    score = 0.45 * q_flood + 0.25 * q_water_peak + 0.15 * q_water_pre + 0.15 * spec_base
    return ScoreBreakdown(float(score), float(q_flood), float(q_water_peak), float(q_water_pre), spec_base)

