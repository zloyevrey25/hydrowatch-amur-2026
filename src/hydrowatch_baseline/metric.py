from __future__ import annotations

from dataclasses import dataclass, asdict
from collections.abc import Iterable
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


def _load_scoring_table(
    submission_path: str | Path,
    data_root: str | Path,
    pair_ids: Iterable[str] | None = None,
) -> pd.DataFrame:
    data_root = Path(data_root)
    pairs = pd.read_csv(data_root / "pairs.csv")
    if pair_ids is not None:
        requested = set(pair_ids)
        unknown = requested - set(pairs["pair_id"])
        if unknown:
            raise ValueError(f"Requested unknown pair_id values: {sorted(unknown)}")
        pairs = pairs[pairs["pair_id"].isin(requested)].copy()
    submission = pd.read_csv(submission_path)
    required = {"pair_id", "flood_ha", "water_pre_ha", "water_peak_ha"}
    missing = required - set(submission.columns)
    if missing:
        raise ValueError(f"Submission is missing columns: {sorted(missing)}")
    if submission["pair_id"].duplicated().any():
        raise ValueError("Submission contains duplicate pair_id values")
    unexpected = set(submission["pair_id"]) - set(pairs["pair_id"])
    if unexpected:
        raise ValueError(f"Submission contains unknown pair_id values: {sorted(unexpected)}")

    table = pairs.merge(submission, on="pair_id", how="left", validate="one_to_one")
    if table[list(required - {"pair_id"})].isna().any().any():
        raise ValueError("Submission is missing pairs or contains NaN values")
    if (table[["flood_ha", "water_pre_ha", "water_peak_ha"]] < 0).any().any():
        raise ValueError("Submission area values must be non-negative")

    references = [_read_reference_stats(data_root, pair_id) for pair_id in table["pair_id"]]
    table["ref_flood_ha"] = [x["flood_ha"] for x in references]
    table["ref_water_pre_ha"] = [x["water_pre_ha"] for x in references]
    table["ref_water_peak_ha"] = [x["water_peak_ha"] for x in references]
    return table


def _score_table(table: pd.DataFrame) -> ScoreBreakdown:
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


def score_submission(
    submission_path: str | Path,
    data_root: str | Path,
    pair_ids: Iterable[str] | None = None,
) -> ScoreBreakdown:
    table = _load_scoring_table(submission_path, data_root, pair_ids)
    return _score_table(table)


def describe_submission(
    submission_path: str | Path,
    data_root: str | Path,
    pair_ids: Iterable[str] | None = None,
) -> tuple[ScoreBreakdown, pd.DataFrame]:
    """Return official score components and per-pair area diagnostics."""
    table = _load_scoring_table(submission_path, data_root, pair_ids)
    score = _score_table(table)
    report = table[["pair_id", "event_id", "event_kind", "aoi_id"]].copy()
    for channel, threshold in (
        ("flood", 50.0),
        ("water_pre", 200.0),
        ("water_peak", 200.0),
    ):
        predicted = table[f"{channel}_ha"].astype(float)
        reference = table[f"ref_{channel}_ha"].astype(float)
        report[f"{channel}_ha"] = predicted
        report[f"ref_{channel}_ha"] = reference
        report[f"{channel}_error_ha"] = predicted - reference
        report[f"{channel}_abs_error_ha"] = (predicted - reference).abs()
        report[f"{channel}_quality"] = [
            area_convergence(p, r, threshold) for p, r in zip(predicted, reference, strict=True)
        ]

    baseline = table["event_kind"] == "baseline"
    report["control_excess_flood_ha"] = np.where(
        baseline,
        np.maximum(0.0, table["flood_ha"] - table["ref_flood_ha"]),
        np.nan,
    )
    report["control_spec_base"] = np.where(
        baseline,
        [
            1.0 - min(
                1.0,
                max(0.0, row.flood_ha - row.ref_flood_ha) / (row.aoi_km2 * 100.0) / 0.005,
            )
            for row in table.itertuples(index=False)
        ],
        np.nan,
    )
    return score, report


def write_submission_report(
    submission_path: str | Path,
    data_root: str | Path,
    output_dir: str | Path,
    pair_ids: Iterable[str] | None = None,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    score, details = describe_submission(submission_path, data_root, pair_ids)
    summary_path = output_dir / "score_summary.json"
    details_path = output_dir / "pair_diagnostics.csv"
    summary_path.write_text(
        json.dumps(score.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    details.to_csv(details_path, index=False)
    return {"summary": summary_path, "details": details_path}

