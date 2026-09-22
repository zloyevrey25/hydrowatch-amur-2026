from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def write_eda_report(preparation_dir: Path, output_dir: Path) -> dict[str, Path]:
    """Summarize dataset readiness, event areas and patch imbalance."""
    audit_path = preparation_dir / "data_audit.csv"
    manifest_path = preparation_dir / "patch_manifest.csv"
    if not audit_path.exists() or not manifest_path.exists():
        raise FileNotFoundError(
            "Run `hydrowatch-baseline prepare` before generating the EDA report"
        )

    audit = pd.read_csv(audit_path)
    patches = pd.read_csv(manifest_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    availability = (
        audit.groupby(["event_id", "event_kind"], dropna=False)
        .agg(
            pairs=("pair_id", "count"),
            ready_s1_pairs=("s1_ready", "sum"),
            ready_s2_pairs=("s2_ready", "sum"),
            min_s1_coverage=("s1_coverage_min", "min"),
        )
        .reset_index()
    )
    event_areas = (
        audit.groupby(["event_id", "event_kind"], dropna=False)[
            ["flood_ha", "water_pre_ha", "water_peak_ha", "receded_ha"]
        ]
        .sum(min_count=1)
        .reset_index()
    )

    patch_rows: list[dict] = []
    for (event_id, event_kind), group in patches.groupby(
        ["event_id", "event_kind"], dropna=False
    ):
        flood_positive = group["flood_fraction"] > 0
        water_positive = (group["water_pre_fraction"] > 0) | (
            group["water_peak_fraction"] > 0
        )
        patch_rows.append(
            {
                "event_id": event_id,
                "event_kind": event_kind,
                "patches": int(len(group)),
                "flood_positive_patches": int(flood_positive.sum()),
                "water_positive_patches": int(water_positive.sum()),
                "flood_positive_fraction": round(float(flood_positive.mean()), 6),
                "water_positive_fraction": round(float(water_positive.mean()), 6),
                "mean_flood_pixel_fraction": round(float(group["flood_fraction"].mean()), 6),
            }
        )
    patch_balance = pd.DataFrame(patch_rows)

    outputs = {
        "availability": output_dir / "scene_availability.csv",
        "event_areas": output_dir / "event_areas.csv",
        "patch_balance": output_dir / "patch_balance.csv",
        "summary": output_dir / "eda_summary.json",
        "report": output_dir / "EDA_REPORT.md",
    }
    availability.to_csv(outputs["availability"], index=False)
    event_areas.to_csv(outputs["event_areas"], index=False)
    patch_balance.to_csv(outputs["patch_balance"], index=False)

    event_patches = patches[patches["event_kind"] != "baseline"]
    summary = {
        "pairs": int(len(audit)),
        "ready_s1_pairs": int(audit["s1_ready"].sum()),
        "ready_s2_pairs": int(audit["s2_ready"].sum()),
        "patches": int(len(patches)),
        "flood_positive_patches": int((patches["flood_fraction"] > 0).sum()),
        "flood_positive_fraction": round(
            float((patches["flood_fraction"] > 0).mean()), 6
        ),
        "event_flood_positive_fraction": round(
            float((event_patches["flood_fraction"] > 0).mean()), 6
        ),
        "pairs_without_optical": audit.loc[~audit["s2_ready"], "pair_id"].tolist(),
        "incomplete_pairs": audit.loc[audit["status"] != "ready", "pair_id"].tolist(),
    }
    outputs["summary"].write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# Exploratory data analysis",
        "",
        "## Readiness",
        "",
        f"- Pairs: **{summary['pairs']}**.",
        f"- Sentinel-1 ready: **{summary['ready_s1_pairs']} / {summary['pairs']}**.",
        f"- Sentinel-2 ready: **{summary['ready_s2_pairs']} / {summary['pairs']}**.",
        f"- Indexed patches: **{summary['patches']}**.",
        "",
        "## Class imbalance",
        "",
        f"Flood-positive patches: **{summary['flood_positive_patches']}** "
        f"(**{summary['flood_positive_fraction']:.2%}** of all indexed patches).",
        "This imbalance is handled by positive-patch retention, negative sampling, "
        "flood-channel loss weighting and event-level validation.",
        "",
        "## Optical availability",
        "",
        "Pairs without a complete optical pre/peak pair: "
        + (", ".join(summary["pairs_without_optical"]) or "none"),
        "Optical dropout and explicit missing values keep these SAR-only pairs usable.",
        "",
        "## Generated tables",
        "",
        "- `scene_availability.csv`: sensor coverage by event;",
        "- `event_areas.csv`: reference water, flood and recession areas;",
        "- `patch_balance.csv`: patch-level class imbalance by event.",
        "",
    ]
    outputs["report"].write_text("\n".join(lines), encoding="utf-8")
    return outputs
