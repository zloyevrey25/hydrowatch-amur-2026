from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any, Literal
import json
import os

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator


LAYER_NAMES = ("water_pre", "water_peak", "flood", "receded")


class AnalysisRequest(BaseModel):
    pair_id: str | None = None
    bbox: list[float] | None = Field(default=None, min_length=4, max_length=4)
    geometry: dict[str, Any] | None = None
    date_pre: str | None = None
    date_peak: str | None = None

    @model_validator(mode="after")
    def has_spatial_selector(self):
        if not self.pair_id and self.bbox is None and self.geometry is None:
            raise ValueError("Provide pair_id, bbox or GeoJSON geometry")
        return self


def _geometry_bbox(geometry: dict[str, Any]) -> list[float]:
    coordinates = geometry.get("coordinates")
    if not coordinates:
        raise ValueError("GeoJSON geometry has no coordinates")

    points: list[tuple[float, float]] = []

    def collect(value):
        if (
            isinstance(value, (list, tuple)) and len(value) >= 2
            and isinstance(value[0], (int, float)) and isinstance(value[1], (int, float))
        ):
            points.append((float(value[0]), float(value[1])))
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                collect(item)

    collect(coordinates)
    if not points:
        raise ValueError("GeoJSON geometry has no coordinate pairs")
    xs, ys = zip(*points, strict=True)
    return [min(xs), min(ys), max(xs), max(ys)]


def _intersection_area(left: list[float], right: list[float]) -> float:
    width = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    height = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    return width * height


def _projected_geometry_area_m2(geometry: dict[str, Any]) -> float:
    def ring_area(ring) -> float:
        return abs(sum(
            float(x1) * float(y2) - float(x2) * float(y1)
            for (x1, y1), (x2, y2) in zip(ring, ring[1:] + ring[:1])
        )) / 2.0

    def polygon_area(polygon) -> float:
        return ring_area(polygon[0]) - sum(ring_area(hole) for hole in polygon[1:])

    if geometry["type"] == "Polygon":
        return polygon_area(geometry["coordinates"])
    if geometry["type"] == "MultiPolygon":
        return sum(polygon_area(polygon) for polygon in geometry["coordinates"])
    return 0.0


class ResultStore:
    def __init__(self, data_root: Path, results_root: Path):
        self.data_root = Path(data_root)
        self.results_root = Path(results_root)
        self.pairs = pd.read_csv(self.data_root / "pairs.csv")
        aoi_path = self.data_root / "vectors" / "aoi.geojson"
        collection = json.loads(aoi_path.read_text(encoding="utf-8"))
        self.aois = {feature["properties"]["aoi_id"]: feature for feature in collection["features"]}
        submission_path = self.results_root / "submission.csv"
        self.submission = (
            pd.read_csv(submission_path).set_index("pair_id")
            if submission_path.exists() else pd.DataFrame()
        )
        self.cache_dir = self.results_root / "service_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def list_pairs(self) -> list[dict[str, Any]]:
        columns = [
            "pair_id", "aoi_id", "aoi_name", "event_id", "event_name", "event_kind",
            "date_pre_sar", "date_peak_sar", "date_pre_opt", "date_peak_opt", "aoi_km2",
        ]
        return self.pairs[columns].where(pd.notna(self.pairs[columns]), None).to_dict("records")

    def match_pair(self, request: AnalysisRequest) -> pd.Series:
        candidates = self.pairs.copy()
        if request.pair_id:
            candidates = candidates[candidates["pair_id"] == request.pair_id]
        if request.date_pre:
            candidates = candidates[
                (candidates["date_pre_sar"] == request.date_pre)
                | (candidates["date_pre_opt"] == request.date_pre)
            ]
        if request.date_peak:
            candidates = candidates[
                (candidates["date_peak_sar"] == request.date_peak)
                | (candidates["date_peak_opt"] == request.date_peak)
            ]
        if candidates.empty:
            raise HTTPException(404, "No prepared scene matches the requested territory and dates")

        request_bbox = request.bbox
        if request_bbox is None and request.geometry is not None:
            try:
                request_bbox = _geometry_bbox(request.geometry)
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from exc
        if request_bbox is not None:
            ranked = []
            for index, row in candidates.iterrows():
                aoi_bbox = _geometry_bbox(self.aois[row["aoi_id"]]["geometry"])
                ranked.append((_intersection_area(request_bbox, aoi_bbox), index))
            overlap, index = max(ranked)
            if overlap <= 0:
                raise HTTPException(404, "The requested geometry does not intersect a prepared AOI")
            return candidates.loc[index]
        return candidates.iloc[0]

    def layer_path(self, pair_id: str, layer: str) -> Path:
        if layer not in LAYER_NAMES:
            raise HTTPException(404, f"Unknown layer: {layer}")
        path = self.results_root / "predictions" / f"{pair_id}_{layer}.tif"
        if not path.exists():
            raise HTTPException(404, f"Layer has not been generated: {layer}")
        return path

    def _mask_area(self, pair_id: str, layer: str) -> float:
        import rasterio

        with rasterio.open(self.layer_path(pair_id, layer)) as source:
            mask = source.read(1)
            pixel_ha = abs(
                source.transform.a * source.transform.e - source.transform.b * source.transform.d
            ) / 10_000.0
        return round(float(np.count_nonzero(mask == 1) * pixel_ha), 2)

    def analyze(self, request: AnalysisRequest) -> dict[str, Any]:
        pair = self.match_pair(request)
        pair_id = str(pair["pair_id"])
        if self.submission.empty or pair_id not in self.submission.index:
            raise HTTPException(409, "Predictions for this prepared scene are not available yet")
        payload = request.model_dump(mode="json") | {"pair_id": pair_id}
        result_id = sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]
        report_path = self.cache_dir / f"{result_id}.json"
        if report_path.exists():
            return json.loads(report_path.read_text(encoding="utf-8"))

        areas = self.submission.loc[pair_id]
        receded_ha = self._mask_area(pair_id, "receded")
        report = {
            "id": result_id,
            "pair_id": pair_id,
            "aoi_id": pair["aoi_id"],
            "aoi_name": pair["aoi_name"],
            "event_id": pair["event_id"],
            "event_name": pair["event_name"],
            "dates": {"pre": pair["date_pre_sar"], "peak": pair["date_peak_sar"]},
            "areas": {
                "water_pre_ha": float(areas["water_pre_ha"]),
                "water_peak_ha": float(areas["water_peak_ha"]),
                "flood_ha": float(areas["flood_ha"]),
                "receded_ha": receded_ha,
                "water_increase_ha": round(
                    max(0.0, float(areas["water_peak_ha"]) - float(areas["water_pre_ha"])), 2
                ),
                "flood_share_aoi": round(float(areas["flood_ha"]) / (pair["aoi_km2"] * 100.0), 6),
            },
            "landcover_distribution": self.landcover_distribution(pair_id),
            "layers": {name: f"/api/v1/results/{result_id}/layers/{name}" for name in LAYER_NAMES},
            "contours": f"/api/v1/results/{result_id}/contours.geojson",
        }
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return report

    def landcover_distribution(self, pair_id: str) -> dict[str, Any]:
        # Populated automatically once a local ESA WorldCover raster is exported.
        worldcover = self.data_root / "landcover" / f"{pair_id}_worldcover.tif"
        flood_path = self.results_root / "predictions" / f"{pair_id}_flood.tif"
        if not worldcover.exists() or not flood_path.exists():
            return {"status": "not_available", "classes": {}}
        import rasterio

        with rasterio.open(worldcover) as cover, rasterio.open(flood_path) as flood:
            if (cover.width, cover.height, cover.transform, cover.crs) != (
                flood.width, flood.height, flood.transform, flood.crs,
            ):
                return {"status": "grid_mismatch", "classes": {}}
            classes = cover.read(1)[flood.read(1) == 1]
            pixel_ha = abs(cover.transform.a * cover.transform.e) / 10_000.0
        values, counts = np.unique(classes, return_counts=True)
        return {
            "status": "ready",
            "classes": {str(int(value)): round(float(count * pixel_ha), 2) for value, count in zip(values, counts)},
        }

    def report(self, result_id: str) -> dict[str, Any]:
        path = self.cache_dir / f"{result_id}.json"
        if not path.exists():
            raise HTTPException(404, "Unknown result id")
        return json.loads(path.read_text(encoding="utf-8"))

    def contours(self, result_id: str) -> dict[str, Any]:
        import rasterio
        from rasterio.features import shapes
        from rasterio.warp import transform_geom

        report = self.report(result_id)
        features = []
        for layer in LAYER_NAMES:
            path = self.layer_path(report["pair_id"], layer)
            with rasterio.open(path) as source:
                mask = source.read(1).astype("uint8")
                for index, (geometry, value) in enumerate(shapes(mask, mask=mask == 1, transform=source.transform)):
                    if int(value) != 1:
                        continue
                    area_ha = _projected_geometry_area_m2(geometry) / 10_000.0
                    features.append({
                        "type": "Feature",
                        "id": f"{report['pair_id']}:{layer}:{index}",
                        "geometry": transform_geom(source.crs, "EPSG:4326", geometry),
                        "properties": {
                            "id": f"{report['pair_id']}:{layer}:{index}",
                            "type": layer,
                            "area_ha": round(area_ha, 2),
                            "area_km2": round(area_ha / 100.0, 4),
                        },
                    })
        return {"type": "FeatureCollection", "features": features}


def create_app(data_root: Path, results_root: Path, web_root: Path | None = None) -> FastAPI:
    store = ResultStore(data_root, results_root)
    app = FastAPI(title="HydroWatch Amur", version="1.0.0")

    @app.get("/api/v1/pairs")
    def pairs():
        return store.list_pairs()

    @app.post("/api/v1/analyze")
    def analyze(request: AnalysisRequest):
        return store.analyze(request)

    @app.get("/api/v1/results/{result_id}/report")
    def report(result_id: str, format: Literal["json", "csv"] = "json"):
        result = store.report(result_id)
        if format == "json":
            return result
        row = {
            "pair_id": result["pair_id"], "aoi_name": result["aoi_name"],
            "date_pre": result["dates"]["pre"], "date_peak": result["dates"]["peak"],
            **result["areas"],
        }
        return PlainTextResponse(pd.DataFrame([row]).to_csv(index=False), media_type="text/csv")

    @app.get("/api/v1/results/{result_id}/contours.geojson")
    def contours(result_id: str):
        return store.contours(result_id)

    @app.get("/api/v1/results/{result_id}/layers/{layer}")
    def layer(result_id: str, layer: Literal["water_pre", "water_peak", "flood", "receded"]):
        result = store.report(result_id)
        return FileResponse(store.layer_path(result["pair_id"], layer), media_type="image/tiff")

    static = web_root or Path(__file__).with_name("web")
    app.mount("/", StaticFiles(directory=static, html=True), name="web")
    return app


def app_from_environment() -> FastAPI:
    return create_app(
        Path(os.getenv("HYDROWATCH_DATA_ROOT", "data/hydrowatch_amur")),
        Path(os.getenv("HYDROWATCH_RESULTS_ROOT", "outputs/final")),
        Path(os.environ["HYDROWATCH_WEB_ROOT"]) if "HYDROWATCH_WEB_ROOT" in os.environ else None,
    )


app = app_from_environment()
