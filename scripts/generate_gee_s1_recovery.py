from __future__ import annotations

from pathlib import Path
import argparse
import json

from generate_gee_exports import build_specs


TEMPLATE = r"""// Recovery export for an incomplete Sentinel-1 scene.
// Candidates are ranked by distance from the requested date; closest pixels win.
var pair = __PAIR_SPEC__;
var exportFolder = '__EXPORT_FOLDER__';
var searchDays = __SEARCH_DAYS__;

function recover(windowName, dateText) {
  var target = ee.Date(dateText);
  var region = ee.Geometry.Rectangle(pair.bounds, pair.crs, false);
  var candidates = ee.ImageCollection('COPERNICUS/S1_GRD')
    .filterBounds(region)
    .filterDate(target.advance(-searchDays, 'day'), target.advance(searchDays + 1, 'day'))
    .filter(ee.Filter.eq('instrumentMode', 'IW'))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
    .select(['VV', 'VH'])
    .map(function(image) {
      var distance = image.date().difference(target, 'day').abs();
      return image.set('dateDistance', distance);
    });
  print(pair.pairId + ' ' + windowName + ' recovery candidates',
    candidates.aggregate_array('system:time_start'),
    candidates.aggregate_array('relativeOrbitNumber_start'),
    candidates.aggregate_array('orbitProperties_pass'));
  // Earth Engine mosaic gives the last image priority. Sort far-to-near so
  // pixels from the acquisition closest to the requested date win.
  var sar = candidates.sort('dateDistance', false).mosaic().clip(region);
  // Training reads VV and VH directly; omitting the derived ratio makes the
  // emergency export one third smaller without changing model inputs.
  var output = sar.toFloat().unmask(-9999);
  Export.image.toDrive({
    image: output,
    description: pair.pairId + '_S1_' + windowName + '_recovery',
    folder: exportFolder,
    fileNamePrefix: pair.pairId + '__S1_' + windowName + '_' + dateText,
    region: region,
    crs: pair.crs,
    crsTransform: pair.transform,
    maxPixels: 1e10,
    fileFormat: 'GeoTIFF',
    formatOptions: {cloudOptimized: true, noData: -9999}
  });
}

recover('pre', pair.datePreSar);
recover('peak', pair.datePeakSar);
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate recovery exports for incomplete S1 coverage")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--pair-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--drive-folder", default="hydrowatch_amur")
    parser.add_argument("--search-days", type=int, default=6)
    args = parser.parse_args()
    matches = [spec for spec in build_specs(args.data_root) if spec["pairId"] == args.pair_id]
    if len(matches) != 1:
        raise ValueError(f"Expected one pair_id={args.pair_id!r}, found {len(matches)}")
    script = TEMPLATE.replace("__PAIR_SPEC__", json.dumps(matches[0], ensure_ascii=False, indent=2))
    script = script.replace("__EXPORT_FOLDER__", args.drive_folder.replace("'", ""))
    script = script.replace("__SEARCH_DAYS__", str(args.search_days))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(script, encoding="utf-8")
    print(f"Generated {args.output}")


if __name__ == "__main__":
    main()
