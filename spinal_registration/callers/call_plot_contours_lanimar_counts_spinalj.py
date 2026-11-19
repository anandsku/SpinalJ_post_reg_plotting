#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Session: Plotting contour plots for spinal J

Purpose:
    Caller for plot_contours_lanimar_counts_spinalj: assigns cells (one or many reference
    segments), writes CSVs and plots per reference segment (GM/WM, lamina color, KDE, bar).

Inputs (edit paths below — all required except optional overrides noted):

    cell_csv_path (required)
        Cell metadata CSV (X, Y, Reference section, scaling factor).

    atlas_dir (required — must be set explicitly; no DEFAULT_ATLAS_DIR in batch API)
        SpinalJ atlas folder (Atlas_Regions.csv, Segments.csv, Annotation_Segments_Only.tif).

    plot_settings_csv (optional explicit path; default None → beside cell_csv_path)
        Format: Parameter, Default, User_value, Description; booleans TRUE/FALSE.

    outputdir (optional explicit path; default None → <cell_csv parent>/output)

Outputs:
    Per-segment CSVs and PNGs under outputdir; console summary via _print_summary.

Associated code:
    Function module: spinal_registration/functions/plot_contours_lanimar_counts_spinalj.py

Process:
    process_spinalj_by_reference_segment() with validated atlas_dir → print summaries.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

_this_dir = os.path.dirname(os.path.abspath(__file__))
_base = os.path.basename(_this_dir)
if _base == "cursor_output":
    _spinal_root = os.path.dirname(_this_dir)
elif _base == "callers":
    _spinal_root = os.path.dirname(_this_dir)
else:
    _spinal_root = os.path.dirname(_this_dir)

if _spinal_root not in sys.path:
    sys.path.insert(0, _spinal_root)

from functions.plot_contours_lanimar_counts_spinalj import (  # noqa: E402
    process_spinalj_by_reference_segment,
)

_REPO_ROOT = Path(_spinal_root)

# --- Paths (relative to spinal_registration package root) ---
cell_csv_path = _REPO_ROOT / "test_data" / "cell_loc_and_metadata.csv"
plot_settings_csv = _REPO_ROOT / "test_data" / "spinalj_plot_settings.csv"
atlas_dir = _REPO_ROOT / "atlas_data" / "useful_atlas_data"
outputdir = _REPO_ROOT / "test_data" / "output"


def _print_segment_summary(segment: str, seg_result: dict[str, Any]) -> None:
    assigned = seg_result["assigned_df"]
    counts = seg_result["laminar_counts"]

    print(f"\n{'=' * 60}")
    print(f"Reference segment: {segment}")
    print(f"{'=' * 60}")
    print(f"  n_cells        : {seg_result['n_cells']}")
    print(f"  in_mask_bounds : {int(assigned['in_mask_bounds'].sum())}")

    print("\n  Fine region counts (top 15):")
    fine = assigned["region_acronym"].value_counts().head(15)
    for acr, n in fine.items():
        print(f"    {acr:8s} {n}")

    print("\n  Combined lamina counts:")
    for lam, n in counts.items():
        print(f"    {lam:16s} {n}")

    if seg_result.get("laminar_counts_path"):
        print(f"\n  Wrote laminar counts: {seg_result['laminar_counts_path']}")
    if seg_result.get("assigned_cells_path"):
        print(f"  Wrote per-cell assignments: {seg_result['assigned_cells_path']}")

    scatter = seg_result["scatter_plot"]
    scatter_outline = seg_result["scatter_matter_outline_plot"]
    scatter_filled = seg_result["scatter_outline_filled_plot"]
    lamina_color = seg_result["lamina_color_plot"]
    lamina_color_outline = seg_result["lamina_color_matter_outline_plot"]
    bar = seg_result["bar_plot"]
    kde_pct = seg_result["kde_percentile_plot"]
    kde_pct_outline = seg_result["kde_percentile_matter_outline_plot"]
    kde_pct_filled = seg_result["kde_percentile_outline_filled_plot"]
    kde_frac = seg_result["kde_fraction_max_plot"]
    kde_frac_outline = seg_result["kde_fraction_max_matter_outline_plot"]
    kde_frac_filled = seg_result["kde_fraction_max_outline_filled_plot"]

    print(f"  Wrote GM/WM scatter backdrop: {scatter['output_png']}")
    print(f"  Wrote GM/WM scatter + outlines: {scatter_outline['output_png']}")
    print(f"  Wrote GM/WM scatter outline-filled: {scatter_filled['output_png']}")
    print(f"  Wrote lamina color backdrop: {lamina_color['output_png']}")
    print(f"  Wrote lamina color + outlines: {lamina_color_outline['output_png']}")
    print(f"  Wrote laminar bar plot: {bar['output_png']}")
    print(f"  Wrote KDE percentile contours: {kde_pct['output_png']}")
    print(f"  Wrote KDE percentile + outlines: {kde_pct_outline['output_png']}")
    print(f"  Wrote KDE percentile outline-filled: {kde_pct_filled['output_png']}")
    print(f"  Wrote KDE fraction-of-max contours: {kde_frac['output_png']}")
    print(f"  Wrote KDE fraction-of-max + outlines: {kde_frac_outline['output_png']}")
    print(f"  Wrote KDE fraction-of-max outline-filled: {kde_frac_filled['output_png']}")


def _print_summary(result: dict[str, Any]) -> None:
    print("Region assignment (all segments)")
    print(f"  total cells    : {len(result['assigned_df'])}")
    print(f"  segments       : {result['reference_segments']}")
    print(f"  output_dir     : {result['output_dir']}")
    print(f"  plot_settings  : {result['plot_settings_csv']}")
    for segment in result["reference_segments"]:
        _print_segment_summary(segment, result["by_segment"][segment])


def main() -> None:
    result = process_spinalj_by_reference_segment(
        str(cell_csv_path),
        str(atlas_dir),
        outputdir=str(outputdir),
        plot_settings_csv=str(plot_settings_csv),
    )
    _print_summary(result)
    print("\nDone.")


if __name__ == "__main__":
    main()