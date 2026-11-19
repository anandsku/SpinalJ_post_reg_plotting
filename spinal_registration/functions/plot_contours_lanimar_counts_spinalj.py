#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Session: Plotting contour plots for spinal J

Purpose:
    Map registered cell coordinates into SpinalJ atlas region IDs using segment-level
    annotation masks, then summarize laminar (combined lamina I–X) cell counts.
    Also renders cell locations in atlas coordinates on a gray-matter / white-matter
    segment mask backdrop (high-DPI PNG).

Inputs (assign_cells_to_spinalj_regions):
    cell_csv_path           : str — CSV with Cell, X, Y, Reference section, scaling factor.
    atlas_dir               : str — directory containing Atlas_Regions.csv, Segments.csv,
                              Annotation_Segments_Only.tif (default DEFAULT_ATLAS_DIR).
    reference_section_col   : str — column name for segment label (default
                              "Reference section").
    scaling_col             : str — column name for atlas→cell scale factor (default
                              "Scaling factor for reference section").
    x_col, y_col            : str — cell coordinate columns (default "X", "Y").
    out_of_bounds_region_id : int — label when scaled coords fall outside mask
                              (default 0).

Inputs (summarize_laminar_counts):
    assigned_df             : DataFrame from assign_cells_to_spinalj_regions.
    count_column            : str — column used for value_counts (default
                              "lamina_combined_acronym").

Outputs:
    assign_cells_to_spinalj_regions returns DataFrame (original columns plus):
      atlas_x, atlas_y, segment, segment_frame_index, atlas_region_id,
      region_acronym, region_name, lamina_combined_acronym, matter_class (GM/WM/other),
      in_mask_bounds.
    summarize_laminar_counts returns Series (counts per combined lamina) and accepts
    optional paths to write CSV (laminar summary and/or full per-cell assignments).
    plot_laminar_counts_bar, plot_cells_on_spinalj_matter_backdrop,
    plot_cells_on_lamina_color_backdrop,
    plot_cell_kde_on_spinalj_backdrop, and plot_cell_kde_fraction_of_max_on_spinalj_backdrop
    each write a high-DPI PNG path in their return dict.
    process_spinalj_by_reference_segment runs assignment once and writes CSVs/plots per
    reference segment. Outputs go under outputdir (default: <cell_csv parent>/output)
    using fixed basenames; each file gets _<segment> before the extension.

Inputs (plot_laminar_counts_bar):
    laminar_counts : Series from summarize_laminar_counts.
    output_png     : str — bar chart PNG path.
    dpi            : int — default DEFAULT_PLOT_DPI (300).

Inputs (plot_cell_kde_on_spinalj_backdrop):
    kernel_width_fraction : float — KDE Gaussian σ (pixels) = fraction × GM vertical height
                            (default DEFAULT_KERNEL_WIDTH_FRACTION = 0.05).
    kde_contour_percentile_min/max/step : int — contour levels at percentiles of positive KDE
                            grid (defaults 45, 95, step 10).

Inputs (plot_cell_kde_fraction_of_max_on_spinalj_backdrop):
    Same kernel_width_fraction and backdrop/outline options as percentile KDE plot.
    kde_fraction_max_min/max/step : float — levels at fraction × grid max KDE
                            (defaults 0.35, 0.95, step 0.1).

Inputs (plot_cells_on_spinalj_matter_backdrop):
    assigned_df    : DataFrame with atlas_x, atlas_y, segment (from assignment step).
    output_png     : str — output PNG path.
    atlas_dir      : str — atlas folder (default DEFAULT_ATLAS_DIR).
    segment        : str or None — spinal segment label; default None uses unique segment
                     in assigned_df (error if multiple).
    dpi            : int — matplotlib savefig DPI (default DEFAULT_PLOT_DPI, 300).
    gm_color       : RGB tuple for gray matter (default medium gray).
    wm_color       : RGB tuple for white matter (default white).
    other_color    : RGB tuple for non-GM/WM pixels in frame (default light gray).
    cell_color     : matplotlib color for cell markers (default crimson).
    cell_marker_size : float — scatter marker size in points^2 (default 18).

Inputs (process_spinalj_by_reference_segment):
    cell_csv_path (required)
        Cell metadata CSV; drives assignment and default output directory.

    atlas_dir (required — no default in process_spinalj_by_reference_segment)
        SpinalJ atlas folder containing Atlas_Regions.csv, Segments.csv,
        Annotation_Segments_Only.tif (validated before assignment).

    outputdir (default: None → <cell_csv parent>/output)
        Folder for laminar/assignment CSVs and all plot PNG basenames.

    plot_settings_csv (default: None → <cell_csv parent>/spinalj_plot_settings.csv)
        Batch plot parameters in CSV (created from module defaults if missing).
        Columns: Parameter, Default, User_value, Description (blank User_value uses Default).
        Settings include plot_dpi (300), kernel_width_fraction (0.05), KDE percentile/fraction
        levels, show_contour_inline_labels (False), plot_lamina_roman_labels (True),
        gm_outline_smooth_sigma_fraction (0.03), wm_outline_smooth_sigma_fraction (0.03).

    **assign_kwargs
        Forwarded to assign_cells_to_spinalj_regions (column names, out_of_bounds_region_id, …).

Associated code:
    Caller: spinal_registration/callers/call_plot_contours_lanimar_counts_spinalj.py

Process:
    Load Segments.csv → map Reference section → TIFF frame index; load
    Annotation_Segments_Only.tif frame; scale cell X/Y into atlas pixel space
    (atlas_coord = cell_coord / scaling_factor); sample region id; join
    Atlas_Regions.csv; roll fine ids up to Lamina_I–Lamina_X via children_IDs;
    classify GM vs WM by parent chain.

Prompt preceding this version:
    Require explicit atlas_dir in batch API; align with caller path edits (settings/output).

Python environment: /research/groups/bikoffgrp/home/common/AK_python_envs/general_analysis
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal, Optional, Union

MatterBackdropMode = Literal["atlas_matter", "outline_filled"]
MATTER_BACKDROP_ATLAS: MatterBackdropMode = "atlas_matter"
MATTER_BACKDROP_OUTLINE_FILLED: MatterBackdropMode = "outline_filled"

import matplotlib

matplotlib.use("Agg")

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi

DEFAULT_ATLAS_DIR = (
    "/research/groups/bikoffgrp/home/common/AK_RD_IS/code_AK/Py/spinal_registration/"
    "atlas_data/useful_atlas_data"
)
ATLAS_REGIONS_FILENAME = "Atlas_Regions.csv"
SEGMENTS_FILENAME = "Segments.csv"
SEGMENT_MASK_FILENAME = "Annotation_Segments_Only.tif"
ANNOTATION_COLOR_FILENAME = "Annotation_Color.tif"
ANNOTATION_COLOR_CHANNELS_PER_Z = 3

DEFAULT_PLOT_DPI = 300
OUTPUT_LAMINAR_COUNTS_CSV_NAME = "laminar_counts_from_spinalj_match.csv"
OUTPUT_ASSIGNED_CELLS_CSV_NAME = "cell_loc_spinalj_region_assignments.csv"
OUTPUT_CELLS_ON_ATLAS_PNG_NAME = "cell_locations_on_spinalj_backdrop.png"
OUTPUT_CELLS_ON_LAMINA_COLOR_PNG_NAME = "cell_locations_on_lamina_color_backdrop.png"
OUTPUT_LAMINAR_COUNTS_BAR_PNG_NAME = "laminar_counts_bar.png"
OUTPUT_CELL_KDE_ON_ATLAS_PNG_NAME = "cell_kde_on_spinalj_backdrop.png"
OUTPUT_CELL_KDE_FRAC_MAX_PNG_NAME = "cell_kde_fraction_of_max_spinalj_backdrop.png"
SPINALJ_DEFAULT_OUTPUT_SUBDIR = "output"
SPINALJ_PLOT_SETTINGS_CSV_NAME = "spinalj_plot_settings.csv"
SPINALJ_PLOT_SETTINGS_CSV_COLUMNS: tuple[str, ...] = (
    "Parameter",
    "Default",
    "User_value",
    "Description",
)
DEFAULT_GM_RGB = (0.45, 0.45, 0.45)
DEFAULT_WM_RGB = (1.0, 1.0, 1.0)
DEFAULT_OTHER_RGB = (0.88, 0.88, 0.88)
DEFAULT_KERNEL_WIDTH_FRACTION = 0.05
DEFAULT_KDE_CONTOUR_PERCENTILE_MIN = 45
DEFAULT_KDE_CONTOUR_PERCENTILE_MAX = 95
DEFAULT_KDE_CONTOUR_PERCENTILE_STEP = 10
DEFAULT_KDE_FRACTION_MAX_MIN = 0.35
DEFAULT_KDE_FRACTION_MAX_MAX = 0.95
DEFAULT_KDE_FRACTION_MAX_STEP = 0.1

ATLAS_XY_UM_PER_PIXEL = 10.0
SCALE_BAR_LENGTH_PX = 10
SCALE_BAR_LABEL = "100 um"
SCALE_BAR_INSET_RIGHT_FRAC = 0.10
SCALE_BAR_INSET_BOTTOM_FRAC = 0.10
SCALE_BAR_LABEL_FONTSIZE = 10.8  # 9 pt × 1.2
LAMINA_ROMAN_LABEL_FONTSIZE = 11.0
DEFAULT_MATTER_OUTLINE_SMOOTH_SIGMA_FRACTION = 0.03
DEFAULT_GM_OUTLINE_SMOOTH_SIGMA_FRACTION = DEFAULT_MATTER_OUTLINE_SMOOTH_SIGMA_FRACTION
DEFAULT_WM_OUTLINE_SMOOTH_SIGMA_FRACTION = 0.03
DEFAULT_WM_OUTLINE_INITIAL_DILATION_ITERS = 0
DEFAULT_WM_OUTLINE_ENCLOSURE_MAX_DILATION_STEPS = 48
DEFAULT_GM_OUTLINE_COLOR = "black"
DEFAULT_WM_OUTLINE_COLOR = "dimgray"
DEFAULT_MATTER_OUTLINE_LINEWIDTH = 2.0
DEFAULT_GM_OUTLINE_LINEWIDTH = DEFAULT_MATTER_OUTLINE_LINEWIDTH

COMBINED_LAMINA_ORDER: tuple[str, ...] = (
    "Lamina_I",
    "Lamina_II",
    "Lamina_III",
    "Lamina_IV",
    "Lamina_V",
    "Lamina_VI",
    "Lamina_VII",
    "Lamina_VIII",
    "Lamina_IX",
    "Lamina_X",
)

PathLike = Union[str, os.PathLike[str]]
RGB = tuple[float, float, float]


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace from column names (cell CSV has trailing spaces)."""
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out


def _build_lamina_combined_lookup(atlas_regions: pd.DataFrame) -> dict[int, str]:
    """Map fine atlas region id → combined lamina acronym (Lamina_I … Lamina_X)."""
    lookup: dict[int, str] = {}
    combined = atlas_regions[
        atlas_regions["acronym"].astype(str).str.startswith("Lamina_", na=False)
    ]
    for _, row in combined.iterrows():
        lamina_acr = str(row["acronym"])
        child_field = row["children_IDs"]
        if pd.isna(child_field) or str(child_field).strip() == "":
            continue
        for part in str(child_field).split(","):
            part = part.strip()
            if part:
                lookup[int(part)] = lamina_acr
    return lookup


def _build_matter_class_lookup(atlas_regions: pd.DataFrame) -> dict[int, str]:
    """Map region id → 'GM', 'WM', or 'other' by walking parent_ID to GM/WM."""
    parent = dict(zip(atlas_regions["id"].astype(int), atlas_regions["parent_ID"].astype(int)))
    gm_id = int(atlas_regions.loc[atlas_regions["acronym"] == "GM", "id"].iloc[0])
    wm_id = int(atlas_regions.loc[atlas_regions["acronym"] == "WM", "id"].iloc[0])

    def matter_for(rid: int) -> str:
        if rid <= 0:
            return "other"
        seen: set[int] = set()
        cur = rid
        while cur not in seen and cur != 0:
            if cur == gm_id:
                return "GM"
            if cur == wm_id:
                return "WM"
            seen.add(cur)
            cur = parent.get(cur, 0)
        return "other"

    ids = atlas_regions["id"].astype(int).tolist()
    return {i: matter_for(i) for i in ids}


def validate_spinalj_atlas_dir(atlas_dir: PathLike) -> Path:
    """
    Ensure atlas_dir exists and contains required SpinalJ atlas files.

    Raises FileNotFoundError if the directory or a required file is missing.
    """
    atlas_path = Path(atlas_dir).resolve()
    if not atlas_path.is_dir():
        raise FileNotFoundError(f"Atlas directory not found: {atlas_path}")
    required = (
        ATLAS_REGIONS_FILENAME,
        SEGMENTS_FILENAME,
        SEGMENT_MASK_FILENAME,
    )
    missing = [name for name in required if not (atlas_path / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Atlas directory {atlas_path} is missing required file(s): {missing}"
        )
    return atlas_path


def load_spinalj_segment_masks(atlas_dir: PathLike) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Load Segments.csv and all frames from Annotation_Segments_Only.tif.

    Returns (segments_df, masks) where masks.shape == (n_segments, height, width).
    Frame order matches row order in Segments.csv (C1 = index 0, …, Co3 = index 33).
    """
    atlas_path = Path(atlas_dir)
    segments = pd.read_csv(atlas_path / SEGMENTS_FILENAME)
    segments.columns = [str(c).strip() for c in segments.columns]

    tif_path = atlas_path / SEGMENT_MASK_FILENAME
    masks: list[np.ndarray] = []
    with Image.open(tif_path) as im:
        n_frames = getattr(im, "n_frames", 1)
        if n_frames != len(segments):
            raise ValueError(
                f"{SEGMENT_MASK_FILENAME} has {n_frames} frames but "
                f"{SEGMENTS_FILENAME} has {len(segments)} rows."
            )
        for frame_idx in range(n_frames):
            im.seek(frame_idx)
            masks.append(np.array(im))

    stack = np.stack(masks, axis=0)
    return segments, stack


def segment_frame_index_for_label(segments: pd.DataFrame, segment_label: str) -> int:
    """0-based TIFF frame index for a segment name (e.g. 'L4')."""
    col = "Segment"
    if col not in segments.columns:
        raise KeyError(f"Segments table missing '{col}' column.")
    matches = segments.index[segments[col].astype(str) == str(segment_label)]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one row for segment {segment_label!r}, found {len(matches)}."
        )
    return int(matches[0])


def assign_cells_to_spinalj_regions(
    cell_csv_path: PathLike,
    atlas_dir: PathLike = DEFAULT_ATLAS_DIR,
    *,
    reference_section_col: str = "Reference section",
    scaling_col: str = "Scaling factor for reference section",
    x_col: str = "X",
    y_col: str = "Y",
    out_of_bounds_region_id: int = 0,
) -> pd.DataFrame:
    """
    Match each cell (X, Y) to an atlas region id on the appropriate segment mask.

    Atlas pixel coordinates: atlas_x = X / scaling_factor, atlas_y = Y / scaling_factor
    (scaling harmonizes the atlas framework with the cell coordinate system).
    """
    atlas_path = Path(atlas_dir)
    atlas_regions = pd.read_csv(atlas_path / ATLAS_REGIONS_FILENAME)
    id_to_acronym = dict(zip(atlas_regions["id"].astype(int), atlas_regions["acronym"]))
    id_to_name = dict(zip(atlas_regions["id"].astype(int), atlas_regions["name"]))
    lamina_lookup = _build_lamina_combined_lookup(atlas_regions)
    matter_lookup = _build_matter_class_lookup(atlas_regions)

    segments, mask_stack = load_spinalj_segment_masks(atlas_path)
    segment_to_frame = {
        str(row["Segment"]): segment_frame_index_for_label(segments, str(row["Segment"]))
        for _, row in segments.iterrows()
    }

    cells = _normalize_columns(pd.read_csv(cell_csv_path))
    ref_col = reference_section_col.strip()
    scale_col = scaling_col.strip()
    if ref_col not in cells.columns:
        raise KeyError(f"Cell CSV missing column {ref_col!r}.")
    if scale_col not in cells.columns:
        raise KeyError(f"Cell CSV missing column {scale_col!r}.")

    n_frames, height, width = mask_stack.shape

    atlas_x_list: list[float] = []
    atlas_y_list: list[float] = []
    frame_idx_list: list[int] = []
    region_ids: list[int] = []
    in_bounds: list[bool] = []

    for _, row in cells.iterrows():
        segment = str(row[ref_col])
        if segment not in segment_to_frame:
            raise ValueError(f"Unknown reference section {segment!r} (not in Segments.csv).")
        frame_idx = segment_to_frame[segment]
        scale = float(row[scale_col])
        if scale <= 0:
            raise ValueError(f"Scaling factor must be positive, got {scale}.")

        ax = float(row[x_col]) / scale
        ay = float(row[y_col]) / scale
        col_i = int(round(ax))
        row_i = int(round(ay))
        ok = 0 <= row_i < height and 0 <= col_i < width
        if ok:
            rid = int(mask_stack[frame_idx, row_i, col_i])
        else:
            rid = out_of_bounds_region_id

        atlas_x_list.append(ax)
        atlas_y_list.append(ay)
        frame_idx_list.append(frame_idx)
        region_ids.append(rid)
        in_bounds.append(ok)

    out = cells.copy()
    out["atlas_x"] = atlas_x_list
    out["atlas_y"] = atlas_y_list
    out["segment"] = out[ref_col].astype(str)
    out["segment_frame_index"] = frame_idx_list
    out["atlas_region_id"] = region_ids
    out["region_acronym"] = [id_to_acronym.get(r, None) for r in region_ids]
    out["region_name"] = [id_to_name.get(r, None) for r in region_ids]
    out["lamina_combined_acronym"] = [lamina_lookup.get(r, None) for r in region_ids]
    out["matter_class"] = [matter_lookup.get(r, "other") for r in region_ids]
    out["in_mask_bounds"] = in_bounds
    return out


def summarize_laminar_counts(
    assigned_df: pd.DataFrame,
    *,
    count_column: str = "lamina_combined_acronym",
    include_unassigned: bool = True,
) -> pd.Series:
    """
    Count cells per combined lamina label (Lamina_I … Lamina_X).

    Rows with null count_column are omitted unless include_unassigned is True
    (then counted under '<unassigned>').
    """
    if count_column not in assigned_df.columns:
        raise KeyError(f"Assigned table missing {count_column!r}.")

    labels = assigned_df[count_column].copy()
    if include_unassigned:
        labels = labels.fillna("<unassigned>")
    else:
        labels = labels.dropna()

    counts = labels.value_counts().sort_index()
    return counts


def write_laminar_counts_csv(counts: pd.Series, output_path: PathLike) -> str:
    """Write laminar count Series to a two-column CSV (lamina, n_cells)."""
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = counts.rename("n_cells").reset_index()
    df.columns = ["lamina", "n_cells"]
    df.to_csv(out_path, index=False)
    return str(out_path)


def write_assigned_cells_csv(assigned_df: pd.DataFrame, output_path: PathLike) -> str:
    """Write full per-cell assignment table (original columns plus atlas fields)."""
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    assigned_df.to_csv(out_path, index=False)
    return str(out_path)


def match_cells_and_laminar_counts(
    cell_csv_path: PathLike,
    atlas_dir: PathLike = DEFAULT_ATLAS_DIR,
    *,
    laminar_counts_csv: Optional[PathLike] = None,
    assigned_cells_csv: Optional[PathLike] = None,
    **assign_kwargs: Any,
) -> dict[str, Any]:
    """
    Convenience wrapper: assign regions and return assigned table + laminar counts.

    Returns dict with keys: assigned_df, laminar_counts, laminar_counts_path,
    assigned_cells_path (optional CSV paths when corresponding arguments are set).
    """
    assigned = assign_cells_to_spinalj_regions(
        cell_csv_path, atlas_dir=atlas_dir, **assign_kwargs
    )
    counts = summarize_laminar_counts(assigned)
    result: dict[str, Any] = {
        "assigned_df": assigned,
        "laminar_counts": counts,
        "laminar_counts_path": None,
        "assigned_cells_path": None,
    }
    if laminar_counts_csv is not None:
        result["laminar_counts_path"] = write_laminar_counts_csv(counts, laminar_counts_csv)
    if assigned_cells_csv is not None:
        result["assigned_cells_path"] = write_assigned_cells_csv(assigned, assigned_cells_csv)
    return result


def reference_segments_in_assigned(assigned_df: pd.DataFrame) -> list[str]:
    """Sorted unique reference segment labels from an assignment table."""
    if "segment" not in assigned_df.columns:
        raise KeyError("assigned_df missing 'segment' column.")
    return sorted(assigned_df["segment"].astype(str).unique().tolist())


def output_path_for_reference_segment(output_path: PathLike, segment: str) -> str:
    """Insert _<segment> before the file extension (e.g. counts.csv → counts_L4.csv)."""
    path = Path(output_path)
    return str(path.with_name(f"{path.stem}_{segment}{path.suffix}"))


def output_path_matter_outline_variant(output_path: PathLike) -> str:
    """Insert _matter_outline before the extension (foo.png → foo_matter_outline.png)."""
    path = Path(output_path)
    return str(path.with_name(f"{path.stem}_matter_outline{path.suffix}"))


def output_path_outline_filled_variant(output_path: PathLike) -> str:
    """Insert _outline_filled before the extension (foo.png → foo_outline_filled.png)."""
    path = Path(output_path)
    return str(path.with_name(f"{path.stem}_outline_filled{path.suffix}"))


def resolve_output_path_for_reference_segment(
    output_path: Optional[PathLike],
    segment: str,
) -> Optional[str]:
    """Per-segment output path: always inserts _<segment> before the file extension."""
    if output_path is None:
        return None
    return output_path_for_reference_segment(output_path, segment)


def filter_assigned_to_reference_segment(
    assigned_df: pd.DataFrame,
    segment: str,
) -> pd.DataFrame:
    """Rows for one reference segment only."""
    return assigned_df[assigned_df["segment"].astype(str) == str(segment)].copy()


def resolve_spinalj_output_directory(
    cell_csv_path: PathLike,
    outputdir: Optional[PathLike] = None,
) -> Path:
    """Output folder for segment CSVs/PNGs (default: <cell_csv parent>/output)."""
    if outputdir is not None:
        return Path(outputdir)
    return Path(cell_csv_path).resolve().parent / SPINALJ_DEFAULT_OUTPUT_SUBDIR


def default_spinalj_segment_output_base_paths(output_dir: Path) -> dict[str, str]:
    """Base paths (before _<segment> suffix) for all process_spinalj_by_reference_segment outputs."""
    return {
        "laminar_counts_csv": str(output_dir / OUTPUT_LAMINAR_COUNTS_CSV_NAME),
        "assigned_cells_csv": str(output_dir / OUTPUT_ASSIGNED_CELLS_CSV_NAME),
        "cells_on_atlas_png": str(output_dir / OUTPUT_CELLS_ON_ATLAS_PNG_NAME),
        "cells_on_lamina_color_png": str(output_dir / OUTPUT_CELLS_ON_LAMINA_COLOR_PNG_NAME),
        "laminar_counts_bar_png": str(output_dir / OUTPUT_LAMINAR_COUNTS_BAR_PNG_NAME),
        "cell_kde_on_atlas_png": str(output_dir / OUTPUT_CELL_KDE_ON_ATLAS_PNG_NAME),
        "cell_kde_frac_max_png": str(output_dir / OUTPUT_CELL_KDE_FRAC_MAX_PNG_NAME),
    }


_SPINALJ_PLOT_SETTINGS_INT = frozenset(
    {
        "plot_dpi",
        "kde_contour_percentile_min",
        "kde_contour_percentile_max",
        "kde_contour_percentile_step",
    }
)
_SPINALJ_PLOT_SETTINGS_FLOAT = frozenset(
    {
        "kernel_width_fraction",
        "kde_fraction_max_min",
        "kde_fraction_max_max",
        "kde_fraction_max_step",
        "gm_outline_smooth_sigma_fraction",
        "wm_outline_smooth_sigma_fraction",
    }
)
_SPINALJ_PLOT_SETTINGS_BOOL = frozenset(
    {"show_contour_inline_labels", "plot_lamina_roman_labels"}
)


def _spinalj_plot_settings_default_display(value: Any, *, param: str) -> str:
    if param in _SPINALJ_PLOT_SETTINGS_BOOL:
        return "TRUE" if value else "FALSE"
    return str(value)


def spinalj_plot_settings_rows() -> list[dict[str, str]]:
    """Rows for spinalj_plot_settings.csv (Parameter, Default, User_value, Description)."""
    spec: list[tuple[str, Any, str]] = [
        ("plot_dpi", DEFAULT_PLOT_DPI, "Matplotlib savefig DPI for all PNG outputs"),
        (
            "kernel_width_fraction",
            DEFAULT_KERNEL_WIDTH_FRACTION,
            "KDE Gaussian sigma as fraction of GM vertical height (pixels)",
        ),
        (
            "kde_contour_percentile_min",
            DEFAULT_KDE_CONTOUR_PERCENTILE_MIN,
            "Lowest percentile for KDE percentile contour levels",
        ),
        (
            "kde_contour_percentile_max",
            DEFAULT_KDE_CONTOUR_PERCENTILE_MAX,
            "Highest percentile for KDE percentile contour levels",
        ),
        (
            "kde_contour_percentile_step",
            DEFAULT_KDE_CONTOUR_PERCENTILE_STEP,
            "Step between KDE percentile contour levels",
        ),
        (
            "kde_fraction_max_min",
            DEFAULT_KDE_FRACTION_MAX_MIN,
            "Lowest fraction-of-max KDE contour level",
        ),
        (
            "kde_fraction_max_max",
            DEFAULT_KDE_FRACTION_MAX_MAX,
            "Highest fraction-of-max KDE contour level",
        ),
        (
            "kde_fraction_max_step",
            DEFAULT_KDE_FRACTION_MAX_STEP,
            "Step between fraction-of-max KDE contour levels",
        ),
        (
            "show_contour_inline_labels",
            False,
            "Draw labels on KDE contour lines (TRUE/FALSE)",
        ),
        (
            "plot_lamina_roman_labels",
            True,
            "Roman I-X on lamina color scatter backdrops (TRUE/FALSE)",
        ),
        (
            "gm_outline_smooth_sigma_fraction",
            DEFAULT_GM_OUTLINE_SMOOTH_SIGMA_FRACTION,
            "GM outline sigma as fraction of GM vertical height",
        ),
        (
            "wm_outline_smooth_sigma_fraction",
            DEFAULT_WM_OUTLINE_SMOOTH_SIGMA_FRACTION,
            "WM outline sigma as fraction of GM vertical height (WM dilation before smooth)",
        ),
    ]
    rows: list[dict[str, str]] = []
    for param, default_val, description in spec:
        rows.append(
            {
                "Parameter": param,
                "Default": _spinalj_plot_settings_default_display(
                    default_val, param=param
                ),
                "User_value": "",
                "Description": description,
            }
        )
    return rows


def resolve_spinalj_plot_settings_csv_path(
    cell_csv_path: PathLike,
    plot_settings_csv: Optional[PathLike] = None,
) -> Path:
    """Path to plot settings CSV (default: same folder as cell_csv_path)."""
    if plot_settings_csv is not None:
        return Path(plot_settings_csv)
    return Path(cell_csv_path).resolve().parent / SPINALJ_PLOT_SETTINGS_CSV_NAME


def write_spinalj_plot_settings_csv(
    output_path: PathLike,
    *,
    overwrite: bool = False,
) -> str:
    """Write template CSV (Parameter, Default, User_value, Description); bools as TRUE/FALSE."""
    out_path = Path(output_path)
    if out_path.exists() and not overwrite:
        return str(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(spinalj_plot_settings_rows())
    df = df[list(SPINALJ_PLOT_SETTINGS_CSV_COLUMNS)]
    df.to_csv(out_path, index=False)
    return str(out_path)


def _normalize_spinalj_plot_settings_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map Parameter/Default/User_value/Description (any casing) to lowercase internal names."""
    rename: dict[str, str] = {}
    for col in df.columns:
        key = str(col).strip().lower().replace(" ", "_")
        if key == "parameter":
            rename[col] = "parameter"
        elif key == "default":
            rename[col] = "default"
        elif key in ("user_value", "user_val"):
            rename[col] = "user_value"
        elif key == "description":
            rename[col] = "description"
    out = df.rename(columns=rename)
    required = {"parameter", "default", "user_value"}
    if not required.issubset(out.columns):
        raise ValueError(
            "Plot settings CSV must have Parameter, Default, and User_value columns "
            f"(optional Description). Got: {list(df.columns)}."
        )
    return out


def _coerce_spinalj_plot_setting(name: str, raw: str) -> Any:
    value = raw.strip()
    if name in _SPINALJ_PLOT_SETTINGS_BOOL:
        lower = value.lower()
        if lower in ("true", "1", "yes", "y"):
            return True
        if lower in ("false", "0", "no", "n"):
            return False
        raise ValueError(f"Plot setting {name!r} must be TRUE/FALSE, got {raw!r}.")
    if name in _SPINALJ_PLOT_SETTINGS_INT:
        return int(float(value))
    if name in _SPINALJ_PLOT_SETTINGS_FLOAT:
        return float(value)
    raise ValueError(f"Unknown plot setting parameter {name!r}.")


def load_spinalj_plot_settings(plot_settings_csv: PathLike) -> dict[str, Any]:
    """Load plot kwargs from settings CSV (User_value overrides Default when non-empty)."""
    path = Path(plot_settings_csv)
    if not path.is_file():
        raise FileNotFoundError(f"Plot settings CSV not found: {path}")
    df = _normalize_spinalj_plot_settings_columns(pd.read_csv(path))
    settings: dict[str, Any] = {}
    for _, row in df.iterrows():
        key = str(row["parameter"]).strip()
        if not key or key.startswith("#"):
            continue
        user_raw = row["user_value"]
        use_user = pd.notna(user_raw) and str(user_raw).strip() != ""
        raw = str(user_raw if use_user else row["default"]).strip()
        settings[key] = _coerce_spinalj_plot_setting(key, raw)
    expected = {r["Parameter"] for r in spinalj_plot_settings_rows()}
    missing = expected - set(settings)
    if missing:
        raise ValueError(f"Plot settings CSV missing parameters: {sorted(missing)}.")
    return settings


def process_spinalj_by_reference_segment(
    cell_csv_path: PathLike,
    atlas_dir: PathLike,
    *,
    outputdir: Optional[PathLike] = None,
    plot_settings_csv: Optional[PathLike] = None,
    **assign_kwargs: Any,
) -> dict[str, Any]:
    """
    Assign all cells, then write CSVs and plots separately for each reference segment.

    atlas_dir must be supplied explicitly (there is no default atlas path in this entrypoint).

    Plot parameters (DPI, KDE, outlines, lamina labels) are read from plot_settings_csv;
    if that file is missing it is created beside the cell CSV with module defaults.

    Returns dict with assigned_df (all rows), reference_segments, output_dir, plot_settings_csv,
    and by_segment mapping each segment to laminar counts, paths, and plot result metadata.
    """
    validate_spinalj_atlas_dir(atlas_dir)
    settings_path = resolve_spinalj_plot_settings_csv_path(cell_csv_path, plot_settings_csv)
    write_spinalj_plot_settings_csv(settings_path, overwrite=False)
    plot_settings = load_spinalj_plot_settings(settings_path)
    plot_dpi = int(plot_settings["plot_dpi"])
    kernel_width_fraction = float(plot_settings["kernel_width_fraction"])
    kde_contour_percentile_min = int(plot_settings["kde_contour_percentile_min"])
    kde_contour_percentile_max = int(plot_settings["kde_contour_percentile_max"])
    kde_contour_percentile_step = int(plot_settings["kde_contour_percentile_step"])
    kde_fraction_max_min = float(plot_settings["kde_fraction_max_min"])
    kde_fraction_max_max = float(plot_settings["kde_fraction_max_max"])
    kde_fraction_max_step = float(plot_settings["kde_fraction_max_step"])
    show_contour_inline_labels = bool(plot_settings["show_contour_inline_labels"])
    plot_lamina_roman_labels = bool(plot_settings["plot_lamina_roman_labels"])
    gm_outline_smooth_sigma_fraction = float(plot_settings["gm_outline_smooth_sigma_fraction"])
    wm_outline_smooth_sigma_fraction = float(plot_settings["wm_outline_smooth_sigma_fraction"])

    assigned = assign_cells_to_spinalj_regions(
        cell_csv_path, atlas_dir=atlas_dir, **assign_kwargs
    )
    segments = reference_segments_in_assigned(assigned)
    out_dir = resolve_spinalj_output_directory(cell_csv_path, outputdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base_paths = default_spinalj_segment_output_base_paths(out_dir)
    laminar_counts_csv = base_paths["laminar_counts_csv"]
    assigned_cells_csv = base_paths["assigned_cells_csv"]
    cells_on_atlas_png = base_paths["cells_on_atlas_png"]
    cells_on_lamina_color_png = base_paths["cells_on_lamina_color_png"]
    laminar_counts_bar_png = base_paths["laminar_counts_bar_png"]
    cell_kde_on_atlas_png = base_paths["cell_kde_on_atlas_png"]
    cell_kde_frac_max_png = base_paths["cell_kde_frac_max_png"]
    by_segment: dict[str, Any] = {}
    outline_kw = {
        "gm_outline_smooth_sigma_fraction": gm_outline_smooth_sigma_fraction,
        "wm_outline_smooth_sigma_fraction": wm_outline_smooth_sigma_fraction,
    }

    for segment in segments:
        segment_df = filter_assigned_to_reference_segment(assigned, segment)
        laminar_path = resolve_output_path_for_reference_segment(
            laminar_counts_csv, segment
        )
        assigned_path = resolve_output_path_for_reference_segment(
            assigned_cells_csv, segment
        )
        counts = summarize_laminar_counts(segment_df)
        laminar_counts_path = None
        assigned_cells_path = None
        if laminar_path is not None:
            laminar_counts_path = write_laminar_counts_csv(counts, laminar_path)
        if assigned_path is not None:
            assigned_cells_path = write_assigned_cells_csv(segment_df, assigned_path)

        n_cells = len(segment_df)
        bar_png = resolve_output_path_for_reference_segment(
            laminar_counts_bar_png, segment
        )
        scatter_png = resolve_output_path_for_reference_segment(
            cells_on_atlas_png, segment
        )
        scatter_outline_png = resolve_output_path_for_reference_segment(
            output_path_matter_outline_variant(cells_on_atlas_png), segment
        )
        scatter_filled_png = resolve_output_path_for_reference_segment(
            output_path_outline_filled_variant(cells_on_atlas_png), segment
        )
        lamina_color_png = resolve_output_path_for_reference_segment(
            cells_on_lamina_color_png, segment
        )
        lamina_color_outline_png = resolve_output_path_for_reference_segment(
            output_path_matter_outline_variant(cells_on_lamina_color_png), segment
        )
        kde_pct_png = resolve_output_path_for_reference_segment(
            cell_kde_on_atlas_png, segment
        )
        kde_pct_outline_png = resolve_output_path_for_reference_segment(
            output_path_matter_outline_variant(cell_kde_on_atlas_png), segment
        )
        kde_pct_filled_png = resolve_output_path_for_reference_segment(
            output_path_outline_filled_variant(cell_kde_on_atlas_png), segment
        )
        kde_frac_png = resolve_output_path_for_reference_segment(
            cell_kde_frac_max_png, segment
        )
        kde_frac_outline_png = resolve_output_path_for_reference_segment(
            output_path_matter_outline_variant(cell_kde_frac_max_png), segment
        )
        kde_frac_filled_png = resolve_output_path_for_reference_segment(
            output_path_outline_filled_variant(cell_kde_frac_max_png), segment
        )

        bar_result = plot_laminar_counts_bar(
            counts,
            bar_png,
            total_cells=n_cells,
            dpi=plot_dpi,
            title=f"Combined lamina cell percentages — {segment}",
        )
        scatter_result = plot_cells_on_spinalj_matter_backdrop(
            segment_df,
            scatter_png,
            atlas_dir=atlas_dir,
            segment=segment,
            dpi=plot_dpi,
            draw_matter_outlines=False,
            **outline_kw,
        )
        scatter_outline_result = plot_cells_on_spinalj_matter_backdrop(
            segment_df,
            scatter_outline_png,
            atlas_dir=atlas_dir,
            segment=segment,
            dpi=plot_dpi,
            draw_matter_outlines=True,
            **outline_kw,
        )
        scatter_filled_result = plot_cells_on_spinalj_matter_backdrop(
            segment_df,
            scatter_filled_png,
            atlas_dir=atlas_dir,
            segment=segment,
            dpi=plot_dpi,
            matter_backdrop_mode=MATTER_BACKDROP_OUTLINE_FILLED,
            draw_matter_outlines=False,
            **outline_kw,
        )
        lamina_color_result = plot_cells_on_lamina_color_backdrop(
            segment_df,
            lamina_color_png,
            atlas_dir=atlas_dir,
            segment=segment,
            dpi=plot_dpi,
            plot_lamina_roman_labels=plot_lamina_roman_labels,
            draw_matter_outlines=False,
            **outline_kw,
        )
        lamina_color_outline_result = plot_cells_on_lamina_color_backdrop(
            segment_df,
            lamina_color_outline_png,
            atlas_dir=atlas_dir,
            segment=segment,
            dpi=plot_dpi,
            plot_lamina_roman_labels=plot_lamina_roman_labels,
            draw_matter_outlines=True,
            **outline_kw,
        )
        kde_pct_result = plot_cell_kde_on_spinalj_backdrop(
            segment_df,
            kde_pct_png,
            atlas_dir=atlas_dir,
            segment=segment,
            dpi=plot_dpi,
            kernel_width_fraction=kernel_width_fraction,
            kde_contour_percentile_min=kde_contour_percentile_min,
            kde_contour_percentile_max=kde_contour_percentile_max,
            kde_contour_percentile_step=kde_contour_percentile_step,
            show_contour_inline_labels=show_contour_inline_labels,
            draw_matter_outlines=False,
            **outline_kw,
        )
        kde_pct_outline_result = plot_cell_kde_on_spinalj_backdrop(
            segment_df,
            kde_pct_outline_png,
            atlas_dir=atlas_dir,
            segment=segment,
            dpi=plot_dpi,
            kernel_width_fraction=kernel_width_fraction,
            kde_contour_percentile_min=kde_contour_percentile_min,
            kde_contour_percentile_max=kde_contour_percentile_max,
            kde_contour_percentile_step=kde_contour_percentile_step,
            show_contour_inline_labels=show_contour_inline_labels,
            draw_matter_outlines=True,
            **outline_kw,
        )
        kde_pct_filled_result = plot_cell_kde_on_spinalj_backdrop(
            segment_df,
            kde_pct_filled_png,
            atlas_dir=atlas_dir,
            segment=segment,
            dpi=plot_dpi,
            kernel_width_fraction=kernel_width_fraction,
            kde_contour_percentile_min=kde_contour_percentile_min,
            kde_contour_percentile_max=kde_contour_percentile_max,
            kde_contour_percentile_step=kde_contour_percentile_step,
            show_contour_inline_labels=show_contour_inline_labels,
            matter_backdrop_mode=MATTER_BACKDROP_OUTLINE_FILLED,
            draw_matter_outlines=False,
            **outline_kw,
        )
        kde_frac_result = plot_cell_kde_fraction_of_max_on_spinalj_backdrop(
            segment_df,
            kde_frac_png,
            atlas_dir=atlas_dir,
            segment=segment,
            dpi=plot_dpi,
            kernel_width_fraction=kernel_width_fraction,
            kde_fraction_max_min=kde_fraction_max_min,
            kde_fraction_max_max=kde_fraction_max_max,
            kde_fraction_max_step=kde_fraction_max_step,
            show_contour_inline_labels=show_contour_inline_labels,
            draw_matter_outlines=False,
            **outline_kw,
        )
        kde_frac_outline_result = plot_cell_kde_fraction_of_max_on_spinalj_backdrop(
            segment_df,
            kde_frac_outline_png,
            atlas_dir=atlas_dir,
            segment=segment,
            dpi=plot_dpi,
            kernel_width_fraction=kernel_width_fraction,
            kde_fraction_max_min=kde_fraction_max_min,
            kde_fraction_max_max=kde_fraction_max_max,
            kde_fraction_max_step=kde_fraction_max_step,
            show_contour_inline_labels=show_contour_inline_labels,
            draw_matter_outlines=True,
            **outline_kw,
        )
        kde_frac_filled_result = plot_cell_kde_fraction_of_max_on_spinalj_backdrop(
            segment_df,
            kde_frac_filled_png,
            atlas_dir=atlas_dir,
            segment=segment,
            dpi=plot_dpi,
            kernel_width_fraction=kernel_width_fraction,
            kde_fraction_max_min=kde_fraction_max_min,
            kde_fraction_max_max=kde_fraction_max_max,
            kde_fraction_max_step=kde_fraction_max_step,
            show_contour_inline_labels=show_contour_inline_labels,
            matter_backdrop_mode=MATTER_BACKDROP_OUTLINE_FILLED,
            draw_matter_outlines=False,
            **outline_kw,
        )

        by_segment[segment] = {
            "segment": segment,
            "assigned_df": segment_df,
            "laminar_counts": counts,
            "laminar_counts_path": laminar_counts_path,
            "assigned_cells_path": assigned_cells_path,
            "bar_plot": bar_result,
            "scatter_plot": scatter_result,
            "scatter_matter_outline_plot": scatter_outline_result,
            "scatter_outline_filled_plot": scatter_filled_result,
            "lamina_color_plot": lamina_color_result,
            "lamina_color_matter_outline_plot": lamina_color_outline_result,
            "kde_percentile_plot": kde_pct_result,
            "kde_percentile_matter_outline_plot": kde_pct_outline_result,
            "kde_percentile_outline_filled_plot": kde_pct_filled_result,
            "kde_fraction_max_plot": kde_frac_result,
            "kde_fraction_max_matter_outline_plot": kde_frac_outline_result,
            "kde_fraction_max_outline_filled_plot": kde_frac_filled_result,
            "n_cells": n_cells,
        }

    return {
        "assigned_df": assigned,
        "reference_segments": segments,
        "output_dir": str(out_dir),
        "plot_settings_csv": str(settings_path),
        "plot_settings": plot_settings,
        "by_segment": by_segment,
    }


def _load_atlas_matter_lookup(atlas_dir: PathLike) -> pd.DataFrame:
    atlas_path = Path(atlas_dir)
    return pd.read_csv(atlas_path / ATLAS_REGIONS_FILENAME)


def _load_segment_region_mask(
    atlas_dir: PathLike,
    segment_label: str,
) -> tuple[pd.DataFrame, np.ndarray, int]:
    """Return (segments_df, 2-D region-id mask, frame index) for one segment."""
    segments, mask_stack = load_spinalj_segment_masks(atlas_dir)
    frame_idx = segment_frame_index_for_label(segments, segment_label)
    return segments, mask_stack[frame_idx], frame_idx


def _resolve_segment_label(assigned_df: pd.DataFrame, segment: Optional[str]) -> str:
    if segment is not None:
        return str(segment)
    if "segment" not in assigned_df.columns:
        raise KeyError("assigned_df missing 'segment' column.")
    labels = assigned_df["segment"].astype(str).unique().tolist()
    if len(labels) != 1:
        raise ValueError(
            f"assigned_df has multiple segments {labels}; pass segment= explicitly."
        )
    return labels[0]


def build_matter_backdrop_rgb(
    region_id_mask: np.ndarray,
    matter_lookup: dict[int, str],
    *,
    gm_color: RGB = DEFAULT_GM_RGB,
    wm_color: RGB = DEFAULT_WM_RGB,
    other_color: RGB = DEFAULT_OTHER_RGB,
) -> np.ndarray:
    """
    RGB image (height, width, 3) float32 in [0, 1] from atlas region-id mask.

    GM → gm_color, WM → wm_color, all other ids (including 0) → other_color.
    """
    if region_id_mask.ndim != 2:
        raise ValueError("region_id_mask must be 2-D (height, width).")

    class_colors: dict[str, RGB] = {
        "GM": gm_color,
        "WM": wm_color,
        "other": other_color,
    }
    height, width = region_id_mask.shape
    rgb = np.zeros((height, width, 3), dtype=np.float32)
    ids = np.unique(region_id_mask)
    for rid in ids:
        rid_int = int(rid)
        matter = matter_lookup.get(rid_int, "other")
        color = class_colors.get(matter, other_color)
        rgb[region_id_mask == rid] = color
    return rgb


def _gm_wm_smoothed_fields_full_cord(
    region_mask: np.ndarray,
    matter_lookup: dict[int, str],
    *,
    gm_sigma_pixels: float,
    wm_sigma_pixels: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Smoothed GM and WM fields used for outlines and outline-filled backdrops."""
    gm_mask = build_matter_class_binary_mask(region_mask, matter_lookup, "GM")
    wm_mask = build_matter_class_binary_mask(region_mask, matter_lookup, "WM")
    gm_field = ndi.gaussian_filter(
        gm_mask.astype(np.float32), sigma=float(gm_sigma_pixels), mode="constant"
    )
    wm_field = _wm_enclosing_smoothed_field(
        wm_mask, smooth_sigma_pixels=float(wm_sigma_pixels)
    )
    return gm_field, wm_field


def draw_outline_filled_matter_patches(
    ax: plt.Axes,
    region_mask: np.ndarray,
    matter_lookup: dict[int, str],
    width: int,
    height: int,
    *,
    gm_outline_smooth_sigma_fraction: float = DEFAULT_GM_OUTLINE_SMOOTH_SIGMA_FRACTION,
    wm_outline_smooth_sigma_fraction: float = DEFAULT_WM_OUTLINE_SMOOTH_SIGMA_FRACTION,
    gm_color: RGB = DEFAULT_GM_RGB,
    wm_color: RGB = DEFAULT_WM_RGB,
    other_color: RGB = DEFAULT_OTHER_RGB,
) -> None:
    """
    Fill right-hemisphere GM/WM as smooth patches from smoothed-mask 0.5 contours (contourf).

    Not atlas region IDs. WM is drawn first; GM contourf overwrites where both exceed 0.5.
    """
    gm_sigma = matter_outline_smooth_sigma_pixels(
        region_mask, matter_lookup, gm_outline_smooth_sigma_fraction
    )
    wm_sigma = matter_outline_smooth_sigma_pixels(
        region_mask, matter_lookup, wm_outline_smooth_sigma_fraction
    )
    gm_field, wm_field = _gm_wm_smoothed_fields_full_cord(
        region_mask,
        matter_lookup,
        gm_sigma_pixels=gm_sigma,
        wm_sigma_pixels=wm_sigma,
    )
    mid = spinal_cord_midline_col(region_mask.shape[1])
    gm_rh = gm_field[:, mid:]
    wm_rh = wm_field[:, mid:]
    extent = (0.0, float(width), float(height), 0.0)
    ax.set_facecolor(other_color)
    wm_fill = wm_rh.copy()
    wm_fill[gm_rh > 0.5] = 0.0
    ax.contourf(
        wm_fill,
        levels=[0.5, 1.01],
        colors=[wm_color],
        origin="upper",
        extent=extent,
        antialiased=True,
        zorder=1,
    )
    ax.contourf(
        gm_rh,
        levels=[0.5, 1.01],
        colors=[gm_color],
        origin="upper",
        extent=extent,
        antialiased=True,
        zorder=2,
    )


def build_matter_class_binary_mask(
    region_id_mask: np.ndarray,
    matter_lookup: dict[int, str],
    matter_class: str,
) -> np.ndarray:
    """True where matter_lookup classifies the pixel as matter_class (e.g. GM, WM)."""
    mask = np.zeros(region_id_mask.shape, dtype=bool)
    for rid in np.unique(region_id_mask):
        if matter_lookup.get(int(rid), "other") == matter_class:
            mask |= region_id_mask == rid
    return mask


def build_gray_matter_binary_mask(
    region_id_mask: np.ndarray,
    matter_lookup: dict[int, str],
) -> np.ndarray:
    """True where matter_lookup classifies the pixel as gray matter."""
    return build_matter_class_binary_mask(region_id_mask, matter_lookup, "GM")


def spinal_cord_midline_col(full_width: int) -> int:
    """Column index of the sagittal midline (right half starts here)."""
    if full_width < 2:
        raise ValueError("full_width must be at least 2.")
    return int(full_width // 2)


def fold_atlas_x_to_right_hemisphere_plot_coords(
    atlas_x: float,
    full_width: int,
) -> float:
    """
    Fold atlas x into right-hemisphere plot coordinates.

    Cells with x left of the midline are mirrored: x → (full_width - 1) - x.
    All coordinates are then shifted so the midline is 0 (right-half crop only).
    """
    mid = spinal_cord_midline_col(full_width)
    x = float(atlas_x)
    if x < mid:
        x = (full_width - 1) - x
    return x - mid


def crop_to_right_hemisphere(region_id_mask: np.ndarray) -> np.ndarray:
    """Return region-id mask cropped to columns at/ right of the midline."""
    mid = spinal_cord_midline_col(region_id_mask.shape[1])
    return region_id_mask[:, mid:].copy()


def crop_rgb_to_right_hemisphere(rgb: np.ndarray) -> np.ndarray:
    """Crop an (H, W, 3) RGB image to the right half at the midline."""
    mid = spinal_cord_midline_col(rgb.shape[1])
    return rgb[:, mid:, :].copy()


def representative_z_index_1based_for_segment(
    segments: pd.DataFrame,
    segment_label: str,
) -> int:
    """Midpoint Z index (1-based, Segments.csv Start/End) for a reference segment."""
    col = "Segment"
    matches = segments.index[segments[col].astype(str) == str(segment_label)]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one Segments.csv row for {segment_label!r}, found {len(matches)}."
        )
    row = segments.loc[matches[0]]
    start = int(row["Start"])
    end = int(row["End"])
    return (start + end) // 2


def load_annotation_color_rgb_for_segment(
    atlas_dir: PathLike,
    segment_label: str,
) -> tuple[np.ndarray, int]:
    """
    Load RGB lamina color backdrop for a segment from Annotation_Color.tif.

    Uses the midpoint Z plane between Segments.csv Start and End (1-based). TIFF stores
    three grayscale pages per Z (R, G, B).
    """
    atlas_path = Path(atlas_dir)
    segments = pd.read_csv(atlas_path / SEGMENTS_FILENAME)
    segments.columns = [str(c).strip() for c in segments.columns]
    z_1based = representative_z_index_1based_for_segment(segments, segment_label)
    z_0based = z_1based - 1
    frame_base = z_0based * ANNOTATION_COLOR_CHANNELS_PER_Z

    tif_path = atlas_path / ANNOTATION_COLOR_FILENAME
    with Image.open(tif_path) as im:
        n_frames = getattr(im, "n_frames", 1)
        if frame_base + ANNOTATION_COLOR_CHANNELS_PER_Z - 1 >= n_frames:
            raise ValueError(
                f"Segment {segment_label!r} Z index {z_1based} requires TIFF frames "
                f"{frame_base}–{frame_base + 2} but {ANNOTATION_COLOR_FILENAME} has "
                f"{n_frames} pages."
            )
        planes: list[np.ndarray] = []
        for offset in range(ANNOTATION_COLOR_CHANNELS_PER_Z):
            im.seek(frame_base + offset)
            planes.append(np.array(im))
    rgb = np.stack(planes, axis=-1)
    return rgb.astype(np.uint8), z_1based


def combined_lamina_separable_colors() -> dict[str, RGB]:
    """Distinct colors for Lamina_I–X (matplotlib tab10, one color per combined lamina)."""
    cmap = plt.get_cmap("tab10")
    colors: dict[str, RGB] = {}
    for i, lam_acr in enumerate(COMBINED_LAMINA_ORDER):
        rgba = cmap(i % 10)
        colors[lam_acr] = (float(rgba[0]), float(rgba[1]), float(rgba[2]))
    return colors


def build_combined_lamina_separable_backdrop_rgb(
    region_id_mask: np.ndarray,
    atlas_regions: pd.DataFrame,
    *,
    other_color: RGB = DEFAULT_OTHER_RGB,
) -> np.ndarray:
    """
    RGB backdrop: each fine region id colored by its combined lamina (I–X).

    Unmapped ids (0, white matter, etc.) use other_color (default light gray).
    """
    id_to_lamina = _build_lamina_combined_lookup(atlas_regions)
    lamina_colors = combined_lamina_separable_colors()
    height, width = region_id_mask.shape
    rgb = np.tile(np.array(other_color, dtype=np.float32), (height, width, 1))
    for rid in np.unique(region_id_mask):
        rid_int = int(rid)
        if rid_int == 0:
            continue
        lam_acr = id_to_lamina.get(rid_int)
        if lam_acr is None:
            continue
        rgb[region_id_mask == rid] = lamina_colors.get(lam_acr, other_color)
    return rgb


def _lamina_combined_child_region_ids(atlas_regions: pd.DataFrame) -> dict[str, list[int]]:
    """Map combined lamina acronym → fine atlas region ids (children_IDs)."""
    mapping: dict[str, list[int]] = {}
    combined = atlas_regions[
        atlas_regions["acronym"].astype(str).str.startswith("Lamina_", na=False)
    ]
    for _, row in combined.iterrows():
        acr = str(row["acronym"])
        child_field = row["children_IDs"]
        if pd.isna(child_field) or str(child_field).strip() == "":
            mapping[acr] = []
            continue
        mapping[acr] = [int(p.strip()) for p in str(child_field).split(",") if p.strip()]
    return mapping


def lamina_combined_acronym_to_roman(lamina_acronym: str) -> str:
    """Lamina_I → I, Lamina_X → X, etc."""
    prefix = "Lamina_"
    if lamina_acronym.startswith(prefix):
        return lamina_acronym[len(prefix) :]
    return lamina_acronym


def compute_lamina_label_positions_centroid_x(
    region_id_mask: np.ndarray,
    atlas_regions: pd.DataFrame,
) -> dict[str, tuple[float, float]]:
    """
    Label position per combined lamina in right-hemisphere plot coordinates.

    x is the mean x of all lamina mask pixels (centroid x, midline-folded like cells).
    y is the median row among lamina pixels whose x is closest to that centroid x, so
    the label sits on the mask at the centroid column.
    """
    _, full_width = region_id_mask.shape
    child_map = _lamina_combined_child_region_ids(atlas_regions)
    positions: dict[str, tuple[float, float]] = {}

    for lam_acr in COMBINED_LAMINA_ORDER:
        child_ids = child_map.get(lam_acr, [])
        if not child_ids:
            continue
        in_lamina = np.isin(region_id_mask, child_ids)
        if not in_lamina.any():
            continue
        ys, xs = np.where(in_lamina)
        x_plot = np.array(
            [
                fold_atlas_x_to_right_hemisphere_plot_coords(float(x), full_width)
                for x in xs
            ],
            dtype=float,
        )
        y_plot = ys.astype(float)
        cx = float(np.mean(x_plot))
        x_nearest = float(np.min(np.abs(x_plot - cx)))
        at_centroid_column = np.abs(x_plot - cx) <= x_nearest + 1e-9
        y_label = float(np.median(y_plot[at_centroid_column]))
        positions[lam_acr] = (cx, y_label)

    return positions


def draw_lamina_roman_labels(
    ax: plt.Axes,
    label_positions: dict[str, tuple[float, float]],
    *,
    fontsize: float = LAMINA_ROMAN_LABEL_FONTSIZE,
) -> None:
    """Draw roman-numeral lamina labels at the given plot coordinates."""
    stroke = pe.withStroke(linewidth=2.5, foreground="white")
    for lam_acr in COMBINED_LAMINA_ORDER:
        if lam_acr not in label_positions:
            continue
        x_c, y_c = label_positions[lam_acr]
        roman = lamina_combined_acronym_to_roman(lam_acr)
        ax.text(
            x_c,
            y_c,
            roman,
            ha="center",
            va="center",
            fontsize=fontsize,
            color="black",
            fontweight="bold",
            zorder=4,
            path_effects=[stroke],
        )


def fold_cells_xy_for_right_hemisphere_plot(
    atlas_x: np.ndarray,
    atlas_y: np.ndarray,
    full_width: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply midline fold to x; y unchanged."""
    x_plot = np.array(
        [fold_atlas_x_to_right_hemisphere_plot_coords(x, full_width) for x in atlas_x],
        dtype=float,
    )
    y_plot = np.asarray(atlas_y, dtype=float)
    return x_plot, y_plot


def gray_matter_vertical_height_pixels(
    region_id_mask: np.ndarray,
    matter_lookup: dict[int, str],
) -> int:
    """
    Vertical extent (rows) of gray matter in the segment mask.

    Computed as (max row index − min row index + 1) over all GM-classified pixels.
    """
    gm_mask = np.zeros(region_id_mask.shape, dtype=bool)
    for rid in np.unique(region_id_mask):
        if matter_lookup.get(int(rid), "other") == "GM":
            gm_mask |= region_id_mask == rid
    if not gm_mask.any():
        raise ValueError("No gray-matter pixels found in segment mask.")
    ys = np.where(gm_mask)[0]
    return int(ys.max() - ys.min() + 1)


def matter_outline_smooth_sigma_pixels(
    region_id_mask: np.ndarray,
    matter_lookup: dict[int, str],
    smooth_sigma_fraction: float = DEFAULT_MATTER_OUTLINE_SMOOTH_SIGMA_FRACTION,
) -> float:
    """Gaussian σ (pixels) for smoothing GM/WM masks before outline contours."""
    gm_height = gray_matter_vertical_height_pixels(region_id_mask, matter_lookup)
    return float(gaussian_kernel_sigma_pixels(gm_height, smooth_sigma_fraction))


def gray_matter_outline_smooth_sigma_pixels(
    region_id_mask: np.ndarray,
    matter_lookup: dict[int, str],
    smooth_sigma_fraction: float = DEFAULT_GM_OUTLINE_SMOOTH_SIGMA_FRACTION,
) -> float:
    """Alias for matter_outline_smooth_sigma_pixels."""
    return matter_outline_smooth_sigma_pixels(
        region_id_mask, matter_lookup, smooth_sigma_fraction
    )


def _contour_segments_at_level(
    field: np.ndarray,
    level: float,
    *,
    extent_x_max: float,
    extent_y_max: float,
) -> list[np.ndarray]:
    """Contour lines for a 2-D field in upper-origin atlas coordinates (x, y)."""
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cs = ax.contour(
        field,
        levels=[level],
        origin="upper",
        extent=(0.0, extent_x_max, extent_y_max, 0.0),
    )
    segments = list(cs.allsegs[0]) if cs.allsegs else []
    plt.close(fig)
    return segments


def _clip_polyline_to_x_min(
    seg: np.ndarray,
    x_min: float,
) -> list[np.ndarray]:
    """Keep portions of a contour polyline with x >= x_min (atlas x)."""
    if seg.ndim != 2 or seg.shape[1] != 2 or len(seg) < 2:
        return []
    clipped_parts: list[np.ndarray] = []
    current: list[tuple[float, float]] = []

    def _append_cross(x0: float, y0: float, x1: float, y1: float) -> tuple[float, float]:
        if abs(x1 - x0) < 1e-12:
            return (x_min, y0)
        t = (x_min - x0) / (x1 - x0)
        t = float(np.clip(t, 0.0, 1.0))
        return (x_min, y0 + t * (y1 - y0))

    for i in range(len(seg)):
        x, y = float(seg[i, 0]), float(seg[i, 1])
        inside = x >= x_min
        if i == 0:
            if inside:
                current.append((x, y))
            continue
        x0, y0 = float(seg[i - 1, 0]), float(seg[i - 1, 1])
        in0 = x0 >= x_min
        if in0 and inside:
            current.append((x, y))
        elif in0 and not inside:
            current.append(_append_cross(x0, y0, x, y))
            if len(current) >= 2:
                clipped_parts.append(np.asarray(current, dtype=float))
            current = []
        elif not in0 and inside:
            current = [_append_cross(x0, y0, x, y), (x, y)]
        # both outside: leave current unchanged

    if len(current) >= 2:
        clipped_parts.append(np.asarray(current, dtype=float))
    return clipped_parts


def clip_contour_segments_to_right_hemisphere(
    segments: list[np.ndarray],
    full_width: int,
) -> list[np.ndarray]:
    """Clip full-cord contour segments to columns at/ right of the sagittal midline."""
    x_min = float(spinal_cord_midline_col(full_width))
    clipped: list[np.ndarray] = []
    for seg in segments:
        clipped.extend(_clip_polyline_to_x_min(np.asarray(seg, dtype=float), x_min))
    return clipped


def _closed_polygon_area(seg: np.ndarray) -> float:
    """Shoelace area for a (possibly open) contour vertex list."""
    if len(seg) < 3:
        return 0.0
    x = seg[:, 0].astype(float)
    y = seg[:, 1].astype(float)
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _segment_is_closed_loop(seg: np.ndarray, *, tol: float = 1.0) -> bool:
    if len(seg) < 4:
        return False
    return float(np.hypot(seg[0, 0] - seg[-1, 0], seg[0, 1] - seg[-1, 1])) <= tol


def _filter_wm_outline_segments_exterior_only(
    segments: list[np.ndarray],
) -> list[np.ndarray]:
    """
    Drop inner WM 0.5 contours that trace the GM–WM interface (gray ring around GM).

    Keeps open arcs (e.g. after midline clip) and the largest closed loop(s) only.
    """
    closed: list[tuple[float, np.ndarray]] = []
    open_parts: list[np.ndarray] = []
    for seg in segments:
        arr = np.asarray(seg, dtype=float)
        if len(arr) < 2:
            continue
        area = _closed_polygon_area(arr)
        if _segment_is_closed_loop(arr) and area > 1.0:
            closed.append((area, arr))
        else:
            open_parts.append(arr)
    if not closed:
        return [np.asarray(s, dtype=float) for s in segments if len(s) >= 2]
    max_area = max(a for a, _ in closed)
    kept_closed = [seg for area, seg in closed if area >= max_area * 0.9]
    return open_parts + kept_closed


def _plot_right_hemisphere_contour_segments(
    ax: plt.Axes,
    segments: list[np.ndarray],
    full_width: int,
    *,
    outline_color: str,
    outline_linewidth: float,
    zorder: float,
) -> None:
    """Clip contour segments to the RH crop and plot in midline-shifted coordinates."""
    mid = spinal_cord_midline_col(full_width)
    rh_segments = clip_contour_segments_to_right_hemisphere(segments, full_width)
    for seg in rh_segments:
        if len(seg) < 2:
            continue
        ax.plot(
            seg[:, 0] - mid,
            seg[:, 1],
            color=outline_color,
            linewidth=outline_linewidth,
            zorder=zorder,
            solid_capstyle="round",
        )


def _wm_enclosing_smoothed_field(
    wm_mask_full: np.ndarray,
    *,
    smooth_sigma_pixels: float,
    max_dilation_steps: int = DEFAULT_WM_OUTLINE_ENCLOSURE_MAX_DILATION_STEPS,
) -> np.ndarray:
    """
    Dilate WM, then Gaussian-smooth until every WM pixel has field > 0.5.

    Guarantees the 0.5 contour encloses all WM mask pixels on the full cord.
    """
    if not wm_mask_full.any():
        return np.zeros_like(wm_mask_full, dtype=np.float32)
    wm_sigma = max(float(smooth_sigma_pixels), 0.5)
    dilation_iters = int(DEFAULT_WM_OUTLINE_INITIAL_DILATION_ITERS)
    field: np.ndarray | None = None
    for step in range(max_dilation_steps + 1):
        dilated = wm_mask_full
        if dilation_iters > 0:
            dilated = ndi.binary_dilation(dilated, iterations=dilation_iters)
        field = ndi.gaussian_filter(
            dilated.astype(np.float32), sigma=wm_sigma, mode="constant"
        )
        if np.all(field[wm_mask_full] > 0.499):
            return field
        dilation_iters += 1
    assert field is not None
    return field


def draw_wm_enclosing_matter_mask_outline(
    ax: plt.Axes,
    wm_mask_full: np.ndarray,
    full_width: int,
    *,
    smooth_sigma_pixels: float,
    outline_color: str = DEFAULT_WM_OUTLINE_COLOR,
    outline_linewidth: float = DEFAULT_MATTER_OUTLINE_LINEWIDTH,
    zorder: float = 4.5,
    max_dilation_steps: int = DEFAULT_WM_OUTLINE_ENCLOSURE_MAX_DILATION_STEPS,
) -> None:
    """
    WM outline: dilate before smoothing so the 0.5 contour encloses all WM pixels.

    Interior GM–WM interface loops are omitted so only the exterior WM boundary is drawn.
    Full-cord field; plotted on the right hemisphere only (GM-style clip).
    """
    if not wm_mask_full.any():
        return
    field = _wm_enclosing_smoothed_field(
        wm_mask_full,
        smooth_sigma_pixels=smooth_sigma_pixels,
        max_dilation_steps=max_dilation_steps,
    )
    full_height = wm_mask_full.shape[0]
    segments = _contour_segments_at_level(
        field,
        0.5,
        extent_x_max=float(full_width),
        extent_y_max=float(full_height),
    )
    segments = _filter_wm_outline_segments_exterior_only(segments)
    _plot_right_hemisphere_contour_segments(
        ax,
        segments,
        full_width,
        outline_color=outline_color,
        outline_linewidth=outline_linewidth,
        zorder=zorder,
    )


def draw_smooth_matter_mask_outline(
    ax: plt.Axes,
    matter_mask_full: np.ndarray,
    full_width: int,
    *,
    smooth_sigma_pixels: float,
    outline_color: str = DEFAULT_GM_OUTLINE_COLOR,
    outline_linewidth: float = DEFAULT_MATTER_OUTLINE_LINEWIDTH,
    zorder: float = 5,
) -> None:
    """
    Smooth outline from the full cord, plotted only on the right-hemisphere axes.

    The full binary mask is Gaussian-smoothed and contoured at level 0.5; segments are
    clipped to x >= midline and x is shifted for the cropped RH extent.
    """
    if not matter_mask_full.any():
        return
    full_height = matter_mask_full.shape[0]
    field = ndi.gaussian_filter(
        matter_mask_full.astype(np.float32),
        sigma=float(smooth_sigma_pixels),
        mode="constant",
    )
    segments = _contour_segments_at_level(
        field,
        0.5,
        extent_x_max=float(full_width),
        extent_y_max=float(full_height),
    )
    _plot_right_hemisphere_contour_segments(
        ax,
        segments,
        full_width,
        outline_color=outline_color,
        outline_linewidth=outline_linewidth,
        zorder=zorder,
    )


def draw_smooth_gray_matter_outline(
    ax: plt.Axes,
    gm_mask_full: np.ndarray,
    full_width: int,
    rh_plot_width: int,
    height: int,
    *,
    smooth_sigma_pixels: float,
    outline_color: str = DEFAULT_GM_OUTLINE_COLOR,
    outline_linewidth: float = DEFAULT_GM_OUTLINE_LINEWIDTH,
) -> None:
    """Alias: GM-only outline (rh_plot_width and height unused; kept for compatibility)."""
    del rh_plot_width, height
    draw_smooth_matter_mask_outline(
        ax,
        gm_mask_full,
        full_width,
        smooth_sigma_pixels=smooth_sigma_pixels,
        outline_color=outline_color,
        outline_linewidth=outline_linewidth,
    )


def draw_smooth_gm_wm_matter_outlines(
    ax: plt.Axes,
    region_mask_full: np.ndarray,
    matter_lookup: dict[int, str],
    full_width: int,
    *,
    gm_outline_smooth_sigma_fraction: float = DEFAULT_GM_OUTLINE_SMOOTH_SIGMA_FRACTION,
    wm_outline_smooth_sigma_fraction: float = DEFAULT_WM_OUTLINE_SMOOTH_SIGMA_FRACTION,
    gm_outline_color: str = DEFAULT_GM_OUTLINE_COLOR,
    wm_outline_color: str = DEFAULT_WM_OUTLINE_COLOR,
    outline_linewidth: float = DEFAULT_MATTER_OUTLINE_LINEWIDTH,
) -> None:
    """Draw smooth GM and WM outlines (full cord, right-hemisphere clip).

    GM: Gaussian-smoothed mask, 0.5 contour.
    WM: binary dilate (until enclosed) + Gaussian smooth at wm σ, 0.5 contour.
    """
    gm_mask = build_matter_class_binary_mask(region_mask_full, matter_lookup, "GM")
    wm_mask = build_matter_class_binary_mask(region_mask_full, matter_lookup, "WM")
    gm_sigma = matter_outline_smooth_sigma_pixels(
        region_mask_full, matter_lookup, gm_outline_smooth_sigma_fraction
    )
    wm_sigma = matter_outline_smooth_sigma_pixels(
        region_mask_full, matter_lookup, wm_outline_smooth_sigma_fraction
    )
    draw_wm_enclosing_matter_mask_outline(
        ax,
        wm_mask,
        full_width,
        smooth_sigma_pixels=wm_sigma,
        outline_color=wm_outline_color,
        outline_linewidth=outline_linewidth,
        zorder=4.5,
    )
    draw_smooth_matter_mask_outline(
        ax,
        gm_mask,
        full_width,
        smooth_sigma_pixels=float(gm_sigma),
        outline_color=gm_outline_color,
        outline_linewidth=outline_linewidth,
        zorder=5,
    )


def format_sigma_title_kernel_width_fraction(kernel_width_fraction: float) -> str:
    """Plot-title fragment: σ expressed as % of gray-matter vertical height."""
    pct = kernel_width_fraction * 100.0
    if abs(pct - round(pct)) < 1e-6:
        pct_str = f"{int(round(pct))}"
    else:
        pct_str = f"{pct:g}"
    return f"σ = {pct_str}% GM height"


def gaussian_kernel_sigma_pixels(
    gm_vertical_height: int,
    kernel_width_fraction: float,
) -> float:
    """Isotropic Gaussian σ in atlas pixels = fraction × GM vertical height."""
    if kernel_width_fraction <= 0:
        raise ValueError("kernel_width_fraction must be positive.")
    if gm_vertical_height <= 0:
        raise ValueError("gm_vertical_height must be positive.")
    return float(kernel_width_fraction * gm_vertical_height)


def compute_cell_gaussian_kde_grid(
    atlas_x: np.ndarray,
    atlas_y: np.ndarray,
    grid_shape: tuple[int, int],
    sigma_pixels: float,
) -> np.ndarray:
    """
    2-D Gaussian kernel density on an atlas pixel grid (origin upper-left).

    Places unit mass at each rounded (atlas_x, atlas_y), then applies an isotropic
    Gaussian filter (equivalent to summing Gaussian kernels at each cell).
    """
    height, width = grid_shape
    impulse = np.zeros((height, width), dtype=np.float64)
    for x_val, y_val in zip(atlas_x, atlas_y):
        col = int(round(float(x_val)))
        row = int(round(float(y_val)))
        if 0 <= row < height and 0 <= col < width:
            impulse[row, col] += 1.0
    if sigma_pixels <= 0:
        return impulse
    return ndi.gaussian_filter(impulse, sigma=sigma_pixels, mode="constant")


def _all_lamina_count_series(laminar_counts: pd.Series) -> pd.Series:
    """Lamina_I–X in order, count 0 for laminae absent from laminar_counts."""
    counts = pd.Series(0, index=list(COMBINED_LAMINA_ORDER), dtype=int)
    for lab in COMBINED_LAMINA_ORDER:
        if lab in laminar_counts.index:
            counts[lab] = int(laminar_counts[lab])
    return counts


def _kde_contour_percentile_list(
    percentile_min: int,
    percentile_max: int,
    percentile_step: int,
) -> list[int]:
    if percentile_step <= 0:
        raise ValueError("percentile_step must be positive.")
    if not 0 <= percentile_min <= 100 or not 0 <= percentile_max <= 100:
        raise ValueError("KDE contour percentiles must be between 0 and 100.")
    if percentile_min > percentile_max:
        raise ValueError("percentile_min must be <= percentile_max.")
    return list(range(percentile_min, percentile_max + 1, percentile_step))


def kde_density_contour_levels(
    density: np.ndarray,
    percentile_min: int = DEFAULT_KDE_CONTOUR_PERCENTILE_MIN,
    percentile_max: int = DEFAULT_KDE_CONTOUR_PERCENTILE_MAX,
    percentile_step: int = DEFAULT_KDE_CONTOUR_PERCENTILE_STEP,
) -> tuple[list[int], np.ndarray]:
    """
    Contour levels = percentiles of strictly positive KDE grid values.

    Returns (percentile_list, level_values).
    """
    percentiles = _kde_contour_percentile_list(
        percentile_min, percentile_max, percentile_step
    )
    positive = density[density > 0]
    if positive.size == 0:
        raise ValueError("KDE grid has no positive values; cannot compute contours.")
    levels = np.percentile(positive, percentiles)
    # Ensure strictly increasing levels for contour plotting
    levels = np.maximum.accumulate(levels)
    return percentiles, levels


def _kde_fraction_of_max_list(
    fraction_min: float,
    fraction_max: float,
    fraction_step: float,
) -> list[float]:
    if fraction_step <= 0:
        raise ValueError("fraction_step must be positive.")
    if not 0 < fraction_min <= 1 or not 0 < fraction_max <= 1:
        raise ValueError("KDE fraction-of-max values must be in (0, 1].")
    if fraction_min > fraction_max:
        raise ValueError("fraction_min must be <= fraction_max.")
    fractions = np.arange(fraction_min, fraction_max + fraction_step * 0.5, fraction_step)
    return [float(round(f, 6)) for f in fractions]


def kde_density_fraction_of_max_contour_levels(
    density: np.ndarray,
    fraction_min: float = DEFAULT_KDE_FRACTION_MAX_MIN,
    fraction_max: float = DEFAULT_KDE_FRACTION_MAX_MAX,
    fraction_step: float = DEFAULT_KDE_FRACTION_MAX_STEP,
) -> tuple[list[float], np.ndarray]:
    """
    Contour levels = fraction × max(KDE grid).

    Returns (fraction_list, level_values).
    """
    fractions = _kde_fraction_of_max_list(fraction_min, fraction_max, fraction_step)
    dmax = float(np.max(density))
    if dmax <= 0:
        raise ValueError("KDE grid max is non-positive; cannot compute contours.")
    levels = np.array([f * dmax for f in fractions], dtype=float)
    levels = np.maximum.accumulate(levels)
    return fractions, levels


def _prepare_right_hemisphere_kde_context(
    assigned_df: pd.DataFrame,
    atlas_dir: PathLike,
    *,
    segment: Optional[str],
    kernel_width_fraction: float,
    gm_color: RGB,
    wm_color: RGB,
    other_color: RGB,
    matter_backdrop_mode: MatterBackdropMode = MATTER_BACKDROP_ATLAS,
    gm_outline_smooth_sigma_fraction: float = DEFAULT_GM_OUTLINE_SMOOTH_SIGMA_FRACTION,
    wm_outline_smooth_sigma_fraction: float = DEFAULT_WM_OUTLINE_SMOOTH_SIGMA_FRACTION,
) -> dict[str, Any]:
    """Shared KDE grid + right-hemisphere backdrop for contour plots."""
    segment_label = _resolve_segment_label(assigned_df, segment)
    atlas_regions = _load_atlas_matter_lookup(atlas_dir)
    matter_lookup = _build_matter_class_lookup(atlas_regions)
    _, region_mask, frame_idx = _load_segment_region_mask(atlas_dir, segment_label)
    height, full_width = region_mask.shape
    region_mask_rh = crop_to_right_hemisphere(region_mask)
    _, width = region_mask_rh.shape

    gm_height = gray_matter_vertical_height_pixels(region_mask, matter_lookup)
    sigma_pixels = gaussian_kernel_sigma_pixels(gm_height, kernel_width_fraction)

    plot_df = assigned_df.copy()
    if "segment" in plot_df.columns:
        plot_df = plot_df[plot_df["segment"].astype(str) == segment_label]

    x_plot, y_plot = fold_cells_xy_for_right_hemisphere_plot(
        plot_df["atlas_x"].to_numpy(),
        plot_df["atlas_y"].to_numpy(),
        full_width,
    )
    density = compute_cell_gaussian_kde_grid(
        x_plot,
        y_plot,
        (height, width),
        sigma_pixels,
    )
    backdrop = build_matter_backdrop_rgb(
        region_mask_rh,
        matter_lookup,
        gm_color=gm_color,
        wm_color=wm_color,
        other_color=other_color,
    )
    return {
        "segment_label": segment_label,
        "frame_idx": frame_idx,
        "plot_df": plot_df,
        "density": density,
        "backdrop": backdrop,
        "height": height,
        "width": width,
        "full_width": full_width,
        "region_mask": region_mask,
        "matter_lookup": matter_lookup,
        "gm_height": gm_height,
        "sigma_pixels": sigma_pixels,
        "matter_backdrop_mode": matter_backdrop_mode,
        "gm_color": gm_color,
        "wm_color": wm_color,
        "other_color": other_color,
        "gm_outline_smooth_sigma_fraction": gm_outline_smooth_sigma_fraction,
        "wm_outline_smooth_sigma_fraction": wm_outline_smooth_sigma_fraction,
    }


def _hide_spatial_axis_ticks(ax: plt.Axes) -> None:
    """Remove tick marks and tick labels on spatial atlas plots."""
    ax.set_xticks([])
    ax.set_yticks([])


def draw_atlas_scale_bar(
    ax: plt.Axes,
    width: float,
    height: float,
    *,
    length_px: float = SCALE_BAR_LENGTH_PX,
    label: str = SCALE_BAR_LABEL,
    inset_right_frac: float = SCALE_BAR_INSET_RIGHT_FRAC,
    inset_bottom_frac: float = SCALE_BAR_INSET_BOTTOM_FRAC,
    line_color: str = "black",
    line_width: float = 2.0,
    label_fontsize: float = SCALE_BAR_LABEL_FONTSIZE,
) -> None:
    """
    Horizontal scale bar in atlas pixel coordinates (origin upper; 10 µm/pixel).

    Right edge at (1 - inset_right_frac) × width; bar baseline inset_bottom_frac above
    the bottom (y = height × (1 - inset_bottom_frac)). Defaults: 10 px bar, “100 um”
    at 10 µm/pixel.
    """
    x_right = width * (1.0 - inset_right_frac)
    x_left = x_right - length_px
    y_line = height * (1.0 - inset_bottom_frac)
    ax.plot(
        [x_left, x_right],
        [y_line, y_line],
        color=line_color,
        linewidth=line_width,
        solid_capstyle="butt",
        zorder=10,
    )
    label_y = y_line + max(2.0, height * 0.015)
    ax.text(
        (x_left + x_right) / 2.0,
        label_y,
        label,
        ha="center",
        va="top",
        fontsize=label_fontsize,
        color=line_color,
        zorder=10,
    )


def _save_kde_contour_backdrop_png(
    ctx: dict[str, Any],
    output_png: PathLike,
    *,
    contour_levels: np.ndarray,
    contour_labels: list[str],
    title: str,
    dpi: int,
    contour_color: str,
    contour_linewidth: float,
    show_contour_inline_labels: bool = False,
    draw_matter_outlines: bool = False,
    gm_outline_smooth_sigma_fraction: float = DEFAULT_GM_OUTLINE_SMOOTH_SIGMA_FRACTION,
    wm_outline_smooth_sigma_fraction: float = DEFAULT_WM_OUTLINE_SMOOTH_SIGMA_FRACTION,
    gm_outline_color: str = DEFAULT_GM_OUTLINE_COLOR,
    wm_outline_color: str = DEFAULT_WM_OUTLINE_COLOR,
    matter_outline_linewidth: float = DEFAULT_MATTER_OUTLINE_LINEWIDTH,
) -> None:
    height = ctx["height"]
    width = ctx["width"]
    fig_w = 8.0
    fig_h = fig_w * (height / width)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    if ctx.get("matter_backdrop_mode") == MATTER_BACKDROP_OUTLINE_FILLED:
        draw_outline_filled_matter_patches(
            ax,
            ctx["region_mask"],
            ctx["matter_lookup"],
            width,
            height,
            gm_outline_smooth_sigma_fraction=ctx["gm_outline_smooth_sigma_fraction"],
            wm_outline_smooth_sigma_fraction=ctx["wm_outline_smooth_sigma_fraction"],
            gm_color=ctx["gm_color"],
            wm_color=ctx["wm_color"],
            other_color=ctx["other_color"],
        )
        ax.set_aspect("equal")
    else:
        ax.imshow(
            ctx["backdrop"],
            origin="upper",
            extent=(0, width, height, 0),
            interpolation="nearest",
            aspect="equal",
            zorder=1,
        )
    for level, label in zip(contour_levels, contour_labels):
        contour_set = ax.contour(
            ctx["density"],
            levels=[float(level)],
            extent=(0, width, height, 0),
            origin="upper",
            colors=contour_color,
            linewidths=contour_linewidth,
            zorder=2,
        )
        if show_contour_inline_labels:
            ax.clabel(contour_set, inline=True, fontsize=7, fmt=label)

    if draw_matter_outlines:
        region_mask = ctx["region_mask"]
        matter_lookup = ctx["matter_lookup"]
        full_width = ctx["full_width"]
        draw_smooth_gm_wm_matter_outlines(
            ax,
            region_mask,
            matter_lookup,
            full_width,
            gm_outline_smooth_sigma_fraction=gm_outline_smooth_sigma_fraction,
            wm_outline_smooth_sigma_fraction=wm_outline_smooth_sigma_fraction,
            gm_outline_color=gm_outline_color,
            wm_outline_color=wm_outline_color,
            outline_linewidth=matter_outline_linewidth,
        )

    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_title(title)
    _hide_spatial_axis_ticks(ax)
    draw_atlas_scale_bar(ax, width, height)
    fig.tight_layout()

    out_path = Path(output_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _kde_plot_result_metadata(
    ctx: dict[str, Any],
    output_png: PathLike,
    *,
    kernel_width_fraction: float,
    dpi: int,
    extra: dict[str, Any],
) -> dict[str, Any]:
    base = {
        "output_png": str(Path(output_png)),
        "segment": ctx["segment_label"],
        "segment_frame_index": ctx["frame_idx"],
        "n_cells": int(len(ctx["plot_df"])),
        "gray_matter_vertical_height_px": ctx["gm_height"],
        "kernel_width_fraction": kernel_width_fraction,
        "gaussian_sigma_pixels": ctx["sigma_pixels"],
        "dpi": dpi,
        "right_hemisphere_backdrop_size": (ctx["width"], ctx["height"]),
        "midline_col_full_atlas": spinal_cord_midline_col(ctx["full_width"]),
    }
    base.update(extra)
    return base


def plot_laminar_counts_bar(
    laminar_counts: pd.Series,
    output_png: PathLike,
    *,
    total_cells: int,
    dpi: int = DEFAULT_PLOT_DPI,
    bar_color: str = "steelblue",
    title: str = "Combined lamina cell percentages",
) -> dict[str, Any]:
    """Bar plot of combined lamina percentages (% of total_cells); all Lamina_I–X shown."""
    if total_cells <= 0:
        raise ValueError("total_cells must be positive.")
    counts = _all_lamina_count_series(laminar_counts)
    percentages = 100.0 * counts.astype(float) / float(total_cells)
    labels = [lab.replace("Lamina_", "Lamina ") for lab in counts.index.astype(str)]

    fig, ax = plt.subplots(figsize=(10, 5), dpi=dpi)
    x_pos = np.arange(len(percentages))
    ax.bar(x_pos, percentages.values, color=bar_color, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Cells (% of total)")
    ax.set_title(title)
    ax.set_ylim(0, max(percentages.max() * 1.15, 1.0) if percentages.max() > 0 else 1.0)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()

    out_path = Path(output_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return {
        "output_png": str(out_path),
        "dpi": dpi,
        "total_cells": int(total_cells),
        "lamina_labels": counts.index.astype(str).tolist(),
        "counts": counts.to_dict(),
        "percentages": percentages.to_dict(),
    }


def plot_cell_kde_on_spinalj_backdrop(
    assigned_df: pd.DataFrame,
    output_png: PathLike,
    atlas_dir: PathLike = DEFAULT_ATLAS_DIR,
    *,
    segment: Optional[str] = None,
    kernel_width_fraction: float = DEFAULT_KERNEL_WIDTH_FRACTION,
    kde_contour_percentile_min: int = DEFAULT_KDE_CONTOUR_PERCENTILE_MIN,
    kde_contour_percentile_max: int = DEFAULT_KDE_CONTOUR_PERCENTILE_MAX,
    kde_contour_percentile_step: int = DEFAULT_KDE_CONTOUR_PERCENTILE_STEP,
    dpi: int = DEFAULT_PLOT_DPI,
    gm_color: RGB = DEFAULT_GM_RGB,
    wm_color: RGB = DEFAULT_WM_RGB,
    other_color: RGB = DEFAULT_OTHER_RGB,
    contour_color: str = "darkred",
    contour_linewidth: float = 1.2,
    show_contour_inline_labels: bool = False,
    draw_matter_outlines: bool = False,
    matter_backdrop_mode: MatterBackdropMode = MATTER_BACKDROP_ATLAS,
    gm_outline_smooth_sigma_fraction: float = DEFAULT_GM_OUTLINE_SMOOTH_SIGMA_FRACTION,
    wm_outline_smooth_sigma_fraction: float = DEFAULT_WM_OUTLINE_SMOOTH_SIGMA_FRACTION,
) -> dict[str, Any]:
    """
    GM/WM backdrop with 2-D Gaussian KDE contours of cell atlas coordinates.

    Kernel σ (pixels) = kernel_width_fraction × gray-matter vertical height in segment.
    Contour levels are percentiles (default 30–90 step 10) of positive KDE grid values.
    """
    ctx = _prepare_right_hemisphere_kde_context(
        assigned_df,
        atlas_dir,
        segment=segment,
        kernel_width_fraction=kernel_width_fraction,
        gm_color=gm_color,
        wm_color=wm_color,
        other_color=other_color,
        matter_backdrop_mode=matter_backdrop_mode,
        gm_outline_smooth_sigma_fraction=gm_outline_smooth_sigma_fraction,
        wm_outline_smooth_sigma_fraction=wm_outline_smooth_sigma_fraction,
    )
    percentiles, contour_levels = kde_density_contour_levels(
        ctx["density"],
        percentile_min=kde_contour_percentile_min,
        percentile_max=kde_contour_percentile_max,
        percentile_step=kde_contour_percentile_step,
    )
    labels = [f"P{pct}" for pct in percentiles]
    sigma_title = format_sigma_title_kernel_width_fraction(kernel_width_fraction)
    if matter_backdrop_mode == MATTER_BACKDROP_OUTLINE_FILLED:
        backdrop_title = " — outline-filled GM/WM"
    else:
        backdrop_title = " + GM/WM outline" if draw_matter_outlines else ""
    title = (
        f"Cell location KDE contours: percentiles - {ctx['segment_label']} "
        f"({sigma_title}; contours {percentiles[0]}–{percentiles[-1]} pct){backdrop_title}"
    )
    _save_kde_contour_backdrop_png(
        ctx,
        output_png,
        contour_levels=contour_levels,
        contour_labels=labels,
        title=title,
        dpi=dpi,
        contour_color=contour_color,
        contour_linewidth=contour_linewidth,
        show_contour_inline_labels=show_contour_inline_labels,
        draw_matter_outlines=draw_matter_outlines,
        gm_outline_smooth_sigma_fraction=gm_outline_smooth_sigma_fraction,
        wm_outline_smooth_sigma_fraction=wm_outline_smooth_sigma_fraction,
    )
    return _kde_plot_result_metadata(
        ctx,
        output_png,
        kernel_width_fraction=kernel_width_fraction,
        dpi=dpi,
        extra={
            "kde_contour_percentiles": percentiles,
            "kde_contour_levels": contour_levels.tolist(),
            "show_contour_inline_labels": show_contour_inline_labels,
            "draw_matter_outlines": draw_matter_outlines,
            "matter_backdrop_mode": matter_backdrop_mode,
        },
    )


def plot_cell_kde_fraction_of_max_on_spinalj_backdrop(
    assigned_df: pd.DataFrame,
    output_png: PathLike,
    atlas_dir: PathLike = DEFAULT_ATLAS_DIR,
    *,
    segment: Optional[str] = None,
    kernel_width_fraction: float = DEFAULT_KERNEL_WIDTH_FRACTION,
    kde_fraction_max_min: float = DEFAULT_KDE_FRACTION_MAX_MIN,
    kde_fraction_max_max: float = DEFAULT_KDE_FRACTION_MAX_MAX,
    kde_fraction_max_step: float = DEFAULT_KDE_FRACTION_MAX_STEP,
    dpi: int = DEFAULT_PLOT_DPI,
    gm_color: RGB = DEFAULT_GM_RGB,
    wm_color: RGB = DEFAULT_WM_RGB,
    other_color: RGB = DEFAULT_OTHER_RGB,
    contour_color: str = "darkred",
    contour_linewidth: float = 1.2,
    show_contour_inline_labels: bool = False,
    draw_matter_outlines: bool = False,
    matter_backdrop_mode: MatterBackdropMode = MATTER_BACKDROP_ATLAS,
    gm_outline_smooth_sigma_fraction: float = DEFAULT_GM_OUTLINE_SMOOTH_SIGMA_FRACTION,
    wm_outline_smooth_sigma_fraction: float = DEFAULT_WM_OUTLINE_SMOOTH_SIGMA_FRACTION,
) -> dict[str, Any]:
    """
    Same Gaussian KDE and right-hemisphere backdrop as plot_cell_kde_on_spinalj_backdrop.

    Contour levels are fractions of max KDE (default 0.3–0.9 step 0.1); inline labels
    show the fraction when show_contour_inline_labels is True.
    """
    ctx = _prepare_right_hemisphere_kde_context(
        assigned_df,
        atlas_dir,
        segment=segment,
        kernel_width_fraction=kernel_width_fraction,
        gm_color=gm_color,
        wm_color=wm_color,
        other_color=other_color,
        matter_backdrop_mode=matter_backdrop_mode,
        gm_outline_smooth_sigma_fraction=gm_outline_smooth_sigma_fraction,
        wm_outline_smooth_sigma_fraction=wm_outline_smooth_sigma_fraction,
    )
    fractions, contour_levels = kde_density_fraction_of_max_contour_levels(
        ctx["density"],
        fraction_min=kde_fraction_max_min,
        fraction_max=kde_fraction_max_max,
        fraction_step=kde_fraction_max_step,
    )
    labels = [f"{frac:g}" for frac in fractions]
    sigma_title = format_sigma_title_kernel_width_fraction(kernel_width_fraction)
    if matter_backdrop_mode == MATTER_BACKDROP_OUTLINE_FILLED:
        backdrop_title = " — outline-filled GM/WM"
    else:
        backdrop_title = " + GM/WM outline" if draw_matter_outlines else ""
    title = (
        f"Cell location KDE contours: frac. of max - {ctx['segment_label']} "
        f"({sigma_title}; contours {fractions[0]:g}–{fractions[-1]:g} × max){backdrop_title}"
    )
    _save_kde_contour_backdrop_png(
        ctx,
        output_png,
        contour_levels=contour_levels,
        contour_labels=labels,
        title=title,
        dpi=dpi,
        contour_color=contour_color,
        contour_linewidth=contour_linewidth,
        show_contour_inline_labels=show_contour_inline_labels,
        draw_matter_outlines=draw_matter_outlines,
        gm_outline_smooth_sigma_fraction=gm_outline_smooth_sigma_fraction,
        wm_outline_smooth_sigma_fraction=wm_outline_smooth_sigma_fraction,
    )
    return _kde_plot_result_metadata(
        ctx,
        output_png,
        kernel_width_fraction=kernel_width_fraction,
        dpi=dpi,
        extra={
            "kde_density_max": float(np.max(ctx["density"])),
            "kde_fraction_of_max": fractions,
            "kde_contour_levels": contour_levels.tolist(),
            "show_contour_inline_labels": show_contour_inline_labels,
            "draw_matter_outlines": draw_matter_outlines,
            "matter_backdrop_mode": matter_backdrop_mode,
        },
    )


def _plot_cells_on_lamina_color_backdrop(
    assigned_df: pd.DataFrame,
    output_png: PathLike,
    atlas_dir: PathLike,
    *,
    segment: Optional[str],
    dpi: int,
    cell_color: str,
    cell_marker_size: float,
    cell_edge_color: str,
    cell_edge_width: float,
    plot_lamina_roman_labels: bool,
    draw_matter_outlines: bool,
    gm_outline_smooth_sigma_fraction: float,
    wm_outline_smooth_sigma_fraction: float,
    gm_outline_color: str,
    wm_outline_color: str,
    matter_outline_linewidth: float,
    title_suffix: str,
    matter_backdrop_mode: MatterBackdropMode = MATTER_BACKDROP_ATLAS,
    gm_color: RGB = DEFAULT_GM_RGB,
    wm_color: RGB = DEFAULT_WM_RGB,
    other_color: RGB = DEFAULT_OTHER_RGB,
) -> dict[str, Any]:
    if matter_backdrop_mode == MATTER_BACKDROP_OUTLINE_FILLED:
        raise ValueError(
            "Lamina color plots do not support matter_backdrop_mode='outline_filled'."
        )
    segment_label = _resolve_segment_label(assigned_df, segment)
    atlas_regions = _load_atlas_matter_lookup(atlas_dir)
    matter_lookup = _build_matter_class_lookup(atlas_regions)
    _, region_mask, _ = _load_segment_region_mask(atlas_dir, segment_label)

    lamina_label_positions: dict[str, tuple[float, float]] = {}
    if plot_lamina_roman_labels:
        lamina_label_positions = compute_lamina_label_positions_centroid_x(
            region_mask, atlas_regions
        )

    rgb_full = build_combined_lamina_separable_backdrop_rgb(region_mask, atlas_regions)
    rgb = crop_rgb_to_right_hemisphere(rgb_full)
    height, width = rgb.shape[0], rgb.shape[1]
    full_width = region_mask.shape[1]

    plot_df = assigned_df.copy()
    if "segment" in plot_df.columns:
        plot_df = plot_df[plot_df["segment"].astype(str) == segment_label]
    if "atlas_x" not in plot_df.columns or "atlas_y" not in plot_df.columns:
        raise KeyError("assigned_df must contain atlas_x and atlas_y.")

    x_plot, y_plot = fold_cells_xy_for_right_hemisphere_plot(
        plot_df["atlas_x"].to_numpy(),
        plot_df["atlas_y"].to_numpy(),
        full_width,
    )

    gm_outline_sigma_pixels: Optional[float] = None
    wm_outline_sigma_pixels: Optional[float] = None
    if draw_matter_outlines:
        gm_outline_sigma_pixels = matter_outline_smooth_sigma_pixels(
            region_mask, matter_lookup, gm_outline_smooth_sigma_fraction
        )
        wm_outline_sigma_pixels = matter_outline_smooth_sigma_pixels(
            region_mask, matter_lookup, wm_outline_smooth_sigma_fraction
        )

    fig_w = 8.0
    fig_h = fig_w * (height / width)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    ax.imshow(
        rgb,
        origin="upper",
        extent=(0, width, height, 0),
        interpolation="nearest",
        aspect="equal",
    )
    ax.scatter(
        x_plot,
        y_plot,
        s=cell_marker_size,
        c=cell_color,
        edgecolors=cell_edge_color,
        linewidths=cell_edge_width,
        zorder=3,
    )
    if draw_matter_outlines:
        draw_smooth_gm_wm_matter_outlines(
            ax,
            region_mask,
            matter_lookup,
            full_width,
            gm_outline_smooth_sigma_fraction=gm_outline_smooth_sigma_fraction,
            wm_outline_smooth_sigma_fraction=wm_outline_smooth_sigma_fraction,
            gm_outline_color=gm_outline_color,
            wm_outline_color=wm_outline_color,
            outline_linewidth=matter_outline_linewidth,
        )
    if plot_lamina_roman_labels and lamina_label_positions:
        draw_lamina_roman_labels(ax, lamina_label_positions)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_title(f"Cell locations — lamina color{title_suffix} — segment {segment_label}")
    _hide_spatial_axis_ticks(ax)
    draw_atlas_scale_bar(ax, width, height)
    fig.tight_layout()

    out_path = Path(output_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    result: dict[str, Any] = {
        "output_png": str(out_path),
        "segment": segment_label,
        "lamina_backdrop": "combined_lamina_separable_tab10",
        "matter_backdrop_mode": matter_backdrop_mode,
        "n_cells_plotted": int(len(plot_df)),
        "lamina_label_positions_plot_xy": lamina_label_positions,
        "plot_lamina_roman_labels": plot_lamina_roman_labels,
        "dpi": dpi,
        "right_hemisphere_backdrop_size": (width, height),
        "midline_col_full_atlas": spinal_cord_midline_col(full_width),
    }
    result["draw_matter_outlines"] = draw_matter_outlines
    if draw_matter_outlines:
        result["gm_outline_smooth_sigma_pixels"] = gm_outline_sigma_pixels
        result["wm_outline_smooth_sigma_pixels"] = wm_outline_sigma_pixels
        result["gm_outline_smooth_sigma_fraction"] = gm_outline_smooth_sigma_fraction
        result["wm_outline_smooth_sigma_fraction"] = wm_outline_smooth_sigma_fraction
    return result


def plot_cells_on_lamina_color_backdrop(
    assigned_df: pd.DataFrame,
    output_png: PathLike,
    atlas_dir: PathLike = DEFAULT_ATLAS_DIR,
    *,
    segment: Optional[str] = None,
    dpi: int = DEFAULT_PLOT_DPI,
    cell_color: str = "black",
    cell_marker_size: float = 18.0,
    cell_edge_color: str = "none",
    cell_edge_width: float = 0.0,
    plot_lamina_roman_labels: bool = True,
    draw_matter_outlines: bool = False,
    matter_backdrop_mode: MatterBackdropMode = MATTER_BACKDROP_ATLAS,
    gm_outline_smooth_sigma_fraction: float = DEFAULT_GM_OUTLINE_SMOOTH_SIGMA_FRACTION,
    wm_outline_smooth_sigma_fraction: float = DEFAULT_WM_OUTLINE_SMOOTH_SIGMA_FRACTION,
    gm_color: RGB = DEFAULT_GM_RGB,
    wm_color: RGB = DEFAULT_WM_RGB,
    other_color: RGB = DEFAULT_OTHER_RGB,
    gm_outline_color: str = DEFAULT_GM_OUTLINE_COLOR,
    wm_outline_color: str = DEFAULT_WM_OUTLINE_COLOR,
    matter_outline_linewidth: float = DEFAULT_MATTER_OUTLINE_LINEWIDTH,
) -> dict[str, Any]:
    """
    Plot folded cell locations on a combined-lamina color backdrop (right hemisphere).

    Backdrop from Annotation_Segments_Only.tif: one separable color per Lamina_I–X.
    Optional roman-numeral lamina labels (plot_lamina_roman_labels, default False).
    Optional draw_matter_outlines: smooth GM (black) and WM (dimgray) outlines on full cord.
    matter_backdrop_mode='outline_filled' is not supported (use scatter/KDE third variants only).
    """
    if matter_backdrop_mode == MATTER_BACKDROP_OUTLINE_FILLED:
        raise ValueError(
            "Lamina color plots do not support matter_backdrop_mode='outline_filled'."
        )
    title_suffix = " + GM/WM outline" if draw_matter_outlines else ""
    return _plot_cells_on_lamina_color_backdrop(
        assigned_df,
        output_png,
        atlas_dir,
        segment=segment,
        dpi=dpi,
        cell_color=cell_color,
        cell_marker_size=cell_marker_size,
        cell_edge_color=cell_edge_color,
        cell_edge_width=cell_edge_width,
        plot_lamina_roman_labels=plot_lamina_roman_labels,
        draw_matter_outlines=draw_matter_outlines,
        gm_outline_smooth_sigma_fraction=gm_outline_smooth_sigma_fraction,
        wm_outline_smooth_sigma_fraction=wm_outline_smooth_sigma_fraction,
        gm_outline_color=gm_outline_color,
        wm_outline_color=wm_outline_color,
        matter_outline_linewidth=matter_outline_linewidth,
        title_suffix=title_suffix,
        matter_backdrop_mode=matter_backdrop_mode,
        gm_color=gm_color,
        wm_color=wm_color,
        other_color=other_color,
    )


def plot_cells_on_lamina_color_backdrop_with_gm_outline(
    assigned_df: pd.DataFrame,
    output_png: PathLike,
    atlas_dir: PathLike = DEFAULT_ATLAS_DIR,
    **kwargs: Any,
) -> dict[str, Any]:
    """Backward-compatible alias: outlines on with GM+WM (not GM-only)."""
    return plot_cells_on_lamina_color_backdrop(
        assigned_df, output_png, atlas_dir, draw_matter_outlines=True, **kwargs
    )


def plot_cells_on_spinalj_matter_backdrop(
    assigned_df: pd.DataFrame,
    output_png: PathLike,
    atlas_dir: PathLike = DEFAULT_ATLAS_DIR,
    *,
    segment: Optional[str] = None,
    dpi: int = DEFAULT_PLOT_DPI,
    gm_color: RGB = DEFAULT_GM_RGB,
    wm_color: RGB = DEFAULT_WM_RGB,
    other_color: RGB = DEFAULT_OTHER_RGB,
    cell_color: str = "crimson",
    cell_marker_size: float = 18.0,
    cell_edge_color: str = "black",
    cell_edge_width: float = 0.35,
    draw_matter_outlines: bool = False,
    matter_backdrop_mode: MatterBackdropMode = MATTER_BACKDROP_ATLAS,
    gm_outline_smooth_sigma_fraction: float = DEFAULT_GM_OUTLINE_SMOOTH_SIGMA_FRACTION,
    wm_outline_smooth_sigma_fraction: float = DEFAULT_WM_OUTLINE_SMOOTH_SIGMA_FRACTION,
    gm_outline_color: str = DEFAULT_GM_OUTLINE_COLOR,
    wm_outline_color: str = DEFAULT_WM_OUTLINE_COLOR,
    matter_outline_linewidth: float = DEFAULT_MATTER_OUTLINE_LINEWIDTH,
) -> dict[str, Any]:
    """
    Plot folded cell locations on a right-hemisphere GM/WM backdrop; save PNG.

    Left-of-midline cells are x-mirrored onto the right; backdrop is cropped at midline.
    Atlas image uses origin='upper' (row 0 at top).
    draw_matter_outlines adds smooth GM+WM outlines (full cord, RH clip).
    matter_backdrop_mode=outline_filled colors by smoothed outline interiors, not atlas IDs.
    """
    segment_label = _resolve_segment_label(assigned_df, segment)
    atlas_regions = _load_atlas_matter_lookup(atlas_dir)
    matter_lookup = _build_matter_class_lookup(atlas_regions)
    _, region_mask, frame_idx = _load_segment_region_mask(atlas_dir, segment_label)
    height, full_width = region_mask.shape
    region_mask_rh = crop_to_right_hemisphere(region_mask)
    _, width = region_mask_rh.shape

    plot_df = assigned_df.copy()
    if "segment" in plot_df.columns:
        plot_df = plot_df[plot_df["segment"].astype(str) == segment_label]
    if "atlas_x" not in plot_df.columns or "atlas_y" not in plot_df.columns:
        raise KeyError("assigned_df must contain atlas_x and atlas_y.")

    x_plot, y_plot = fold_cells_xy_for_right_hemisphere_plot(
        plot_df["atlas_x"].to_numpy(),
        plot_df["atlas_y"].to_numpy(),
        full_width,
    )

    fig_w = 8.0
    fig_h = fig_w * (height / width)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    if matter_backdrop_mode == MATTER_BACKDROP_OUTLINE_FILLED:
        draw_outline_filled_matter_patches(
            ax,
            region_mask,
            matter_lookup,
            width,
            height,
            gm_outline_smooth_sigma_fraction=gm_outline_smooth_sigma_fraction,
            wm_outline_smooth_sigma_fraction=wm_outline_smooth_sigma_fraction,
            gm_color=gm_color,
            wm_color=wm_color,
            other_color=other_color,
        )
        ax.set_aspect("equal")
    else:
        backdrop = build_matter_backdrop_rgb(
            region_mask_rh,
            matter_lookup,
            gm_color=gm_color,
            wm_color=wm_color,
            other_color=other_color,
        )
        ax.imshow(
            backdrop,
            origin="upper",
            extent=(0, width, height, 0),
            interpolation="nearest",
            aspect="equal",
        )
    ax.scatter(
        x_plot,
        y_plot,
        s=cell_marker_size,
        c=cell_color,
        edgecolors=cell_edge_color,
        linewidths=cell_edge_width,
        zorder=3,
    )
    if draw_matter_outlines and matter_backdrop_mode != MATTER_BACKDROP_OUTLINE_FILLED:
        draw_smooth_gm_wm_matter_outlines(
            ax,
            region_mask,
            matter_lookup,
            full_width,
            gm_outline_smooth_sigma_fraction=gm_outline_smooth_sigma_fraction,
            wm_outline_smooth_sigma_fraction=wm_outline_smooth_sigma_fraction,
            gm_outline_color=gm_outline_color,
            wm_outline_color=wm_outline_color,
            outline_linewidth=matter_outline_linewidth,
        )
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    if matter_backdrop_mode == MATTER_BACKDROP_OUTLINE_FILLED:
        title_suffix = " — outline-filled GM/WM"
    else:
        title_suffix = " + GM/WM outline" if draw_matter_outlines else ""
    ax.set_title(f"Cell locations{title_suffix} — segment {segment_label}")
    _hide_spatial_axis_ticks(ax)
    draw_atlas_scale_bar(ax, width, height)
    fig.tight_layout()

    out_path = Path(output_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return {
        "output_png": str(out_path),
        "segment": segment_label,
        "segment_frame_index": frame_idx,
        "n_cells_plotted": int(len(plot_df)),
        "draw_matter_outlines": draw_matter_outlines,
        "matter_backdrop_mode": matter_backdrop_mode,
        "dpi": dpi,
        "right_hemisphere_backdrop_size": (width, height),
        "midline_col_full_atlas": spinal_cord_midline_col(full_width),
    }
