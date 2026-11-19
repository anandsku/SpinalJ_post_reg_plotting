# SpinalJ post-registration plotting

Assign registered cells to SpinalJ (https://github.com/felixfiederling/SpinalJ) atlas regions, summarize laminar counts, and generate GM/WM, lamina, KDE, and bar-chart figures per spinal reference segment.

## Quick start

### 1. Environment

From the repository root:

```bash
conda env create -f environment.yml
conda activate spinalj_post_reg_plotting
```

Alternatively, create any Python 3.11 environment and run `pip install -r requirements.txt`.

Dependencies: **NumPy**, **pandas**, **SciPy**, **matplotlib**, **Pillow** (see pinned versions in `environment.yml`).

### 2. Atlas data

Example atlas files are included under `spinal_registration/atlas_data/useful_atlas_data/` (`Atlas_Regions.csv`, `Segments.csv`, `Annotation_Segments_Only.tif`). To use another registration version, replace those files with the same names (see the README in that folder).

### 3. Run

Edit paths in `spinal_registration/callers/call_plot_contours_lanimar_counts_spinalj.py` if needed, then:

```bash
cd spinal_registration
python callers/call_plot_contours_lanimar_counts_spinalj.py
```

Example inputs ship under `spinal_registration/test_data/`; outputs default to `spinal_registration/test_data/output/`. Plot aesthetics are controlled by `spinal_registration/test_data/spinalj_plot_settings.csv` (column **User_value**).

---

## Repository layout

| Path | Role |
|------|------|
| `spinal_registration/functions/plot_contours_lanimar_counts_spinalj.py` | Core assignment, statistics, and plotting API |
| `spinal_registration/callers/call_plot_contours_lanimar_counts_spinalj.py` | Example batch entry point |
| `spinal_registration/test_data/` | Example cell CSV, plot settings |
| `spinal_registration/atlas_data/useful_atlas_data/` | Bundled SpinalJ atlas CSV + segment mask TIFF |
| `environment.yml` / `requirements.txt` | Reproducible Python environment |

---

## 1. Entry point and workflow

**Caller:** `spinal_registration/callers/call_plot_contours_lanimar_counts_spinalj.py`  
**Function module:** `spinal_registration/functions/plot_contours_lanimar_counts_spinalj.py`

The caller invokes `process_spinalj_by_reference_segment()`, which:

1. Validates the atlas directory.
2. Loads plotting parameters from a settings CSV (creating a template if the file is missing).
3. Assigns every cell in the location CSV to SpinalJ atlas region IDs (all segments in one pass).
4. For each distinct **Reference section** (spinal segment label, e.g. `L4`) present in the data:
   - Writes two CSV files and thirteen PNG figures (see §5–§6).

All spatial plots use the **right half** of the atlas mask only: cells with atlas \(x\) left of the midline are mirrored onto the right, and the backdrop is cropped at the sagittal midline (`width // 2`). Image coordinates use `origin='upper'` (row 0 = dorsal/top). A 100 µm scale bar is drawn assuming **10 µm/pixel** in atlas space.

---

## 2. Inputs

### 2.1 Caller paths (required / explicit)

| Input | Role |
|--------|------|
| **cell_csv_path** | Cell metadata CSV. Required columns: **X**, **Y**, **Reference section** (must match `Segment` in `Segments.csv`), **Scaling factor for reference section** (positive; atlas coord = cell coord ÷ factor). |
| **atlas_dir** | **Required.** Directory validated to contain `Atlas_Regions.csv`, `Segments.csv`, and `Annotation_Segments_Only.tif`. No default path in the batch API. |
| **plot_settings_csv** | Optional full path; if omitted, defaults to `spinalj_plot_settings.csv` in the **parent folder of the cell CSV**. |
| **outputdir** | Optional; if omitted, defaults to `<cell_csv parent>/output/`. |

The included caller uses `test_data` paths relative to the `spinal_registration` package root.

### 2.2 Atlas files used (under `atlas_dir`)

| File | Use |
|------|-----|
| **Segments.csv** | Maps each **Segment** label (e.g. `L4`) to a row index; that index selects one frame of the multi-page TIFF. |
| **Annotation_Segments_Only.tif** | One 2-D **region-ID mask** per segment (pixel value = fine atlas region ID). Frame count must equal row count in `Segments.csv`. |
| **Atlas_Regions.csv** | Region **id**, **acronym**, **name**, **parent_ID**, **children_IDs**. Used to attach names, roll up **combined lamina** labels (Lamina_I–Lamina_X), and classify **GM / WM / other** by walking the parent chain to GM/WM roots. |

Files often bundled with the atlas but **not** read by this pipeline include `Annotation_Color.tif`, `Annotation.tif`, and `Segments_Reference.csv`. Lamina color backdrops use **tab10** colors per combined lamina from region IDs, not `Annotation_Color.tif`.

### 2.3 Plot settings CSV

**Format (fixed column order):** `Parameter`, `Default`, `User_value`, `Description`  
Booleans in **Default** / **User_value** are written as **TRUE** / **FALSE**.  
If **User_value** is blank, **Default** is used.

| Parameter | Default | Effect |
|-----------|---------|--------|
| **plot_dpi** | 300 | Resolution of every saved PNG (`savefig` DPI). Higher → larger files and sharper raster output. |
| **kernel_width_fraction** | 0.05 | KDE only: isotropic Gaussian σ in pixels = fraction × **gray-matter vertical height** of that segment (see §7). Larger → smoother, broader density maps and wider contours. |
| **kde_contour_percentile_min / max / step** | 45, 95, 10 | Percentile KDE: contour levels at these percentiles of **strictly positive** KDE grid values (e.g. 45, 55, …, 95). Wider range or smaller step → more contour lines; changing min/max shifts which density levels are shown. |
| **kde_fraction_max_min / max / step** | 0.35, 0.95, 0.1 | Fraction-of-max KDE: each level = fraction × max(KDE grid). Same interpretation as percentiles but tied to peak density, not the distribution of positive pixels. |
| **show_contour_inline_labels** | FALSE | If TRUE, text labels on KDE contours (percentile labels `P45`… or fraction values). |
| **plot_lamina_roman_labels** | TRUE | On **lamina color scatter** plots only: Roman numerals I–X at estimated lamina positions (centroid-based). No effect on GM/WM or KDE plots. |
| **gm_outline_smooth_sigma_fraction** | 0.03 | GM outline / outline-filled GM: Gaussian σ (px) = fraction × GM vertical height; mask smoothed, boundary at level 0.5. Larger → smoother, outward-shifted GM boundary. |
| **wm_outline_smooth_sigma_fraction** | 0.03 | WM outline / outline-filled WM: same σ scaling; WM mask is **binary-dilated** until all WM pixels lie inside the 0.5 contour, then smoothed. Larger → smoother exterior WM boundary. Independent of GM fraction. |

---

## 3. Region assignment computation

For each cell row:

1. **Segment** → TIFF frame index via `Segments.csv`.
2. **atlas_x** = X / scaling factor; **atlas_y** = Y / scaling factor.
3. Round to integer pixel indices; sample `mask_stack[frame, row, col]` → **atlas_region_id** (0 if out of bounds).
4. Join **Atlas_Regions.csv**:
   - Fine **region_acronym** / **region_name**
   - **lamina_combined_acronym**: finest region mapped to combined Lamina_I–X using `children_IDs` on combined-lamina rows
   - **matter_class**: GM, WM, or other via parent hierarchy

Output: augmented table (all cells, all segments) used for splitting by segment and for plotting.

---

## 4. Per-segment tabular outputs

Written under **outputdir** with `_<segment>` inserted before the extension (e.g. `_L4`).

| Output file | Content |
|-------------|---------|
| **laminar_counts_from_spinalj_match_&lt;seg&gt;.csv** | Two columns: combined lamina acronym, **n_cells** (value counts on segment subset). |
| **cell_loc_spinalj_region_assignments_&lt;seg&gt;.csv** | Per-cell rows for that segment only (original columns + atlas fields). |

---

## 5. Per-segment figure outputs (13 PNGs)

Base names (before `_<segment>`); variants insert `_matter_outline` or `_outline_filled` before `_<segment>`.

### 5.1 GM/WM cell scatter (3)

| File pattern | Backdrop | Cells |
|--------------|----------|--------|
| `cell_locations_on_spinalj_backdrop_*` | Flat GM/WM RGB from region IDs (RH crop) | Scatter |
| `…_matter_outline_*` | Same + smooth GM (black) and WM (dimgray) line outlines | Scatter |
| `…_outline_filled_*` | GM/WM filled from **smoothed** masks (contourf at 0.5), not flat atlas colors | Scatter |

Outline and filled variants use **gm_outline_smooth_sigma_fraction** and **wm_outline_smooth_sigma_fraction** (§2.3).

### 5.2 Lamina color scatter (2)

| File pattern | Notes |
|--------------|--------|
| `cell_locations_on_lamina_color_backdrop_*` | tab10 color per combined lamina I–X; optional Roman labels if **plot_lamina_roman_labels** |
| `…_matter_outline_*` | Same + GM/WM outlines |

No `_outline_filled_` variant for lamina plots.

### 5.3 Laminar bar chart (1)

`laminar_counts_bar_<segment>.png` — bar heights = **percentage of cells in that segment** for each combined lamina (Lamina_I–X, zeros for empty laminae).

### 5.4 KDE — percentile contours (3)

`cell_kde_on_spinalj_backdrop_*` (+ `_matter_outline_`, `_outline_filled_` variants)

See §7.1.

### 5.5 KDE — fraction of maximum (3)

`cell_kde_fraction_of_max_spinalj_backdrop_*` (+ outline variants)

See §7.2.

---

## 6. Backdrop modes (non-KDE spatial plots)

- **Atlas matter:** Pixels colored by **matter_class** (GM gray, WM white, other light gray) from region IDs.
- **Matter outline:** Same flat backdrop + line outlines from smoothed GM/WM fields (0.5 contours).
- **Outline-filled:** Backdrop is smooth GM/WM **patches** (contourf), not per-pixel atlas IDs; cells scatter on top.

KDE plots use the same three backdrop modes; density contours are drawn on top (dark red by default).

---

## 7. Kernel density estimation (KDE) — shared computation

For each segment, only cells with that **segment** label are used.

1. **Fold** atlas (x, y) into right-hemisphere plot coordinates (mirror if x &lt; midline, subtract midline for display width).
2. **Grid:** size = full mask height × **half** mask width (RH crop).
3. **Impulse grid:** +1 at each rounded (x, y) in plot coordinates (in-bounds only).
4. **Smooth:** `scipy.ndimage.gaussian_filter` with  
   **σ_pixels = kernel_width_fraction × GM_vertical_height_pixels**  
   (GM height = span of GM-labeled rows in the segment mask).

Result: **density[row, col]** = smoothed cell count surface (non-negative).

### 7.1 Percentile KDE contours

- Consider only grid values **&gt; 0**.
- For each percentile \(p\) in `[min, max]` step `step`, level = `numpy.percentile(positive_values, p)`.
- Levels are monotonized with `maximum.accumulate` so contours do not cross incorrectly.
- **Interpretation:** Contours enclose regions whose local smoothed cell density equals or exceeds the density threshold that separates the lowest *p*% of **occupied** KDE pixels from the rest. Raising **kernel_width_fraction** spreads mass → contours shift and smooth. Changing percentile range/step changes how many nested contours appear and which density thresholds they represent.

**Plot title** notes σ as % of GM height and the percentile range used.

### 7.2 Fraction-of-maximum KDE contours

- **d_max** = maximum of the full KDE grid (including zeros).
- For each fraction \(f\) in `[min, max]` step `step`, level = **f × d_max**.
- Same monotonicity fix on levels.

**Interpretation:** Contours show where density reaches fixed proportions of the **peak** density in that segment (not relative to the distribution of positive pixels). Useful when the peak hotspot matters more than global spread. **kernel_width_fraction** affects both peak height and spatial extent of contours.

Inline labels (if enabled): percentiles labeled `P&lt;value&gt;`; fraction plots label the fraction (e.g. `0.35`).

---

## 8. GM/WM outline geometry (outline and outline-filled variants)

**GM:** Binary GM mask → Gaussian blur with σ from **gm_outline_smooth_sigma_fraction** → contour or fill at **0.5**.

**WM:** Binary WM mask → optional iterative binary dilation until every WM pixel has smoothed value &gt; 0.499 → Gaussian blur with σ from **wm_outline_smooth_sigma_fraction** → exterior-only WM polyline (inner GM–WM interface loops removed) for line plots; same fields for contourf in filled mode.

Increasing either σ fraction → smoother, less jagged boundaries; GM and WM can be tuned independently.

---

## 9. Lamina label placement

When **plot_lamina_roman_labels** is TRUE, combined laminae **Lamina_I** through **Lamina_X** get label positions from region centroids in the full-cord mask (centroid x, median y along nearest column). Labels are drawn on the RH plot only.

---

## 10. Summary: parameter → visible change

| Change | Primary effect |
|--------|----------------|
| **plot_dpi** | Image sharpness / file size only. |
| **kernel_width_fraction** | All KDE plots: blur width, contour shape and count; σ title on KDE figures. |
| **kde_contour_percentile_*** | Number and height of percentile KDE isolines. |
| **kde_fraction_max_*** | Number and height of fraction-of-max isolines. |
| **show_contour_inline_labels** | KDE contour text on/off. |
| **plot_lamina_roman_labels** | Roman numerals on lamina color scatters only. |
| **gm_outline_smooth_sigma_fraction** | GM line/filled boundary smoothness and offset. |
| **wm_outline_smooth_sigma_fraction** | WM line/filled boundary smoothness and offset (with WM dilate-to-enclose). |
| **atlas_dir** | Which registration/atlas version is used for masks and region tables (must be valid). |
| **cell_csv_path / scaling / Reference section** | Which cells map to which segment and atlas pixels. |
| **outputdir** | Where all CSV/PNG products are written. |

---

## 11. Python environment

- **Conda (recommended):** `conda env create -f environment.yml` then `conda activate spinalj_post_reg_plotting`
- **Pip:** `pip install -r requirements.txt` in a Python 3.11 virtual environment
- **Plot settings example:** `spinal_registration/test_data/spinalj_plot_settings.csv`

Versions were pinned from a working **Python 3.11** analysis stack (NumPy 2.3.3, pandas 2.3.3, SciPy 1.16.2, matplotlib 3.10.7, Pillow 11.3.0).

---

*Aligns with `process_spinalj_by_reference_segment` (explicit `atlas_dir`, plot-settings CSV workflow).*
