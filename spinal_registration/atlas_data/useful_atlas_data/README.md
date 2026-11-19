# SpinalJ atlas inputs (bundled with this repository)

This directory contains the minimum atlas files required by the plotting pipeline:

| File | Description |
|------|-------------|
| `Atlas_Regions.csv` | Region metadata (id, acronym, name, parent_ID, children_IDs) |
| `Segments.csv` | Maps spinal segment labels to TIFF frame indices |
| `Annotation_Segments_Only.tif` | Multi-page TIFF: one region-ID mask per segment |

The pipeline validates that all three files exist under the path you pass as `atlas_dir` (default in the example caller: this folder).

To use a different atlas version, replace these files with your own SpinalJ / registration bundle exports while keeping the same filenames.