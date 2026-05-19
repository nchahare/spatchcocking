# CSV data files

This folder contains the per-embryo measurement data used to generate the statistical figures in the paper. These CSVs allow readers to reproduce all plots without re-running the full image analysis pipeline.

## Files

| File | Contents | Paper figures |
|---|---|---|
| *(to be populated)* | | |

> **Note:** This README will be updated when CSV files are added. The column descriptions below apply once files are present.

## Common columns

Most CSV files share these columns:

| Column | Description |
|---|---|
| `stage` | Developmental stage (`HH17` or `HH20`) |
| `embryo_id` | Embryo identifier (e.g., `E1`–`E7`) |
| `compartment` | Brain region (`FB` = forebrain, `MB` = midbrain, `HB` = hindbrain) |

## Loading in Python

```python
import pandas as pd
df = pd.read_csv("data/csv/lumen_measurements.csv")
```

## Statistical methods

All group comparisons between stages and compartments were performed using two-sample Welch's t-tests (`scipy.stats.ttest_ind`). Correlations between tissue properties and curvature used Pearson's r (`scipy.stats.pearsonr`). Cross-stage DV pattern comparisons used Spearman's ρ (`scipy.stats.spearmanr`). See Methods section of the paper for full details.

## Reproduction

`notebooks/06_figure_plots.ipynb` reads these CSV files and reproduces the statistical plots from the paper.
