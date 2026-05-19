# CSV data files

Pre-computed measurements used to reproduce the statistical figures in the paper.
All files are comma-separated (UTF-8).  Run `notebooks/06_figure_plots.py` to
regenerate all plots.

---

## `spatchcocked_measurements.csv`

Vertex-level measurements extracted from the spatchcocked lumen meshes
(one row per mesh vertex per embryo).  Used by Figs 3–6 and Supp Fig 1.

| Column | Units | Description |
|---|---|---|
| `stage` | — | Developmental stage (`hh17` or `hh20`) |
| `timepoint` | — | Embryo identifier |
| `radius` | µm | Local lumen radius at this vertex |
| `norm_height` | 0–1 | Normalised rostral-caudal position (0 = rostral, 1 = caudal) |
| `angle_degrees` | ° | Dorso-ventral angle (0° = dorsal midline, ±180° = ventral) |
| `thickness` | µm | Apico-basal wall thickness (lumen to basal surface) |
| `phh3` | cell count | Local pHH3+ mitotic cell density within R = 100 µm |
| `Gauss_Curvature` | µm⁻² | Gaussian curvature *K* |
| `Mean_Curvature` | µm⁻¹ | Mean curvature *H* |
| `K1` | µm⁻¹ | Maximum principal curvature |
| `K2` | µm⁻¹ | Minimum principal curvature |

**Compartment assignment** (applied in the plotting script):
- HH17: Forebrain `norm_height > 0.80`, Midbrain `0.60–0.80`, Hindbrain `< 0.60`
- HH20: Forebrain `norm_height > 0.75`, Midbrain `0.55–0.75`, Hindbrain `< 0.55`

---

## `cross_section_area.csv`

Cross-sectional lumen area sampled along the rostral-caudal axis.  Used by Fig 1j.

| Column | Units | Description |
|---|---|---|
| `z_grid` | µm | Position along the rostral-caudal axis |
| `area` | µm² | Lumen cross-section area |
| `stage` | — | `HH17` or `HH20` |
| `type` | — | `individual` = per-embryo; `mean` = group mean |
| `sample_id` | — | Embryo identifier |

---

## `compartment_lengths.csv`

Lumen length and compartment boundary positions per embryo.  Used by Fig 1e and Supp Fig 1.

| Column | Units | Description |
|---|---|---|
| `stage` | — | `hh17` or `hh20` |
| `end` | µm | Total lumen length |
| `mhb` | µm | Midbrain–hindbrain boundary position along RC axis |
| `fmb` | µm | Forebrain–midbrain boundary position along RC axis |

Compartment lengths: Hindbrain = `mhb`, Midbrain = `fmb − mhb`, Forebrain = `end − fmb`.

---

## `lumen_geometry.csv`

Surface area and volume of each brain vesicle per embryo.  Used by Figs 1f–i and Supp Fig 1.

| Column | Units | Description |
|---|---|---|
| `Sample_ID` | — | Embryo identifier |
| `Stage` | — | `HH17` or `HH20` |
| `Region` | — | `Forebrain`, `Midbrain`, or `Hindbrain` |
| `Area` | µm² | Lumen surface area |
| `Volume` | µm³ | Lumen volume |

---

## Statistical methods

All pairwise comparisons: Welch's t-test (`scipy.stats.ttest_ind`).  
Significance: `*` p < 0.05, `**` p < 0.01, `***` p < 0.001, `ns` not significant.  
Thickness–curvature correlations: Pearson *r* on quantile-binned data (10 bins per stage).  
Cross-stage spatial patterns: Spearman ρ on 40 equal-width DV bins.
