# Finite element analysis

This folder contains the 2D plane-stress finite element model used to estimate the mechanical stress distribution across the cross-section of the cranial neural tube (Fig. S9 in the paper).

**This analysis requires a separate Python environment from the main `spatchcocking` package**, because [SolidsPy](https://github.com/AppliedMechanics-EAFIT/SolidsPy) may have conflicting dependencies.

---

## Installation

```bash
pip install -r finite_element/requirements_fem.txt
```

---

## Model description

The cross-sectional geometry of each brain vesicle (forebrain, midbrain, hindbrain) was parametrically defined using radial and thickness profiles measured from the experimental data. These profiles were derived from representative embryos at HH17 and HH20 at three anchor positions along the dorso-ventral axis: dorsal midline (0°), lateral wall (90°), and ventral midline (180°). The geometry is continuously interpolated between anchor points using a Rational Quadratic Bézier scheme with C¹ continuity at the poles.

| Parameter | Value |
|---|---|
| Element type | 4-node isoparametric quadrilateral (plane stress) |
| Mesh resolution | nθ = 80 (circumferential), nr = 20 (radial) |
| Material model | Linear elastic, isotropic |
| Poisson's ratio ν | 0.45 |
| Shear modulus µ | 300 Pa |
| Internal pressure P | 15 Pa (uniform follower load on inner surface) |
| Outer boundary | Traction-free |
| Symmetry BCs | u_x = 0 at dorsal (0°) and ventral (180°) poles |
| Rigid body constraint | Single basal node pinned vertically (u_y = 0) |

Post-processing converts the Cartesian stress tensor into a local polar basis to extract the transverse hoop stress σ_θ. The normalized average membrane stress σ_mem/µ — representing net tensile force per unit length — is integrated across the wall thickness at each angle:

  σ_mem(θ)/µ = (1 / t(θ)·µ) ∫[t_in to t_out] σ_θ(r, θ) dr

All stresses are normalized by the shear modulus µ so that results are independent of absolute tissue stiffness.

---

## Running the analysis

```bash
python finite_element/fem_plane_stress.py
```

The script will:
1. Build the geometry from the anchor parameters defined in Table S1 of the paper
2. Assemble and solve the FEA system
3. Post-process hoop stress and compute membrane stress profiles
4. Save output plots as PDF/PNG

---

## Input geometry parameters

The radial (R) and thickness (t) profiles at each vesicle are defined by the anchor values in `Table S1` of the paper. To apply the model to a different geometry, modify the `GEOMETRY_PARAMS` dictionary at the top of `fem_plane_stress.py`.

---

## Reference

Guarín-Zapata, N., and Gómez, J.D. (2023). SolidsPy: 2D-Finite Element Analysis with Python. Zenodo.
