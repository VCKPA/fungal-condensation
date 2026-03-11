# Figure Panel Guide — Complete Breakdown for Captions and In-Text References

---

## Figure 1: FigureSCHEMATIC

**Layout**: 3×2 photo grid (left) + 2 data panels (right, B and C)

### Panel A — Microscopy and segmentation overview
- **Row 1**: Green fungus colony on hydrogel (left, brightfield, scale bar 2 mm); same field at higher magnification showing condensation droplets around colony (right)
- **Row 2**: Close-up of droplet field near fungal colony boundary (left, scale bar 1 mm); same region with Cellpose segmentation overlay — each droplet individually labeled with random color (right). Sub-labels: (i₁), (i₂)
- **Row 3**: Time evolution showing two additional frames with segmentation overlay. Sub-labels: (i₃), (i₄)
- **Content**: Demonstrates the imaging pipeline from whole-colony view to individual droplet segmentation

### Panel B — Normalized droplet count vs time
- **X-axis**: Time (min), range ~0–20 min
- **Y-axis**: Normalized Count (dimensionless, 0–1)
- **Data**: Multiple colored curves representing different distance zones from the hygroscopic source
- **Legend**: Distance zones in µm (e.g., 800–1200, 1200–1600, 1600–2200, 2200–3000 µm)
- **Key feature**: Count rises as droplets nucleate, peaks at ~10–12 min, then declines as droplets coalesce/evaporate. Near-source zones peak earlier and decline faster.
- **Generating script**: `FigureSchematic/code/step3_figure_panels.py`
- **Output**: `panel_B_normalized_count.svg`

### Panel C — Beysens scaling: median radius vs time
- **X-axis**: Time (min), range ~0–15 min
- **Y-axis**: Median R (µm)
- **Data**: Multiple colored curves (one per distance bin, 300 µm bins), colored by distance (plasma colormap). Colorbar on right showing distance from source.
- **Key feature**: Growth curves for droplets at different distances from source. Near-source bins show suppressed growth; far-field bins grow faster. Reference power-law slopes (β = 1/3 diffusion-limited, β = 1 coalescence) shown.
- **Trial**: 2to1.4 (2:1 NaCl, δ = 900 µm)
- **Generating script**: `FigureSchematic/code/step3_figure_panels.py`
- **Output**: `panel_C_mean_R.svg`

---

## Figure 2: FigureHGAggregate

**Layout**: Row 1 = 3 photos (A–C); Row 2 = schematic (D) + 2×2 data grid (E, F, G, H); Row 3 = 4 survival panels (I, J, K, L)

### Panels A–C — Microscopy of three hydrogel conditions
- **A**: Agar (control, aw = 1.00) — segmentation overlay, dashed gray arc marking source boundary. Label "Agar"
- **B**: 1:1 NaCl (aw = 0.87) — segmentation overlay, visible suppression zone near source. Label "1:1 NaCl"
- **C**: 2:1 NaCl (aw = 0.75) — segmentation overlay, prominent dry zone (δ ≈ 860 µm). Label "2:1 NaCl"
- **Source**: Manual assembly (not code-generated)

### Panel D — Schematic of vapor-sink mechanism
- **Content**: Circular schematic showing hygroscopic source ("Sink") at center, surrounded by concentric rings of droplets decreasing in size toward the source
- **Annotations**: J = −D∇c (Fick's law vapor flux), δ (dry-zone width), ΔR (near–far size difference), droplet radius R
- **Source**: Manual illustration (not code-generated)

### Panel E — R(d) exemplar with shaded Near/Mid/Far zones
- **X-axis**: Distance from source (mm), 0–2.5
- **Y-axis**: Radius R (µm), 0–75
- **Data**: Individual droplets (semi-transparent scatter, rasterized) + binned mean R(d) profiles for two exemplar trials: Agar (green, agar.4) and NaCl (red, 2to1.2)
- **Zone shading**: Near (per-exemplar annotated bracket), Mid (750–1250 µm, orange), Far (≥1500 µm, blue)
- **Annotations**: R_near, R_mid, R_far horizontal reference lines for the NaCl exemplar
- **Key feature**: NaCl shows steep size gradient from near to far; Agar shows relatively flat profile
- **Generating script**: `FigureHGAggregate/code/step6_panels_GH.py` → `plot_panel_G()`
- **Output**: `panels_EH.svg` (left panel)

### Panel F — Dry-zone width δ vs water activity deficit
- **X-axis**: (1 − aw), range 0–0.3
- **Y-axis**: δ (µm), range 0–1000
- **Data**: 15 individual hydrogel trial points, colored by group (Agar green, 1:1 blue, 2:1 red)
- **Regression**: Dashed black OLS line, R² = 0.919
- **Key feature**: Strong linear relationship — more hygroscopic substrates create wider dry zones
- **Generating script**: `FigureHGAggregate/code/step4_panels_BC.py` → `plot_panel_B()`
- **Output**: `panels_FG.svg` (top-left sub-panel)

### Panel G — Near-field vs far-field radius over time
- **X-axis**: Time (min), range ~6–14
- **Y-axis**: Mean radius R (µm), range ~10–60
- **Data**: 4 curves — NaCl far (red squares), NaCl near (red circles), Agar far (green squares), Agar near (green circles). Shaded ΔR band between near and far for each exemplar.
- **Exemplars**: 2to1.2 (NaCl) and agar.4 (Agar)
- **Key feature**: NaCl shows large ΔR (divergence between near and far); Agar shows small ΔR. Both far-field curves grow similarly, but near-field NaCl is strongly suppressed.
- **Generating script**: `FigureHGAggregate/code/step4_panels_BC.py` → `plot_Rt_near_far()`
- **Output**: `panels_FG.svg` (bottom-left sub-panel)

### Panel H — Zone-based size gradient vs water activity deficit
- **X-axis**: (1 − aw), range 0–0.3
- **Y-axis**: (R_far − R_near) / R_mid (dimensionless)
- **Data**: 15 individual hydrogel trial points, colored by group
- **Regression**: Dashed black OLS line, R² = 0.828
- **Zone definitions**: Near = first 5 valid 50-µm bins; Mid = 750–1250 µm; Far ≥ 1500 µm; min 10 droplets/bin
- **Key feature**: Normalized size gradient scales with hygroscopicity
- **Generating script**: `FigureHGAggregate/code/step6_panels_GH.py` → `plot_panel_H()`
- **Output**: `panels_EH.svg` (right panel)

### Panel I — Kaplan–Meier survival curves by distance
- **X-axis**: Time since birth (min), range 0–17
- **Y-axis**: Fraction surviving, 0–1
- **Data**: 4 KM curves for 2:1 NaCl pooled replicates at distance bands: 0.9 mm (navy), 1.5 mm (purple), 2.1 mm (orange), 2.9 mm (gold). Shaded 95% CI bands.
- **Annotation**: τ₅₀ dots marking median survival time on each curve; dashed horizontal line at 0.50
- **Key feature**: Near-source droplets die faster (shorter τ₅₀); far-field droplets survive longer
- **Generating script**: `FigureHGAggregate/code/test_tracking/make_manuscript_panels.py`
- **Output**: `panel_survival.svg`

### Panel J — Median survival time τ₅₀ vs distance from source
- **X-axis**: Distance from source (mm), 0–3.5
- **Y-axis**: Half-life τ₅₀ (min)
- **Data**: Per-group τ₅₀ profiles (Agar green, 1:1 blue, 2:1 red) with error bars (SEM). Large colored dots mark the distance bands from panel I (0.9, 1.5, 2.1, 2.9 mm).
- **Key feature**: Agar τ₅₀ is high and flat (no gradient); 2:1 NaCl τ₅₀ rises steeply with distance
- **Generating script**: `FigureHGAggregate/code/test_tracking/make_manuscript_panels.py`
- **Output**: `panels_IJ.svg` (left panel, labeled I in script)

### Panel K — Survival gradient dτ₅₀/dr vs water activity deficit
- **X-axis**: (1 − aw), range 0–0.3
- **Y-axis**: dτ/dr (min/mm)
- **Data**: 15 individual trial slopes from linear regression of τ₅₀ vs distance
- **Regression**: Dashed OLS line, R² = 0.878
- **Key feature**: Stronger hygroscopic substrates produce steeper survival gradients
- **Generating script**: `FigureHGAggregate/code/test_tracking/make_manuscript_panels.py`
- **Output**: `panels_IJ.svg` (right panel, labeled J in script)

### Panel L — d²-law decomposition of survival gradient
- **X-axis**: (1 − aw), range 0–0.3
- **Y-axis**: d(ln τ₅₀)/dr (mm⁻¹)
- **Data**: Individual trial points (circles = hydrogels, triangles = size-matched). Three regression components:
  - **Total** (solid black line): d(ln τ₅₀)/dr, R² = 0.93
  - **Size channel** (dashed line): d(ln R₀²)/dr, R² = 0.77
  - **Rate channel** (pink shaded area between Total and Size lines): −d(ln K)/dr
- **Key feature**: Decomposes the survival gradient into a size contribution (initial radius gradient) and a rate contribution (evaporation rate gradient). The pink "rate channel" gap grows with (1−aw), showing that both size and rate contribute.
- **Generating script**: `FigureHGAggregate/code/test_tracking/make_manuscript_panels.py`
- **Output**: `panel_L.svg`

---

## Figure 3: FigureFungi

**Layout**: A (photos, top-left) + B (R* overlay, bottom-left) + C (heatmap, center) + D (scatter, top-right) + E (scatter, bottom-right)

### Panel A — Microscopy of fungal species
- **Content**: Two microscopy images showing condensation around fungal colonies on hydrogel substrates (e.g., green fungus and white/black fungus). Scale bar 1 mm.
- **Source**: Manual assembly (not code-generated)

### Panel B — Universal R*(d) profile across all 30 trials
- **X-axis**: Distance from source (mm), 0–2.5
- **Y-axis**: R* = R / R_far (dimensionless), ~0.4–1.1
- **Data**: 30 individual trial profiles (thin transparent lines) + 6 group means with SEM bands (thick lines):
  - 2:1 NaCl (red), 1:1 NaCl (blue), Agar control (gray), Green fungus (green), White fungus (gray), Black fungus (black)
- **Normalization**: R_far = mean of bins > 1500 µm; 200 µm bins; t = 14.5–15.5 min
- **Key feature**: 2:1 NaCl shows strongest suppression near source (lowest R*); Agar shows nearly flat profile; fungi fall between hydrogel extremes
- **Generating script**: `FigureFungi/code/step3_panel_B_universal_Rstar.py`
- **Output**: `panel_B_universal_Rstar.svg`

### Panel C — Heatmap of R* across all 30 trials
- **X-axis**: Distance from source (mm), 0–2.5
- **Y-axis**: Individual trials grouped by condition (Agar, 1:1, 2:1, Green, White, Black), numbered 1–5 within each group
- **Color**: R* = R / R_far (viridis colormap, 0 = dark purple/suppressed, 1 = yellow-green/ambient). Gray = no data (dry zone).
- **Annotations**: Red dashed vertical ticks = δ (raycast dry-zone width) for each trial. Legend: "| = δ"
- **Group labels**: Color-coded text on left axis (Agar green, 1:1 blue, 2:1 red, Green green, White gray, Black black)
- **Binning**: 50 µm bins, min 10 droplets/bin, normalized per trial by top-quartile far-field mean
- **Key feature**: Clear gradient from dark (near source) to bright (far field). 2:1 NaCl shows widest gray/dark zone. δ markers align with the transition.
- **Generating script**: `FigureHGAggregate/code/step5_heatmap.py`
- **Output**: `heatmap_all_trials.svg`

### Panel D — Survival gradient dτ₅₀/dr vs dry-zone width δ (all 30 trials)
- **X-axis**: δ (µm), range 0–1100
- **Y-axis**: dτ₅₀/dr (min mm⁻¹)
- **Data**: 30 individual trial points (transparent, colored by group: circles for hydrogels, diamonds for fungi) + 6 group means ± SEM (large symbols with error bars). Two regression lines:
  - **Overall** (gray solid): R² = 0.81
  - **Size-matched** (red solid): R² = 0.85 (IQR-band size control)
- **Key feature**: Universal relationship across fungi AND hydrogels — trials with wider dry zones have steeper survival gradients regardless of the biological/chemical identity of the source
- **Generating script**: `FigureFungi/code/step5_universal_panels.py` → `make_panel_G()`
- **Output**: `panel_D.svg`

### Panel E — Zone-based size gradient vs dry-zone width δ (all 30 trials)
- **X-axis**: δ (µm), range 0–1100
- **Y-axis**: (R_far − R_near) / R_mid (dimensionless)
- **Data**: 30 individual trial points (transparent) + 6 group means ± SEM. Single regression line (gray solid), R² = 0.66, n = 30.
- **Zone definitions**: Near = first 5 valid 50-µm bins; Mid = 750–1250 µm; Far ≥ 1500 µm
- **Key feature**: Size gradient also correlates universally with δ, confirming that dry-zone width captures the strength of the vapor-sink effect on both survival and size
- **Generating script**: `FigureFungi/code/step5_universal_panels.py` → `make_panel_H()`
- **Output**: `panel_E.svg`

---

## Figure 4: FigureRSR

**Layout**: A (leaf photos, left column) + B (6 scatter subplots, right, 3×2 grid)

### Panel A — Leaf condensation photographs
- **Content**: Two photographs of real leaves with visible condensation droplets and fungal colonies ("Sink" labeled). Dashed rectangles mark the analysis region. Scale bars: 3 mm each.
- **Top**: Healthy leaf with condensation field around fungal colony
- **Bottom**: Another leaf (healthy or diseased) with similar condensation pattern
- **Source**: Manual assembly (not code-generated)

### Panel B — Droplet radius vs distance from fungal boundary (6 trials)
- **Layout**: 3 rows × 2 columns. Left column = Healthy (RSR1, RSR2, RSR7). Right column = Diseased (RSRDiseased3, RSRDiseased5, RSRDiseased6). Column headers: "Healthy" and "Diseased".
- **X-axis (shared)**: Distance from boundary (mm), range ~0–6
- **Y-axis (shared)**: r_eq (mm), range 0–0.25
- **Data**: Individual droplets (gray scatter, ~120–310 per trial) + red linear regression line
- **Key feature**: All 6 trials show positive slope (dR/dr > 0), confirming that even on real leaves, droplets are smaller near the fungal vapor sink and larger far away. Mean dR/dr = 10.5 ± 1.2 µm/mm across 6 trials; mean Pearson r = 0.57 ± 0.04.
- **Generating script**: `FigureRSR/code/step1_figure_RSR.py`
- **Output**: `panel_B_RSR_scatter.svg`

---

## Summary: Script → Panel Mapping

| Figure | Panel | Content | Script | Output file |
|--------|-------|---------|--------|------------|
| 1 | A | Photos + segmentation | Manual | — |
| 1 | B | Normalized count vs time | `FigureSchematic/step3_figure_panels.py` | `panel_B_normalized_count` |
| 1 | C | Beysens R(t) scaling | `FigureSchematic/step3_figure_panels.py` | `panel_C_mean_R` |
| 2 | A–C | Hydrogel photos | Manual | — |
| 2 | D | Vapor-sink schematic | Manual | — |
| 2 | E | R(d) with zone shading | `FigureHGAggregate/step6_panels_GH.py` | `panels_EH` (left) |
| 2 | F | δ vs (1−aw) | `FigureHGAggregate/step4_panels_BC.py` | `panels_FG` (top-left) |
| 2 | G | R(t) near/far | `FigureHGAggregate/step4_panels_BC.py` | `panels_FG` (bottom-left) |
| 2 | H | Zone metric vs (1−aw) | `FigureHGAggregate/step6_panels_GH.py` | `panels_EH` (right) |
| 2 | I | KM survival curves | `test_tracking/make_manuscript_panels.py` | `panel_survival` |
| 2 | J | τ₅₀ vs distance | `test_tracking/make_manuscript_panels.py` | `panels_IJ` (left) |
| 2 | K | dτ/dr vs (1−aw) | `test_tracking/make_manuscript_panels.py` | `panels_IJ` (right) |
| 2 | L | d²-law decomposition | `test_tracking/make_manuscript_panels.py` | `panel_L` |
| 3 | A | Fungal colony photos | Manual | — |
| 3 | B | Universal R*(d) overlay | `FigureFungi/step3_panel_B_universal_Rstar.py` | `panel_B_universal_Rstar` |
| 3 | C | R* heatmap (30 trials) | `FigureHGAggregate/step5_heatmap.py` | `heatmap_all_trials` |
| 3 | D | dτ₅₀/dr vs δ (universal) | `FigureFungi/step5_universal_panels.py` | `panel_D` |
| 3 | E | Zone metric vs δ | `FigureFungi/step5_universal_panels.py` | `panel_E` |
| 4 | A | Leaf photos | Manual | — |
| 4 | B | r_eq vs distance (6 trials) | `FigureRSR/step1_figure_RSR.py` | `panel_B_RSR_scatter` |

## Key Statistics for In-Text References

| Panel | Statistic | Value |
|-------|-----------|-------|
| 2F | δ vs (1−aw) R² | 0.919 |
| 2H | Zone metric vs (1−aw) R² | 0.828 |
| 2K | dτ/dr vs (1−aw) R² | 0.878 |
| 2L | d(ln τ₅₀)/dr Total R² | 0.93 |
| 2L | d(ln R₀²)/dr Size channel R² | 0.77 |
| 3D | dτ₅₀/dr vs δ Overall R² | 0.81 |
| 3D | dτ₅₀/dr vs δ Size-matched R² | 0.85 |
| 3E | Zone metric vs δ R² | 0.66, n = 30 |
| 4B | Mean dR/dr | 10.5 ± 1.2 µm/mm |
| 4B | Mean Pearson r | 0.57 ± 0.04 |
