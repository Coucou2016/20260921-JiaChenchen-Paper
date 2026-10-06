# Wellington urban-flood super-resolution dataset

Handoff note for another agent. This describes the **derived** dataset under `dataset/`. It does not replace the raw files. Read this before choosing a model, a split, or a loss.

Study area: Wellington, New Zealand. Flood fields come from HiPIMS / UrbanFloodSimulator (GeoClasses). Static fields are the simulator inputs (ESRI ASCII), copied once per resolution from the `100a` case because terrain does not depend on return period.

## What this dataset is for

Guided spatial super-resolution of an **already computed** coarse flood field:

- input: low-resolution water depth (optionally unit-width discharge), plus high-resolution static geography
- target: high-resolution water depth from an independent fine-grid simulation
- default pair: **10 m → 2 m (×5)**, target `h_max`

This is not “rainfall + DEM → depth” (that is what LarNO and U-RNN do). The coarse flood field is an input, not something the model is asked to invent from rainfall alone.

## What is already decided

- Raw inputs were not modified. `wellington-output-data/` (40 NetCDF files) is read-only. `static_geo_data/` is the ASCII source.
- Grids are cropped to one common rectangle and stored as chunked NetCDF-4. A 480 m patch is exactly one chunk at every resolution.
- Train / val / test are **geographic bands**, not a random split of patches. Do not reshuffle.
- The reader returns physical values with nodata as NaN. It does **not** normalize, one-hot encode land use, or apply augmentation.

## Directory layout

```
dataset/
  DATASET.md                  this note
  manifest.json               machine-readable summary
  wellington_sr.py            WellingtonSRDataset
  grids/{2,5,10,20,30}m/
    static.nc                 15 static channels, same for every return period
    flood_20a.nc              h, hux, hvy, h_max
    flood_100a.nc
  rain/
    20a.npy  50a.npy  100a.npy    length-21600 float32 hyetographs
    summary.json
  index/
    patches_480m.json         every 480 m tile, split label, valid/wet fractions
    norm_stats_train.json     mean/std on usable training tiles only
```

There is no `flood_50a.nc`. The 50-year rainfall exists; the 50-year flood simulation was not in the raw output.

## Domain and alignment

All five resolutions share the northwest corner of the original ESRI grids. The stored domain drops a thin southern strip so that 480 m divides every resolution.

| item | value |
|---|---|
| inferred CRS | NZTM2000 / EPSG:2193 (metres; not written inside the source files) |
| west edge | 1743040 m |
| north edge | 5442720 m |
| width × height | 12960 m × 23040 m |
| row 0 | north; y decreases southward |
| x, y variables | cell-centre coordinates, float64, units m |
| nodata | stored as `-9999`; the reader turns this into NaN |
| patch | 480 m × 480 m |
| patch grid | 48 rows × 27 columns = 1296 tiles (index `iy` north→south, `ix` west→east) |

Stored array shapes after the crop (y, x). Time length is 7 for every flood file.

| resolution | shape (y, x) | cells per 480 m patch | chunk size |
|---|---|---|---|
| 30 m | 768 × 432 | 16 × 16 | (16, 16); flood time chunks (1, 16, 16) |
| 20 m | 1152 × 648 | 24 × 24 | (24, 24) |
| 10 m | 2304 × 1296 | 48 × 48 | (48, 48) |
| 5 m | 4608 × 2592 | 96 × 96 | (96, 96) |
| 2 m | 11520 × 6480 | 240 × 240 | (240, 240) |

Nesting check on one land tile (`iy=0`, `ix=20`): the 5×5 block mean of the 2 m DEM matches the 10 m DEM with correlation 1.0 and RMSE about 6×10⁻⁶ m. Coarse and fine static grids are the same terrain. Coarse and fine **flood** grids are not the same field; see below.

Integer scale factors that the reader accepts (`lr_res` divisible by `hr_res`):

| pair | scale |
|---|---|
| 20 m → 10 m, 10 m → 5 m | ×2 |
| 30 m → 10 m | ×3 |
| 20 m → 5 m | ×4 |
| 10 m → 2 m | ×5 |
| 30 m → 5 m | ×6 |
| 20 m → 2 m | ×10 |
| 30 m → 2 m | ×15 |

5 m and 2 m are ×2.5. The reader rejects that pair. Do not bilinear-resize one onto the other and call it an integer-scale sample.

## Flood variables

File: `grids/{res}m/flood_{20a|100a}.nc`.

| variable | dims | units | meaning |
|---|---|---|---|
| `time` | `(time,)` | seconds | 0, 3600, 7200, 10800, 14400, 18000, 21600 |
| `h` | `(time, y, x)` | m | water depth at that hour |
| `hux` | `(time, y, x)` | m²/s | unit-width discharge in x |
| `hvy` | `(time, y, x)` | m²/s | unit-width discharge in y |
| `h_max` | `(y, x)` | m | maximum inundation depth over the run |

`h_max` is the envelope of the simulator’s internal time series. It is deeper and wetter than the maximum of the 7 stored hourly frames. The hourly frames undersample the hydrograph.

`time = 0` is essentially dry (depths at the millimetre level). `WellingtonSRDataset` drops it when `target="h"`. When `target="h_max"` the sample has no time axis; `time_index` in the sample dict is a dummy 0 and does **not** mean the dry frame.

Each resolution is a **separate simulation**, not a downsample of the 2 m run. Measured on maximum depth:

| comparison | correlation | note |
|---|---|---|
| 30 m simulation vs 2 m simulation aggregated to 30 m | about 0.24–0.27 | coarse grid spreads water into a thin film |
| 10 m simulation vs 2 m simulation aggregated to 10 m | about 0.65–0.68 | usable but not a pure interpolation problem |
| nearest-neighbour 30 m → 2 m vs true 2 m | inundation CSI at 0.05 m about 0.21–0.24 | weak baseline |

A model that only sharpens the coarse depth cannot recover the fine drainage pattern. High-resolution terrain has to be an input.

On usable **training** tiles only (`norm_stats_train.json`):

| field | min | max | mean | std |
|---|---|---|---|---|
| 2 m `h_max`, 100a | 0 | 22.78 m | 0.099 m | 0.538 m |
| 2 m `h_max`, 20a | 0 | 20.43 m | 0.076 m | 0.437 m |
| 10 m `h_max`, 100a | 0 | 18.65 m | 0.106 m | 0.517 m |
| 2 m hourly `h`, 100a, frames 1–6 pooled | 0 | 22.78 m | 0.038 m | 0.354 m |

Depth is heavy-tailed: most wet cells are centimetres, a few cells are tens of metres. A plain MSE will be dominated by those peaks.

## Static variables

File: `grids/{res}m/static.nc`. One stack per resolution. Channel order used by the reader:

| index | name | kind | meaning and units |
|---|---|---|---|
| 0 | `DEM` | continuous | elevation, metres. Buildings are **not** burned in. Train tiles: min −1.71, max 444.9, mean 151.3, std 75.9 |
| 1 | `Slope` | continuous | slope derived from that DEM. Train tiles span 0 to 1.56 (dimensionless rise/run, not degrees) |
| 2 | `Aspect_sin` | continuous | sin of aspect |
| 3 | `Aspect_cos` | continuous | cos of aspect |
| 4 | `Curv_plan` | continuous | plan curvature |
| 5 | `Curv_profile` | continuous | profile curvature |
| 6 | `Tpi` | continuous | topographic position index, 100 m neighbourhood |
| 7 | `Twi` | continuous | topographic wetness index |
| 8 | `Dist_building` | continuous | distance to nearest building, metres |
| 9 | `Dist_road` | continuous | distance to nearest road, metres |
| 10 | `Dist_water` | continuous | distance to nearest water, metres |
| 11 | `Infiltration` | continuous | infiltration rate used by the simulator, m/s |
| 12 | `Manning` | continuous | Manning’s n, s·m^(−1/3) |
| 13 | `Landuse` | categorical, codes 1–7 | do not z-score this channel; use an embedding or one-hot |
| 14 | `Building_binary` | binary | 1 = building, 0 = other |

Land-use codes, from the author’s note in `static_geo_data`:

| code | class | Manning n | infiltration (m/s) |
|---|---|---|---|
| 1 | building | 0.5 | 0 |
| 2 | road | 0.013 | 0 |
| 3 | impervious | 0.013 | 0 |
| 4 | pervious | 0.1 | 9.17×10⁻⁷ |
| 5 | missing in the published source data | 0.035 | 4.58×10⁻⁷ |
| 6 | green space | 0.1 | 9.17×10⁻⁷ |
| 7 | water | 0.035 | 0 |

Infiltration is constant per class, not a measured soil map. Class 5 is a gap filled by the author, not a real land-cover class. About 41% of each full grid is valid land; the rest is nodata (sea / outside the catchment mask).

## Rainfall

`rain/{20a,50a,100a}.npy` is a length-21600 float32 series: depth increment per second for 6 hours. Totals: 20a 73.2 mm, 50a 86.4 mm, 100a 96.6 mm. The peak bin is index 10801 (just after 3.0 h). The author’s note said the design peak is at 0.4 of the duration; the file peaks at half the duration. Trust the array.

50a has no flood NetCDF. Do not invent 50a depth targets.

The default SR sample does not include rainfall. Rainfall is only needed if the model is a simulator surrogate (depth from rain + terrain) rather than a super-resolution of the coarse flood field.

## Patch index and splits

`index/patches_480m.json` lists all 1296 tiles. Fields on each patch:

- `iy`, `ix`: tile row and column
- `split`: `train` | `val` | `test` | `buffer`
- `valid_frac`: fraction of the 2 m DEM tile that is not nodata
- `wet_frac.20a`, `wet_frac.100a`: fraction of valid 2 m cells with `h_max > 0.05` m
- `usable`: `valid_frac >= 0.70` and split is not `buffer`

Split is by `iy` only (north to south):

| iy | role | tiles before the valid-fraction cut | usable tiles |
|---|---|---|---|
| 0–29 | train | 810 | 275 |
| 30 | buffer (discard) | 27 | 0 |
| 31–36 | val | 162 | 91 |
| 37 | buffer (discard) | 27 | 0 |
| 38–47 | test | 270 | 121 |

Usable counts × 2 return periods:

| target | train | val | test |
|---|---|---|---|
| `h_max` (one map per event) | 550 | 182 | 242 |
| `h` (6 hourly frames, t = 0 dropped) | 3300 | 1092 | 1452 |

With `include_flow=True` and `target="h"`, each low-resolution sample has 3 channels (`h`, `hux`, `hvy`) instead of 1. The target is still a single depth map.

Independent information is thin: **one city, one terrain, two rainfall events**. Patches from the same event share the storm. The geographic split stops neighbour leakage; it does not create new storms. A reported test score is spatial generalization inside Wellington, not generalization to another city or another return period. Holding out `100a` or `20a` entirely is a separate experiment and is not how the index is built.

`norm_stats_train.json` keys look like `static_2m_DEM` and `flood_100a_2m_h_max`. They are computed on usable training tiles only. Apply them in the training loop. Do not recompute mean and std on val or test.

## How to read one sample

```python
from dataset.wellington_sr import WellingtonSRDataset

ds = WellingtonSRDataset(lr_res=10, hr_res=2, split="train", target="h_max")
sample = ds[0]
```

`sample` keys:

| key | shape for the default 10 m → 2 m, `h_max` | meaning |
|---|---|---|
| `lr` | `(1, 48, 48)` float32 | coarse depth, NaN where nodata |
| `hr` | `(1, 240, 240)` float32 | fine target depth |
| `static` | `(15, 240, 240)` float32 | static stack at `hr_res` (default) or at `lr_res` if `static_res="lr"` |
| `mask` | `(240, 240)` float32 | 1 where both fine DEM and fine depth are finite |
| `scenario` | str | `"20a"` or `"100a"` |
| `time_index` | int | hourly frame 1–6 when `target="h"`; dummy 0 when `target="h_max"` |
| `iy`, `ix` | int | tile indices |
| `scale` | int | `lr_res // hr_res` |
| `lr_names`, `static_names` | list | channel names |

Losses must be multiplied by `mask`. Nodata must not enter the loss as a number.

## Constraints another agent should not undo

1. Do not randomly split patches. Adjacent 480 m tiles share streets and flow paths.
2. Do not treat a coarse flood map as a downsampled fine map, and do not train with bicubic pairs manufactured from the 2 m field unless that is explicitly an ablation.
3. Do not z-score `Landuse`. Codes 1–7 are categories.
4. Do not use the 50-year rain file as a depth target.
5. Do not feed `t = 0` as a flooded frame.
6. Prefer metrics that match the decision: RMSE on wet cells, peak-depth error, and CSI / F1 of inundation at 0.05 m, 0.3 m, and 1.0 m. PSNR and SSIM on the whole tile are secondary because most of the tile is dry or nodata.
7. The practical primary experiment is 10 m → 2 m with a residual against bilinear-upsampled coarse depth, conditioned on the 2 m static stack. ×15 (30 m → 2 m) in one step is a stress test, not the first model.
