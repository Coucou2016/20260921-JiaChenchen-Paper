# 20260921-JiaChenchen-Paper 全面审稿与代码级修改方案

> 审查对象：<https://github.com/Coucou2016/20260921-JiaChenchen-Paper>  
> 审查日期：2026-10-06  
> 审查范围：论文/研究报告、代码、配置、公开结果数据、图表、统计推断、数据组织、可复现性、仓库工程质量。  
> 总体建议：**Major Revision（大修）**。当前工作已有较完整的数据处理、训练、诊断和结果留痕，具备形成论文的基础；但在统计独立性、指标聚合、消融实验定义、多尺度训练实现、基线完整性及论文论证边界方面存在会直接影响结论可信度的硬问题。建议先修正 P0 级问题并重算核心结果，再进入论文定稿。

---

## 1. 审查结论摘要

这个项目的优点不是“只有模型”，而是已经形成了相对完整的工程链条：Wellington 多分辨率洪水场、地形/土地利用等高分辨率静态信息、10 m→2 m 主任务、HydroGeo-SRNO、深水误差诊断、空间统计、训练轨迹、图件和大量机器结果均有保留。仓库的 `FILE_INDEX.md` 显示当前扁平镜像包含 715 个文件、约 148.5 MB，并保留报告、56 张报告图、430 个机器结果/小型网格文件和主要源码。这一点对修稿非常有利。

但是，当前版本如果直接按科研论文投稿，我认为最危险的并不是某一个数值偏低，而是以下几个证据链问题：

1. **统计显著性存在伪重复（pseudoreplication）**：把同一条训练轨迹的连续 20 个 epoch 当作 20 个独立重复做配对 t 检验；又把同一城市、同两场降雨下的 242 个空间瓦片当作 iid 样本 bootstrap，而报告本身又检测到较强空间自相关。这会让自由度、p 值和置信区间显著偏乐观。
2. **核心指标的聚合方式与通常论文语义不一致**：`metrics/aggregation.py` 对每个 batch/tile 的 RMSE、CSI、F1、体积相对误差直接做算术平均。V0 的 batch size=1，因此论文中的“RMSE/CSI/F1”实质上是 tile-macro average，而不是整个验证/测试域上的全局 RMSE 或全局混淆矩阵指标。两者不是同一个 estimand。
3. **消融 A4/A6 存在实现层面的失效**：A4 关闭 residual 后，主模型直接把 `delta_z` 置零，深度输出只剩固定双线性底图；A6 虽然打开 wet head，却仍使用 `masked_l1`，因此 wet head 不进入 loss，不能据此归因 wet head 的贡献。
4. **“arbitrary-scale” 部分当前没有形成可被论文接受的训练/验证证据**：`scripts/train_multiscale.py` 并未执行配置中的 200 epochs、batch_size=2、AMP 等，而是固定 batch_size=1、固定 AdamW(lr=1e-4)、默认 `FloodLoss()`、`use_amp=False`，只遍历一遍 DataLoader；同时配置把 `[5,2]` 标为“core arbitrary-scale test”，却又放进训练 pairs 中。
5. **正式 benchmark 结果只有 nearest / bilinear**：仓库虽有 EDSR、RCAN、ResUNet、SRNO single 等实现入口，但当前 `outputs/benchmarks/table_test.json` 公开结果只有 B0 nearest 和 B1 bilinear。论文若要主张方法优于学习型 SR/神经算子基线，证据不够。
6. **论文主文本缺少正式相关工作与参考文献体系**：当前 `report.md` 更像研究报告/审计报告，未检索到“参考文献”部分。对于以 SRNO、连续隐式表示、洪水超分辨率为核心的方法论文，这是投稿级硬缺口。
7. **若干实现细节会系统性影响边界、极值和可解释性**：包括 nodata 被置零后直接 bilinear 插值、Sobel 边界 loss 在 nodata 边界制造伪边缘、focal alpha 实现不具备正负类平衡语义、PSNR 使用每 tile 自适应动态范围、近似 SSIM 实际是全局统计量而非标准窗口 SSIM。

综合而言：**不建议直接在现有结果上继续“润色论文”；应先进行一次指标/统计/消融/多尺度训练的技术性修复。** 多数 P0 问题可以在不改变总体研究路线的情况下修复。

---

## 2. 审查范围与可验证边界

### 2.1 已直接审查的内容

重点核对了以下公开文件/结果：

- `report.md` / `report_brief.md`
- `README.md`, `FILE_INDEX.md`, `DATA_AND_BINARY_NOTICE.md`
- `dataset/DATASET.md`
- `dataset/wellington_fixed_sr.py`
- `dataset/wellington_multiscale.py`
- `models/hydrogeo_srno.py`
- `models/diffusion/residual_ldm.py`
- `losses/flood_loss.py`
- `metrics/flood_metrics.py`
- `metrics/aggregation.py`
- `scripts/train_fixed.py`
- `scripts/train_multiscale.py`
- `configs/v0_10m2m_hmax.yaml`
- `configs/v0_10m2m_hmax_ablation.yaml`
- `configs/v1_arbitrary_scale.yaml`
- `outputs/benchmarks/*`
- 公开结果/诊断/图件索引

### 2.2 本次无法做的事情

仓库说明明确指出，大体量原始 NetCDF、完整静态栅格、部分 checkpoint/大型中间结果未全部放入公开镜像。因此本次能够完成的是：

- 源码静态审查；
- 配置与代码一致性审查；
- 报告结论与公开结果文件的一致性审查；
- 统计设计审查；
- 结果解释边界审查；
- 公开机器结果的结构性核对。

**不能把本次审查描述为“完整重新运行了 41 GB 原始数据上的全部训练实验”。** 后续修稿时也建议在论文中明确区分“已公开的可审计结果”和“需要原始数据重新训练得到的结果”。

---

# Part I. P0 级问题：投稿前必须修

## P0-1. 当前显著性检验把连续 epoch 当作独立重复，属于伪重复

### 现状

报告用连续 20 个 fine-tune epoch 构造配对差值，并执行配对 t 检验、Cohen's d 和 bootstrap CI。问题在于：

- epoch 1、epoch 2、…、epoch 20 来自**同一条优化轨迹**；
- 参数状态强依赖上一个 epoch；
- 它们不是 20 次独立训练，也不是 20 个独立实验单位；
- “按 epoch 配对”可以对齐训练进度，但不能创造统计独立性。

因此 `df=19` 的配对 t 检验不能作为“独立重复实验显著”的证据。这里的 d 更接近“同一训练轨迹中标准化后的轨迹差异”，而不是通常论文语义下的跨重复效应量。

### 为什么严重

如果审稿人发现连续 checkpoint 被当作 n=20 重复，很容易直接质疑所有显著性结果。尤其报告还用这些结果强化“deep-water fine-tuning statistically significant”等结论，会放大问题。

### 推荐修复

主统计单位改成 **independent random seed**。

最低建议：5 个 seed；算力紧张时至少 3 个，但论文里明确属于小样本重复。

固定一个**事先定义的 checkpoint 选择规则**，例如：

- 每个 seed 都训练相同总 epoch；
- 只用 validation 选择 checkpoint；
- test 只评估一次最终选定 checkpoint；
- deep-finetune 与 control 使用相同初始化 checkpoint 和 seed，形成真正的 paired-by-seed 设计。

建议输出：

```text
seed = [11, 22, 33, 44, 55]
control(seed_i)  -> metric_i
deep_ft(seed_i)  -> metric_i
paired_delta_i   = deep_ft - control
```

主文报告：

- mean delta；
- SD；
- 95% CI；
- paired permutation test 或 paired t-test（n 足够且差值近似合理时）；
- 不再把 epoch 当 replicate。

### 建议新增脚本

`analysis/seed_level_inference.py`

```python
from __future__ import annotations
import numpy as np
from scipy.stats import ttest_rel


def paired_seed_stats(control, treatment, n_boot=20000, seed=20261006):
    control = np.asarray(control, dtype=float)
    treatment = np.asarray(treatment, dtype=float)
    if control.shape != treatment.shape:
        raise ValueError("paired arrays must have the same shape")
    if control.ndim != 1 or control.size < 3:
        raise ValueError("need >=3 independent seeds")

    d = treatment - control
    t = ttest_rel(treatment, control)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boot = d[idx].mean(axis=1)
    ci = np.quantile(boot, [0.025, 0.975])

    sd = d.std(ddof=1)
    dz = d.mean() / sd if sd > 0 else np.nan

    return {
        "n_seeds": len(d),
        "mean_delta": float(d.mean()),
        "sd_delta": float(sd),
        "ci95_low": float(ci[0]),
        "ci95_high": float(ci[1]),
        "t": float(t.statistic),
        "p": float(t.pvalue),
        "cohen_dz": float(dz),
    }
```

### 论文文字应改

不要再写：

> 20 paired epochs demonstrate statistically significant improvement ...

改为：

> The epoch-wise trajectory analysis is treated as an optimization diagnostic rather than an inferential replicate. Statistical inference is based on independently seeded training runs paired by seed.

---

## P0-2. 242 个 test tiles 不能做普通 iid bootstrap

### 现状

测试集是 Wellington 内部按空间带留出的 242 个样本（两个 return-period scenario 下的 tile），而报告自身又给出了明显空间自相关。普通按 tile 随机有放回抽样相当于假定这些 tile 可交换/近独立，这与空间相关结构矛盾。

数据文档还明确说明：

- 只有一个城市；
- 只有 20a、100a 两个 flood scenarios；
- geographic split 防止相邻训练/测试泄漏；
- test 表示 Wellington 内部的空间外推，而**不是**新城市、新 return period 或独立 storm 泛化。

### 修复方案 A：空间 block bootstrap

优先根据现有 `fig82_variogram_scale.png` 或残差变异函数估计 practical range，然后选择 block 尺度。

如果暂时无法稳定估计 range，至少做 block size sensitivity：

```text
1×1 tile
2×2 tiles
3×3 tiles
4×4 tiles
```

主结论必须在合理 block 范围内稳健。

建议代码：

```python
import numpy as np
import pandas as pd


def add_spatial_blocks(df: pd.DataFrame, block_rows: int, block_cols: int):
    out = df.copy()
    out["block_y"] = out["iy"] // block_rows
    out["block_x"] = out["ix"] // block_cols
    # scenario 不跨事件混抽
    out["cluster"] = (
        out["scenario"].astype(str) + "_" +
        out["block_y"].astype(str) + "_" +
        out["block_x"].astype(str)
    )
    return out


def spatial_block_bootstrap(df, value_col, n_boot=10000, seed=42):
    rng = np.random.default_rng(seed)
    clusters = df["cluster"].unique()
    vals = []

    grouped = {c: df.loc[df.cluster == c, value_col].to_numpy()
               for c in clusters}

    for _ in range(n_boot):
        chosen = rng.choice(clusters, size=len(clusters), replace=True)
        x = np.concatenate([grouped[c] for c in chosen])
        vals.append(np.nanmean(x))

    vals = np.asarray(vals)
    return np.quantile(vals, [0.025, 0.5, 0.975])
```

### 修复方案 B：空间交叉验证

比单一 north/south strip 更稳健的做法：

- 以空间 block 构造 4–5 folds；
- 每次留出一个空间块组；
- fold 间存在缓冲带；
- 主要模型和强基线在同一 folds 上评估。

可参考 Valavi et al. 的 blockCV 思路：对空间结构数据使用随机 CV 会低估泛化误差，应进行空间分块。

### 论文中必须收缩的表述

把：

> geographically independent test set

建议写成：

> a geographically held-out set within Wellington, separated from the training bands by spatial buffers

避免把它写成“external validation”或跨城市泛化。

---

## P0-3. 指标聚合改变了指标定义：当前多数结果是 tile-macro，而不是 global-domain metric

### 现状

`metrics/aggregation.py`：

```python
return {k: sums[k] / counts[k] for k in sums if counts[k] > 0}
```

也就是先对每一个 batch 计算 RMSE/CSI/F1/VolumeRelativeError，再对 batch 指标平均。

主配置 `batch_size: 1`，所以本质是：

```text
metric_reported = mean(metric(tile_i))
```

这与下列常见定义不同：

```text
global RMSE = sqrt(sum(all pixel squared errors) / sum(all valid pixels))
global CSI  = sum(TP) / [sum(TP)+sum(FP)+sum(FN)]
```

特别是：

- 大 tile/湿区多的 tile 与小湿区 tile 权重完全相同；
- dry/near-dry tile 的相对面积误差、体积误差可能极端；
- macro CSI 与 pooled CSI 可差很多。

### 推荐：同时报告两类指标，并清楚命名

主文：

1. **domain-pooled**：代表全测试域总体误差；
2. **tile-macro**：代表“典型 tile”误差，用于空间分布/统计分析。

建议命名：

```text
RMSE_domain
RMSE_tile_macro
CSI005_domain
CSI005_tile_macro
VolumeBias_domain
VolumeRE_tile_macro
```

### 建议新增 `FloodMetricAccumulator`

```python
from dataclasses import dataclass, field
import math
import torch

@dataclass
class FloodMetricAccumulator:
    wet_thr: float = 0.05
    sse_all: float = 0.0
    n_all: int = 0
    sse_wet: float = 0.0
    sae_wet: float = 0.0
    n_wet: int = 0
    tp: dict = field(default_factory=lambda: {0.05: 0, 0.30: 0, 1.00: 0})
    fp: dict = field(default_factory=lambda: {0.05: 0, 0.30: 0, 1.00: 0})
    fn: dict = field(default_factory=lambda: {0.05: 0, 0.30: 0, 1.00: 0})
    pred_volume: float = 0.0
    true_volume: float = 0.0

    @torch.no_grad()
    def update(self, pred, target, mask):
        p = pred[:, 0] if pred.ndim == 4 else pred
        t = target[:, 0] if target.ndim == 4 else target
        m = mask[:, 0].bool() if mask.ndim == 4 else mask.bool()

        e = p[m] - t[m]
        self.sse_all += float((e * e).sum())
        self.n_all += int(m.sum())

        wet = m & (t > self.wet_thr)
        ew = p[wet] - t[wet]
        self.sse_wet += float((ew * ew).sum())
        self.sae_wet += float(ew.abs().sum())
        self.n_wet += int(wet.sum())

        for thr in self.tp:
            pb = (p > thr) & m
            tb = (t > thr) & m
            self.tp[thr] += int((pb & tb).sum())
            self.fp[thr] += int((pb & ~tb & m).sum())
            self.fn[thr] += int((~pb & tb & m).sum())

        self.pred_volume += float(p[m].clamp_min(0).sum())
        self.true_volume += float(t[m].clamp_min(0).sum())

    def compute(self):
        out = {
            "RMSE_all_domain": math.sqrt(self.sse_all / max(self.n_all, 1)),
            "RMSE_wet_domain": math.sqrt(self.sse_wet / max(self.n_wet, 1)),
            "MAE_wet_domain": self.sae_wet / max(self.n_wet, 1),
            "VolumeRelativeError_domain": abs(self.pred_volume - self.true_volume)
                / max(self.true_volume, 1e-12),
        }
        for thr in self.tp:
            tp, fp, fn = self.tp[thr], self.fp[thr], self.fn[thr]
            out[f"CSI_{thr:.2f}_domain"] = tp / max(tp + fp + fn, 1)
            out[f"F1_{thr:.2f}_domain"] = 2 * tp / max(2 * tp + fp + fn, 1)
        return out
```

### 重算要求

这是**无需重训练即可完成**的优先修复：直接重新跑 validation/test evaluation 即可。

---

## P0-4. 消融矩阵 A4 和 A6 不是有效的模块消融

### A4 的问题

配置：

```yaml
A4:
  geo_mode: all
  model:
    predict_residual: false
    predict_wet: false
  loss: {name: masked_l1}
```

模型代码：

```python
delta_z = self.residual_head(feat)
if not self.predict_residual:
    delta_z = torch.zeros_like(delta_z)
...
z_pred = z_base + delta_z
```

因此 `predict_residual=false` 时，网络计算出的 feature 完全不进入 depth 输出，`depth` 等于固定 bilinear base。A4 不能代表“全静态信息但不用 residual learning 的直接深度回归模型”，而更接近“固定双线性插值器”。

如果训练 loss 只依赖 `depth`，A4 甚至可能不存在有效的可学习深度路径。

### 正确做法

给模型增加真正的 direct-depth head：

```python
self.direct_head = nn.Conv2d(operator_width, 1, 1)
```

forward：

```python
if self.predict_residual:
    delta_z = self.residual_head(feat)
    z_pred = z_base + delta_z
    h_pred = depth_decode(z_pred, self.depth_ref)
else:
    # 直接预测 log-depth，而不是把 residual 设为 0
    z_pred = self.direct_head(feat)
    h_pred = depth_decode(z_pred, self.depth_ref)
    delta_z = z_pred - z_base
```

然后 A4 vs A5 才能回答：

> 在相同输入/相同 backbone/相同 loss 下，residual parameterization 是否优于 direct-depth parameterization？

### A6 的问题

配置：

```yaml
A6:
  predict_residual: true
  predict_wet: true
  loss: {name: masked_l1}
```

`MaskedL1Loss` 只读取：

```python
pred = output["depth"]
```

不读取 `wet_logits`。因此 wet head 虽存在，但没有训练梯度。A6 不能回答“wet head 是否有效”。

### 正确消融矩阵

建议拆成三个互相正交的系列，而不是 A0–A7 连续叠加：

#### G：地理信息消融

所有模型都保持 residual + 同一 loss：

```text
G0 LR only
G1 LR + DEM
G2 LR + DEM + building
G3 LR + topo
G4 LR + all geo
```

#### R：参数化消融

```text
R0 direct-depth head
R1 residual log-depth head
```

其余完全一致。

#### W：wet head 消融

```text
W0 no wet head, w_wet=0
W1 wet head, w_wet=0.30
```

其余 loss 分量完全一致。

#### L：loss 组成消融

```text
L0 depth only
L1 + wet
L2 + log
L3 + boundary
L4 + extreme
L5 + deep
```

这样才能把每一个 improvement 归因给单一因素。

---

## P0-5. “arbitrary-scale” 配置与训练实现不一致，而且 5→2 被同时作为训练 pair 和“核心测试”

### 配置矛盾

`configs/v1_arbitrary_scale.yaml`：

```yaml
pairs:
  - [20, 10]
  - [10, 5]
  - [30, 10]
  - [20, 5]
  - [10, 2]
  - [30, 5]
  - [5, 2] # ×2.5 fractional — core arbitrary-scale test
```

如果 `[5,2]` 在训练集 pair 中，就不能再把它称为 unseen / zero-shot arbitrary-scale test。

### 训练脚本更严重

`train_multiscale.py` 当前实际执行：

```python
loader = DataLoader(..., batch_size=1, ...)
opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
crit = FloodLoss()
for batch in loader:
    stats = train_step(... use_amp=False)
...
torch.save(... "last_multiscale.pt")
```

而 YAML 写的是：

```yaml
batch_size: 2
epochs: 200
amp: true
```

当前代码没有 epoch loop、没有 validation、没有 scheduler、没有 best checkpoint、没有 per-scale evaluation，不能形成论文级 V1 证据。

### 修复配置

```yaml
dataset:
  train_pairs:
    - [20, 10]
    - [10, 5]
    - [30, 10]
    - [20, 5]
    - [10, 2]
    - [30, 5]

  eval_seen_pairs:
    - [20, 10]
    - [10, 5]
    - [30, 10]

  eval_unseen_pairs:
    - [5, 2]     # fractional ×2.5
    - [20, 2]    # ×10
    - [30, 2]    # ×15
```

### 训练脚本必须至少重构为

```python
for epoch in range(1, epochs + 1):
    model.train()
    for batch in train_loader:
        train_step(...)

    seen = evaluate_pairs(model, eval_seen_pairs)
    unseen = evaluate_pairs(model, eval_unseen_pairs)

    score = np.mean([m["CSI_005_domain"] for m in seen.values()])
    worst = min(m["CSI_005_domain"] for m in seen.values())

    save_last(...)
    if score > best_mean:
        save_best_mean(...)
    if worst > best_worst:
        save_best_worst(...)
```

### 多分辨率 batch 的实现建议

不同 LR/HR spatial shape 无法直接 stack 时，不要永久固定 batch=1。实现 pair-homogeneous batch sampler：

```python
from collections import defaultdict
import random

class ResolutionPairBatchSampler:
    def __init__(self, dataset, batch_size, shuffle=True):
        groups = defaultdict(list)
        for i, meta in enumerate(dataset.sample_meta):
            groups[(meta["lr_res"], meta["hr_res"])].append(i)
        self.groups = groups
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        batches = []
        for _, ids in self.groups.items():
            ids = ids.copy()
            if self.shuffle:
                random.shuffle(ids)
            for j in range(0, len(ids), self.batch_size):
                batches.append(ids[j:j+self.batch_size])
        if self.shuffle:
            random.shuffle(batches)
        yield from batches
```

---

## P0-6. 论文目前没有完成有竞争力的学习型基线比较

正式 benchmark `table_test.json` 当前只有：

- B0 nearest；
- B1 bilinear。

仓库虽然有：

- EDSR；
- RCAN；
- ResUNet；
- RSwinUNet；
- `srno_single.py`；

但没有看到这些模型进入同一正式 test benchmark 表。

### 最低投稿配置

建议至少包含：

| 类别 | 模型 | 输入公平性 |
|---|---|---|
| interpolation | nearest | LR depth |
| interpolation | bilinear | LR depth |
| CNN SR | EDSR | LR depth |
| CNN/U-Net | ResUNet | LR depth；另做 geo-guided 版本 |
| arbitrary SR | LIIF-style | LR depth |
| neural operator SR | SRNO | LR depth |
| proposed | HydroGeo-SRNO | LR depth + HR geography |

如算力允许，再加 RCAN / SwinIR-like baseline。

### 公平比较的关键

必须分两组表：

#### Table A：LR-only capacity comparison

所有方法只看 LR flood depth。

这样回答：

> HydroGeo-SRNO 的 backbone/continuous operator 本身相比普通 SR 方法是否有效？

#### Table B：geo-guided comparison

给所有能够接纳地理信息的模型相同 HR geography。

这样回答：

> 提升到底来自 geography，还是来自 proposed architecture？

否则把“HydroGeo-SRNO + 15 HR 静态通道”与“EDSR 只看 LR depth”直接比较，会被认为输入信息不公平。

### 建议统一输出

除精度外，必须加：

- parameters；
- FLOPs 或 MACs；
- GPU/CPU inference latency；
- peak GPU memory；
- train time/epoch；
- 是否支持非整数尺度；
- 是否支持 unseen scale。

---

## P0-7. 当前报告不是完整投稿论文：缺少正式 Related Work / References

`report.md` 很完整，但体量、章节组织和表达更接近内部研究报告/结果审计，不是期刊论文。

### 必须补齐的相关工作主线

至少四条：

1. 城市洪水高分辨率模拟与 surrogate / neural operator；
2. 洪水超分辨率，包括 FLO-SR 等；
3. arbitrary-scale SR / implicit representation：LIIF、SRNO；
4. 空间泛化与 spatial blocked evaluation。

### 建议至少纳入的外部文献

1. Wei, M., Zhang, X. **Super-Resolution Neural Operator.** CVPR 2023. DOI: 10.1109/CVPR52729.2023.01750.  
   https://openaccess.thecvf.com/content/CVPR2023/html/Wei_Super-Resolution_Neural_Operator_CVPR_2023_paper.html

2. Chen, Y., Liu, S., Wang, X. **Learning Continuous Image Representation with Local Implicit Image Function.** CVPR 2021. DOI: 10.1109/CVPR46437.2021.00852.  
   https://openaccess.thecvf.com/content/CVPR2021/html/Chen_Learning_Continuous_Image_Representation_With_Local_Implicit_Image_Function_CVPR_2021_paper.html

3. Choi, H. et al. **FLO-SR: Deep learning-based urban flood super-resolution model.** Journal of Hydrology, 2025, 661A:133529. DOI: 10.1016/j.jhydrol.2025.133529.  
   https://www.sciencedirect.com/science/article/pii/S0022169425008674

4. LarNO, Journal of Hydrology 2026 / official code and benchmark release.  
   https://github.com/holmescao/LarNO

5. Valavi, R. et al. **blockCV: An R package for generating spatially or environmentally separated folds...** Methods in Ecology and Evolution, 2019, 10:225–232. DOI: 10.1111/2041-210X.13107.  
   https://doi.org/10.1111/2041-210X.13107

### 论文 novelty 建议重新定位

不要把创新写成泛化的：

> We propose an arbitrary-scale neural operator for urban flood super-resolution.

在当前证据下更稳妥：

> We formulate urban-flood super-resolution as a geography-guided mapping between independently simulated coarse- and fine-grid flood fields, and introduce a residual continuous-query model that explicitly combines low-resolution hydraulic states with high-resolution terrain and urban morphology.

真正有区分度的点是：**coarse field 本身来自独立粗网格物理模拟，而不是 2 m 真值下采样；HR geography 用于纠正跨分辨率物理模拟差异。** 这个角度比一般图像 SR 更有科学价值。

---

# Part II. 模型与损失函数代码审查

## 3. nodata→0 后直接 bilinear 会污染陆海/有效域边界

### 现状

`wellington_fixed_sr.py`：

```python
lr_valid = torch.isfinite(lr).float()
lr = torch.nan_to_num(lr, nan=0.0)
```

Hydro encoder 使用了 valid mask，这是对的；但模型的 bilinear base：

```python
lr_depth = lr[:, :1]
base_h = F.interpolate(lr_depth, ..., mode="bilinear")
```

没有对 valid mask 做归一化，因此无效像元填入的 0 会参与插值。

### 后果

在 nodata 邻域，尤其海岸、域边界或不规则有效域处，会系统性把 base depth 拉低。模型 residual 可能学习补偿这一数值人工效应，使所谓“geography correction”混入 mask artifact。

### 推荐 patch：mask-aware interpolation

```python
lr_depth = lr[:, :1]
valid = lr_valid[:, :1].float()

num = F.interpolate(
    lr_depth * valid,
    size=(hr_h, hr_w),
    mode="bilinear",
    align_corners=False,
)
den = F.interpolate(
    valid,
    size=(hr_h, hr_w),
    mode="bilinear",
    align_corners=False,
)

base_h = torch.where(
    den > 1e-6,
    num / den.clamp_min(1e-6),
    torch.zeros_like(num),
).clamp_min(0.0)
```

并保留：

```python
base_valid = den > 1e-6
```

必要时在 HR output mask 中与目标有效域求交。

### 重跑要求

需要重训练主模型，因为 residual 的学习目标基底改变了。

优先级：P1，但如果大量 tile 与 nodata 边界相交，影响可能达到 P0。

---

## 4. focal loss 的 alpha 实现不是标准正负类 alpha balancing

当前：

```python
pt = torch.where(target > 0.5, p, 1 - p)
loss = alpha * (1 - pt).pow(gamma) * bce
```

这里 alpha 对所有样本都是同一个 0.25，只是全局缩放 loss，并没有实现正类 `alpha` / 负类 `1-alpha`。

### 正确写法

```python
alpha_t = torch.where(
    target > 0.5,
    target.new_tensor(alpha),
    target.new_tensor(1.0 - alpha),
)
loss = alpha_t * (1.0 - pt).pow(gamma) * bce
```

如果原意只是 global weight，就应把参数名改为 `scale`，避免方法描述错误。

### 需要做的 sanity test

```python
def test_focal_alpha_is_class_specific():
    logits = torch.tensor([[[[0.0, 0.0]]]])
    target = torch.tensor([[[[1.0, 0.0]]]])
    mask = torch.ones_like(target, dtype=torch.bool)
    # 用内部逐元素版本检查 positive/negative 权重比应约为 alpha:(1-alpha)
```

---

## 5. boundary loss 的“zero nodata prevents leakage”注释与实际卷积不符

当前：

```python
mask4 = mask_b.unsqueeze(1).float()
e_pred = sobel_edges(pred_h * mask4)
e_true = sobel_edges(target * mask4)
l_boundary = masked_l1(e_pred, e_true, mask_b)
```

把 nodata 置零后做 Sobel，**恰恰会在 valid↔nodata 边界形成很强的人工梯度**。即使最终只在 valid pixel 上算 loss，距离 nodata 1 像元的 valid pixel Sobel stencil 仍包含这些人工 0。

### 推荐做法：先腐蚀 mask，只在 3×3 stencil 完全有效的区域算边界 loss

```python
mask4 = mask_b[:, None].float()
k = torch.ones((1, 1, 3, 3), device=mask4.device)
valid_count = F.conv2d(mask4, k, padding=1)
valid_stencil = valid_count.eq(9.0)

e_pred = sobel_edges(pred_h * mask4)
e_true = sobel_edges(target * mask4)

l_boundary = masked_l1(
    e_pred,
    e_true,
    valid_stencil[:, 0],
)
```

进一步建议对图像外边界也排除 1 像元，避免 zero padding 产生伪梯度。

---

## 6. extreme quantile 在 batch>1 时语义会变化

当前：

```python
valid_vals = target[:, 0][mask_b]
thr = torch.quantile(valid_vals, self.extreme_quantile)
```

它会把整个 batch 的 valid pixels 混在一起算阈值。V0 batch=1 时等价于 per-tile quantile；V1 如果 batch=2 或更大，就变成 batch-level quantile。

这样同一个样本的 extreme mask 会依赖同 batch 的另一个样本，产生不可控的 batch-composition effect。

### 建议改为逐样本

```python
terms = []
for i in range(target.shape[0]):
    mi = mask_b[i]
    vals = target[i, 0][mi]
    if vals.numel() == 0:
        continue
    thr_i = torch.quantile(vals, self.extreme_quantile)
    em = mi & (target[i, 0] >= thr_i)
    if em.any():
        terms.append(masked_l1(pred_h[i:i+1], target[i:i+1], em[None]))

l_extreme = torch.stack(terms).mean() if terms else pred_h.new_tensor(0.0)
```

同样检查 `deep_quantile` 是否应逐样本、逐 scenario 或全训练分布固定阈值。论文里需要把语义写清楚。

---

## 7. `scale_dim` 配置接口未完整接入 builder

`HydroGeoSRNO.__init__` 支持 `scale_dim`，但 `build_hydrogeo_srno()` 当前没有传：

```python
scale_dim=m.get("scale_dim", 16)
```

因此 YAML 若修改 `scale_dim` 不会生效。

### patch

```python
return HydroGeoSRNO(
    ...,
    scale_dim=m.get("scale_dim", 16),
    depth_ref=m.get("depth_ref", 0.10),
    ...,
)
```

### 建议新增配置解析测试

```python
def test_scale_dim_config_is_honored():
    cfg = {"model": {"scale_dim": 32}}
    m = build_hydrogeo_srno(cfg)
    assert m.scale_dim == 32
```

---

## 8. Residual LDM 当前只能称为 stub，不能称为已经实现的 diffusion model

`models/diffusion/residual_ldm.py`：

```python
def forward(self, x, t=None):
    return self.unet(x)
```

`t` 完全没被使用，网络是 3 层 Conv+GELU。扩散模型的 denoiser 至少需要感知 timestep/noise level，否则不能对不同扩散时刻的噪声条件建模。

### 如果 V2 要进入论文，需要补齐

1. sinusoidal timestep embedding；
2. timestep FiLM / additive conditioning；
3. residual UNet block；
4. beta schedule / alpha_bar；
5. `q_sample`；
6. epsilon-prediction 或 v-prediction；
7. reverse DDPM/DDIM sampler；
8. EMA weights；
9. deterministic model + geo conditioning；
10. uncertainty evaluation：CRPS、coverage、interval width、reliability。

### 最小 timestep embedding 示例

```python
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freq = torch.exp(
            -torch.log(torch.tensor(10000.0, device=t.device)) *
            torch.arange(half, device=t.device) / max(half - 1, 1)
        )
        args = t.float()[:, None] * freq[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
```

当前建议：论文主线暂时不放 V2 结果，只在 Future Work/experimental branch 中说明。

---

# Part III. 评价指标审查

## 9. empty-set CSI/F1 目前被自动设为 0，语义不严谨

当前：

```python
csi = tp / (tp + fp + fn).clamp_min(1.0)
```

当 truth 与 prediction 都没有任何超过阈值的像元时：

```text
TP=FP=FN=0
CSI=0
```

但数学上这是 undefined，不是“性能为零”。

### 建议

- tile-level：返回 NaN，并记录 `n_defined_tiles`；
- domain-pooled：只要全域 denominator > 0 就正常计算。

```python
den = tp + fp + fn
if den == 0:
    return float("nan"), float("nan")
```

这也能防止大量 dry tile 把 macro CSI 人工拉低。

---

## 10. FloodAreaRelativeError / VolumeRelativeError 的 dry-tile 语义需重新定义

当前：

```python
abs(pa-ta) / ta.clamp_min(1.0)
```

当 truth flooded area=0 时，输出本质上是 predicted flooded pixel count，而名称仍叫“relative error”。

体积同理：

```python
abs(pv-tv) / tv.clamp_min(1e-6)
```

极小 true volume 会导致异常大比例值。

### 推荐

主文 domain 指标：

```text
AreaBias = (A_pred - A_true) / A_true
VolumeBias = (V_pred - V_true) / V_true
```

按整个 split 聚合后计算。

Tile-level：

- `A_true=0` 时 RE=NaN；另报告 false-positive area；
- `V_true < epsilon_physical` 时 RE=NaN；另报告 absolute volume error。

---

## 11. 当前 PSNR 每个 tile 自己定义 data range，不适合跨 tile 比较

当前默认：

```python
data_range = max(target)-min(target)
```

于是同样的 MSE，在浅水 tile 和深水 tile 得到完全不同的 PSNR scale。随后再对 PSNR macro-average，解释性更差。

### 建议

洪水任务中 PSNR 本身不是核心物理指标。可：

- 降为 supplementary；
- 用固定 data range，例如训练集 99.9 percentile 或固定物理上限；
- 或干脆不作为主结论。

---

## 12. 当前所谓 SSIM 是 global-statistics approximation，不是标准局部窗口 SSIM

代码明确：

```python
"""Lightweight masked SSIM on valid region (global stats; secondary metric)."""
```

它用整张有效区的均值/方差/协方差，没有滑动窗口。因此不应在论文表里直接写成标准 `SSIM` 而不说明。

### 两种选择

A. 改名：

```text
Global-SSIM proxy
```

B. 换成标准实现：

- `torchmetrics.image.StructuralSimilarityIndexMeasure`
- 或 `skimage.metrics.structural_similarity`

同时正确处理 mask，最好只在完整有效窗口上计算。

---

# Part IV. 数据与物理问题审查

## 13. 这个任务不是普通图像 SR，论文必须强调“独立物理模拟之间的映射”

数据说明里最有价值的信息是：不同分辨率 flood fields 是**分别进行的数值模拟**，不是从 2 m 真值下采样出来的。

例如：

- 30 m simulation 与 2 m→30 m aggregate 的相关只有约 0.24–0.27；
- 10 m 与 2 m aggregate 约 0.65–0.68；
- coarse grid 会产生薄水膜等系统差异。

这意味着模型做的不是简单锐化，而是：

> learning a conditional correction from a coarse-grid hydrodynamic solution to a fine-grid hydrodynamic solution given high-resolution geography.

### 这应该成为论文的第一创新点

方法图建议明确：

```text
coarse simulator solution + HR static geography
                ↓
      HydroGeo-SRNO correction operator
                ↓
      fine-grid simulator-equivalent field
```

而不是画成一般 LR image→HR image。

---

## 14. 单城市 + 两事件意味着泛化结论必须严格限定

数据只有：

- Wellington；
- 20a；
- 100a。

因此当前可以证明：

- Wellington 内 held-out spatial bands；
- 在训练已见的两个 return-period scenarios 下空间泛化。

不能证明：

- unseen city；
- unseen catchment；
- unseen storm；
- unseen return period；
- general urban flooding。

### 最小增强方案

如果暂时没有第二城市：

1. 做 spatial blocked CV；
2. 做 terrain/urban morphology covariate shift 分析；
3. 将结论限定在 within-city spatial generalization；
4. 若可补算 50a flood，则 50a 可作为真正 event-intensity holdout，但不能用已有 rainfall 直接替代 flood truth。

---

## 15. 50a rainfall 存在，但 50a flood truth 不存在，绝对不能混用

`DATASET.md` 明确：

```text
rain/50a.npy exists
flood_50a.nc does not exist
```

论文数据章节应专门说明，避免读者误以为三个 return periods 都有 flood target。

如果未来要做 50a 验证，必须重新运行物理模型或取得真实 50a flood simulation，不能从 20a/100a 插值后当 ground truth。

---

## 16. CRS 是“推断”为 EPSG:2193，而不是源文件显式携带

这是数据 provenance 问题。

建议在 dataset build 阶段：

- 把 `crs_wkt` / `epsg=2193` 写进 derived NetCDF global attrs；
- 保存 affine transform；
- 自动检查四角坐标/extent；
- 输出 `grid_provenance.json`。

示例：

```python
nc.setncattr("crs_epsg", 2193)
nc.setncattr("crs_name", "NZTM2000")
nc.setncattr("source_crs_status", "inferred_and_verified")
```

论文中不要写成“source data are EPSG:2193”这种过强表述，建议写“coordinates were interpreted/verified as NZTM2000 (EPSG:2193)”。

---

## 17. land-use code 5 的 provenance 不完整，需要敏感性分析

数据文档写明 code 5 在 published source data 中缺失/不明，但当前赋：

```text
Manning n = 0.035
infiltration = 4.58e-7 m/s
```

如果模型输入直接包含 Manning / infiltration / land use，这个类别定义会进入预测。

### 建议

在 Supplementary 加：

- code 5 像元比例；
- train/val/test 分布；
- code 5 是否集中于特定区域；
- 将 code 5 分别替换为相邻 plausible 类别参数的敏感性实验；
- 或把它标记 `unknown` 单独 embedding，不把推定参数写成事实。

---

# Part V. 深水问题的解释需要收缩

## 18. “深水欠估源于训练目标，网络容量充裕”目前证据不足

现有深水 fine-tuning 的确显示 CSI@1m 等阈值指标可能改善，但固定 panel 中更深 bin 的 MAE/bias 并没有一致改善，甚至 >3m bias 有恶化迹象。

因此现阶段最多可说：

> The results are consistent with objective-induced underweighting of rare deep-water cells.

不应直接写：

> deep-water underestimation originates from the training objective and network capacity is sufficient.

因为还没有排除：

- coarse input 在深水处信息缺失；
- geography feature 不足；
- output parameterization saturation；
- resolution-dependent simulator discrepancy；
- capacity / optimization interaction；
- extreme cells 的物理异常或局部数值差异。

### 建议新增 4 个诊断

1. `error vs LR truth discrepancy`：看 10m→2m 物理差异越大时模型是否越难；
2. deep bins 中 conditional variance；
3. residual head activation / gradient magnitude vs depth；
4. capacity sweep（width 128/192/256 或 blocks）至少小规模验证一次。

只有 capacity sweep 没显著改善、而 loss reweight 能稳定跨 seed 改善时，才能较强地支持“objective—not capacity”。

---

## 19. 深水优化建议改成 Pareto/多目标，而不是单独追 CSI@1m

目前深水项可能改善 threshold hit rate，但加剧极深水数值 bias。

建议定义综合 selection score：

```text
S = z(CSI_005)
  + 0.5 z(CSI_100)
  - 0.5 z(RMSE_wet)
  - 0.5 z(abs(VolumeBias))
  - 0.5 z(abs(Bias_3m_plus))
```

更规范的做法不是人为混合，而是报告 Pareto front：

- general inundation skill；
- deep-water detection；
- deep-water continuous error；
- volume bias。

选择 checkpoint 时先在 validation 上确定 Pareto-optimal candidate，再用预先定义的 tie-break rule。

---

# Part VI. 训练可复现性与工程问题

## 20. seed 设置不完整

`train_fixed.py` 当前只设置：

```python
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
```

没有设置 NumPy，也没有 DataLoader worker seed。

### 建议

```python
import os
import random
import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = False):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


g = torch.Generator()
g.manual_seed(seed)

loader = DataLoader(
    ds,
    ...,
    worker_init_fn=seed_worker,
    generator=g,
)
```

并把：

- seed；
- git commit；
- Python/PyTorch/CUDA；
- GPU name；
- config resolved YAML；

写进每次 run 的 `provenance.json`。

---

## 21. requirements.txt 不足以重现实验

目前只有：

```text
torch>=2.0
numpy
netCDF4
pyyaml
```

但仓库明显还包含：

- scipy 统计；
- scikit-learn 聚类/指标；
- matplotlib 绘图；
- pandas 表格；
- R 脚本/SABRE cross-check；
- 可能的 raster/tif 处理依赖。

### 建议拆成

```text
requirements-core.txt
requirements-analysis.txt
requirements-paper.txt
environment.yml
```

至少把精确版本冻结：

```bash
python -m pip freeze > requirements-lock.txt
```

更推荐 `pyproject.toml`：

```toml
[project]
dependencies = [
  "torch>=2.0",
  "numpy",
  "netCDF4",
  "pyyaml",
]

[project.optional-dependencies]
analysis = [
  "scipy",
  "pandas",
  "scikit-learn",
  "matplotlib",
  "torchmetrics",
]
test = ["pytest", "pytest-cov"]
```

---

## 22. 原始数据未公开不等于不可投稿，但 Data Availability 必须改得更精确

仓库已经诚实说明 41 GB 级原始输入/大型 checkpoint 未全部放入镜像，这是好的。

但论文里不要笼统写“all data are publicly available”除非确实有稳定 DOI/公开地址。

建议数据可用性声明拆成：

```text
1. source/raw simulator outputs: [location / access condition / version]
2. derived indices and compact machine results: GitHub repository
3. preprocessing code: GitHub repository
4. trained weights: release/Zenodo/HuggingFace DOI
5. checksums: _manifest.json
```

最好发布一个不可变 release：

```text
v1.0-paper
Zenodo DOI
```

---

## 23. 建议取消“扁平镜像作为主工程仓库”的角色

扁平镜像便于自动审计，但不适合作为人类开发/复现主仓库，因为：

- 原始路径被编码进文件名；
- import 关系不直观；
- 用户难以 `git clone && run`；
- diff/PR 可读性差。

建议：

```text
main branch       = 正常目录结构，可运行
paper-release     = 论文冻结版本
flat-audit-mirror = 可选自动生成 artifact，不作为源码主入口
```

扁平镜像通过 CI 自动生成，不手工维护。

---

# Part VII. 测试体系需要从“shape sanity”升级到“科学语义测试”

当前 tests 数量不多，建议重点新增下面这些单元测试。

## 24. 必加测试清单

### 24.1 mask-aware interpolation

```python
def test_masked_bilinear_does_not_dilute_valid_depth():
    depth = torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]])
    valid = torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]])
    # masked interpolation 后，有效邻域不应因 nodata 的填零被错误稀释
```

### 24.2 A4 必须存在 gradient path

```python
def test_direct_depth_ablation_has_trainable_gradient():
    cfg = make_cfg(predict_residual=False, direct_depth=True)
    model = build_hydrogeo_srno(cfg)
    out = model(**fake_batch())
    loss = out["depth"].mean()
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in model.parameters())
```

### 24.3 wet head gradient

```python
def test_wet_head_gets_gradient_when_enabled():
    ...
    losses = FloodLoss(w_wet=0.3)(out, y, mask)
    losses["total"].backward()
    assert model.wet_head.out.weight.grad.abs().sum() > 0
```

### 24.4 unseen pair 不得出现在训练集

```python
def test_unseen_pairs_are_disjoint():
    train = {tuple(x) for x in cfg["dataset"]["train_pairs"]}
    unseen = {tuple(x) for x in cfg["dataset"]["eval_unseen_pairs"]}
    assert train.isdisjoint(unseen)
```

### 24.5 metric accumulator 与一次性拼接等价

```python
def test_global_rmse_accumulator_matches_concat():
    # 两个 batch 分开 update
    # 与 concatenate 后直接 RMSE 比较，必须相等
```

### 24.6 empty-event semantics

```python
def test_empty_csi_is_nan_not_zero():
    ...
```

### 24.7 boundary loss 不读取 nodata stencil

构造一个 valid 区常数 1、周围 nodata；若 valid 内无真实边界，loss 应接近 0。

### 24.8 multiscale trainer 读取 YAML

检查：

- epochs；
- batch_size；
- amp；
- optimizer lr；
- output dir；

实际对象参数与 YAML 一致。

---

# Part VIII. 图表审稿

## 25. 当前图很多，但主文信息密度过高，应从“全结果展示”改成“证据链展示”

当前报告图超过 50 张。论文主文不应全部保留。

### 建议主文仅保留 7–8 张

**Fig. 1** 任务定义与数据：独立 10m/2m simulation + HR geography，不是 downsample SR。  
**Fig. 2** HydroGeo-SRNO architecture。  
**Fig. 3** spatial split + buffer + study area。  
**Fig. 4** strong baselines 主结果。  
**Fig. 5** representative spatial maps：truth / bilinear / strongest learned baseline / proposed / error。  
**Fig. 6** depth-bin performance + deep bias。  
**Fig. 7** ablation（修复后的正交消融）。  
**Fig. 8** multiscale seen/unseen scale（只有真正重跑 V1 后才放）。

其余：training history、loss part、selection noise、bootstrap、variogram、K sensitivity 等放 Supplementary。

---

## 26. 所有主要结果图都应增加“样本单位”和“不确定性单位”

例如 forest plot 不只写：

```text
95% bootstrap CI
```

而写：

```text
95% spatial-block bootstrap CI; blocks defined from residual variogram range
```

seed-level 比较写：

```text
mean ± 95% CI across 5 paired random seeds
```

不要让读者误以为 20 epochs 是 n=20 independent runs。

---

## 27. 深度 bin 图建议同时画 prevalence / volume contribution

报告已经发现 >深水像元比例很小但体积贡献很大。建议 Fig.6 做成三个 panel：

```text
(a) pixel prevalence by depth bin
(b) share of total water volume by depth bin
(c) model MAE / bias / CSI by depth bin
```

这样可以自然解释为什么“整体 RMSE 看起来改善有限，但深水安全相关性能仍值得关注”。

---

# Part IX. 论文逻辑重构建议

## 28. 推荐论文主问题只保留三个

当前报告内容很多，容易形成“做了很多检查，但论文中心不突出”。建议收敛到：

### RQ1

在 coarse/fine flood fields 来自独立物理模拟、并非降采样关系时，HR geography 是否能帮助从 coarse simulation 重建 fine-grid inundation？

### RQ2

连续查询/residual neural operator 相比 interpolation 和强学习型 baseline，是否在空间 held-out Wellington 区域显著提高洪水深度与淹没边界精度？

### RQ3

模型在 rare deep-water cells 上的偏差来自什么，经过目标函数重加权后能否在不破坏总体性能的情况下改善？

多尺度 arbitrary-scale 可以作为 RQ4，但前提是修复 V1 并有真实 unseen-scale 结果。否则建议放成未来工作。

---

## 29. 推荐论文结构

### 1 Introduction

- coarse simulation 的效率需求；
- 传统 SR 的 downsample assumption 与本任务不一致；
- HR terrain/morphology 的信息价值；
- 三个 contribution。

### 2 Related Work

- flood emulation / surrogate；
- flood SR；
- implicit/arbitrary SR；
- spatial validation。

### 3 Data and Problem Formulation

必须明确：

```math
h_H = \mathcal{F}_\theta(h_L, G_H, r_L, r_H)
```

其中 `h_L` 和 `h_H` 来自两个分辨率独立模拟，而不是 `h_H` 的 downsample/upscale 对。

### 4 Method

- hydro encoder；
- HR geo encoder；
- continuous query/operator；
- residual log-depth parameterization；
- loss。

### 5 Experimental Design

- spatial split/buffer；
- baselines；
- seed repeats；
- domain vs tile metrics；
- spatial block inference；
- ablation。

### 6 Results

6.1 main benchmark  
6.2 spatial examples  
6.3 ablation  
6.4 deep-water behavior  
6.5 arbitrary-scale（若完成）

### 7 Discussion

- why geography helps；
- coarse/fine simulator discrepancy；
- deep-water tradeoff；
- within-city limitation。

### 8 Conclusions

只陈述被数据直接支持的结论。

---

## 30. 建议改写 contribution

可以写成类似：

1. We formulate flood super-resolution between **independently simulated** coarse and fine hydraulic fields rather than artificially downsampled image pairs, making the task a correction of resolution-dependent hydrodynamic discrepancies.
2. We develop a geography-guided residual continuous-query model that fuses the coarse hydraulic state with fine terrain and urban morphology to recover fine-grid flood depth.
3. We establish spatially held-out evaluation with flood-centric metrics and explicitly diagnose rare deep-water errors, separating inundation detection skill from continuous extreme-depth bias.

如果 V1 修复后成立，再加：

4. We evaluate scale generalization on resolution pairs excluded from training, including fractional-scale queries.

---

# Part X. 基线和实验重跑矩阵

## 31. 建议的最终实验矩阵

### E0：现有结果重算，不训练

- global/domain metrics；
- tile-macro metrics；
- empty-case metric fix；
- block-bootstrap CI；
- per-scenario metrics；
- depth-bin metrics；
- volume bias。

### E1：主模型多 seed

```text
seed 11,22,33,44,55
10m→2m
V0 fixed-scale
```

### E2：强基线

```text
nearest
bilinear
EDSR
ResUNet
SRNO-single
HydroGeo-SRNO
```

至少 3 seeds/learned baseline；主 proposed 5 seeds。

### E3：正交消融

```text
G0..G4
R0/R1
W0/W1
L0..L5
```

主消融至少 3 seeds。

### E4：deep loss

```text
w_deep = 0, 0.03, 0.05, 0.10, 0.20
```

selection 只看 validation，test 最后一次。

### E5：arbitrary scale

train 不含 5→2、20→2、30→2；测试 seen/unseen separately。

### E6：空间 blocked robustness

至少 4 folds 或 3 种 block size sensitivity。

---

# Part XI. 代码改造建议：建议直接新增/修改的文件

## 32. 推荐目录

```text
hydrogeo_sr/
  data/
  models/
  losses/
  metrics/
  training/
  evaluation/
  statistics/
  plotting/

configs/
  fixed/
  multiscale/
  ablations/

scripts/
  train.py
  evaluate.py
  run_benchmarks.py
  run_ablation.py
  run_seed_repeats.py
  run_multiscale.py
  make_paper_tables.py
  make_paper_figures.py

tests/

artifacts/
  paper_v1/
    tables/
    figures/
    metrics/
    provenance/
```

避免继续积累大量 `_verify_*`, `_diag_*`, `_shot_*` 脚本而没有统一入口。

---

## 33. 统一 run manifest

每次实验写：

```json
{
  "run_id": "v0_seed11_20261006",
  "git_commit": "...",
  "seed": 11,
  "config_sha256": "...",
  "dataset_manifest_sha256": "...",
  "python": "3.x",
  "torch": "x.y",
  "cuda": "x.y",
  "gpu": "...",
  "split": "spatial-band-v1",
  "train_pairs": [[10,2]],
  "checkpoint_rule": "best validation CSI_005, tie by RMSE_wet"
}
```

这样论文任何一个数字都能追到 run。

---

## 34. 把 paper number 变成程序生成，不再手工复制

建议：

```text
artifacts/paper_v1/metrics/all_runs.parquet
artifacts/paper_v1/tables/table_main.csv
artifacts/paper_v1/tables/table_ablation.csv
artifacts/paper_v1/stats/deep_seed_stats.json
```

报告/论文 Markdown 只从这些冻结 artifact 取数字。

加 CI 检查：

```python
assert abs(number_in_paper - number_in_json) < 5e-4
```

仓库已有很多 verify 脚本，可以整合成一个 `paper_audit.py`。

---

# Part XII. 需要删除、降调或改写的论文结论

## 35. 不建议保留的强结论

### 35.1 “网络容量充裕/深水偏差由 loss 导致”

改为：

> The observed improvement after deep-water reweighting suggests that objective imbalance is one contributor to the deep-water bias; capacity and input-information limitations cannot yet be excluded.

### 35.2 “statistically significant across 20 epochs”

删除。epoch analysis 只保留为 training-dynamics diagnostic。

### 35.3 “geographically independent test demonstrates generalization”

改为 within-Wellington held-out spatial generalization。

### 35.4 “arbitrary-scale capability validated”

在当前 V1 脚本修复、unseen pair 严格隔离之前，不要作为已验证主贡献。

### 35.5 “SSIM”

如果仍用 global approximation，改名，不要与标准 SSIM 混淆。

---

# Part XIII. 可直接执行的修改优先级

## 36. 第一批：不重训练即可完成

1. 重写 metrics accumulator，输出 global + macro 两套指标；
2. 修 empty CSI/F1、area/volume denominator；
3. 用 spatial block bootstrap 重算 CI；
4. 删除 epoch-based inferential p-value；
5. 重新生成 main metric table；
6. 把主文泛化措辞限定到 Wellington spatial holdout；
7. 增加 Related Work / References；
8. 主文图缩到 7–8 张；
9. 修数据可用性说明；
10. 将近似 SSIM 改名或替换。

这些应该先完成，因为会决定“旧结果哪些还能继续使用”。

---

## 37. 第二批：需要小规模/中等规模重训练

1. mask-aware bilinear base；
2. focal alpha；
3. valid-stencil boundary loss；
4. per-sample extreme quantile；
5. direct-depth head；
6. wet-head valid ablation；
7. 3–5 seeds；
8. EDSR/ResUNet/SRNO strong baselines；
9. 正交消融；
10. deep-loss sweep。

---

## 38. 第三批：决定是否把 arbitrary-scale 作为主贡献

1. 重写 multiscale trainer；
2. train/eval pair 隔离；
3. seen/unseen scale table；
4. 5→2 fractional zero-shot；
5. 20→2 / 30→2 large-factor extrapolation；
6. per-scale calibration/robustness；
7. 与 LIIF/SRNO 连续 SR baseline 比较。

如果这部分结果不强，建议论文第一版先聚焦 fixed 10m→2m，而不是为了“范围大”把一个未成熟分支强行塞进主线。

---

# Part XIV. 最终主表建议模板

## 39. Table 1：main benchmark

```text
Model | HR geo | Params | RMSE_domain↓ | MAE_wet_domain↓ | CSI005_domain↑ |
CSI100_domain↑ | VolumeBias↓ | Tile RMSE median [IQR] | latency
```

所有 learned model 给 mean ± SD across seeds。

---

## 40. Table 2：ablation

```text
Variant | geography | residual | wet head | flood loss | RMSE | CSI005 | CSI100 | deep bias
```

注意每行只改变一个因素。

---

## 41. Table 3：spatial/statistical robustness

```text
Method | iid tile CI (diagnostic only) | block 2x2 CI | block 3x3 CI | block 4x4 CI
```

这样审稿人会看到你主动处理空间相关，而不是忽略。

---

## 42. Table 4：scale generalization（仅 V1 修复后）

```text
Pair | scale | seen in training? | Bilinear | SRNO | HydroGeo-SRNO
20→10 | 2x | yes | ...
10→5  | 2x | yes | ...
5→2   | 2.5x | NO | ...
20→2  | 10x | NO | ...
30→2  | 15x | NO | ...
```

---

# Part XV. 建议的审稿人式问题清单

如果我是外审，我会要求作者回答以下问题；建议你在修稿前主动准备答案。

1. 为什么把连续训练 epoch 当独立统计重复？
2. 空间自相关达到明显水平时，为什么使用 iid tile bootstrap？
3. 报告的 RMSE/CSI 是 pooled-domain 还是 per-tile macro？
4. 为什么强基线只有 nearest/bilinear，而模型仓库已有 EDSR/RCAN/ResUNet？
5. HydroGeo-SRNO 比 baseline 多用了 HR static geography，输入是否公平？
6. A4 关闭 residual 后网络还有什么可训练的 depth path？
7. A6 的 wet head 在 masked-L1 下如何获得梯度？
8. 5→2 若参与训练，为什么还能称为 arbitrary-scale test？
9. `train_multiscale.py` 为什么没有使用 YAML 的 epochs/batch/AMP？
10. 论文所谓 SSIM 是否标准 local-window SSIM？
11. nodata 置零后 bilinear 是否污染边界？
12. Sobel loss 是否把 nodata edge 当作真实 flood edge？
13. 只有一个城市和两个 return periods，外部泛化结论的依据是什么？
14. 10m/2m 本来就是独立模拟，模型到底是在做 SR 还是跨网格物理校正？
15. 极深水误差恶化时，为什么仍得出 deep-water bias 已被解决？
16. land-use code 5 的参数来源是什么？
17. 原始数据和 checkpoint 如何被第三方完整获取？
18. 为什么 requirements 只有四个包，而分析/绘图显然使用更多依赖？
19. 主结果是否对 random seed 稳健？
20. proposed gain 是否在 spatial blocked folds 上保持？

修订稿应做到：审稿人在正文或 Supplementary 里能直接找到这 20 个问题的答案。

---

# Part XVI. 建议的最终审稿意见（可直接作为内部评审摘要）

## Major comments

### Major 1 — statistical unit of replication

The manuscript currently treats consecutive fine-tuning epochs as paired experimental replicates and spatial tiles as iid bootstrap units. Neither assumption is justified. Training epochs from a single optimization trajectory are strongly dependent, while test tiles are spatially structured and nested within only two scenarios in one city. The inferential analysis should be rebuilt around independently seeded runs and spatial block/cluster resampling.

### Major 2 — metric aggregation

The reported split-level metrics are produced by averaging batch/tile-level metrics. With batch size one, this corresponds to macro-averaging tiles and is not equivalent to the conventional pooled-domain RMSE, CSI, F1 or volume error. Both estimands should be explicitly defined and reported, with domain-pooled metrics used for the main physical performance claims.

### Major 3 — ablation validity

Two critical ablations do not isolate the intended factors. Disabling residual prediction currently zeros the residual and reduces the output to the fixed bilinear base rather than creating a direct-depth model. Enabling the wet head while retaining masked L1 does not train the wet head. The ablation study should therefore be redesigned as orthogonal, one-factor-at-a-time comparisons.

### Major 4 — baseline sufficiency

The formal benchmark currently contains interpolation baselines only. This is insufficient for a method paper centered on a learned SR/operator model. At least one strong CNN SR baseline, one U-Net-type spatial model, and one continuous/arbitrary-scale SR baseline should be trained and evaluated under matched inputs and splits.

### Major 5 — arbitrary-scale evidence

The current multiscale training script does not implement the training protocol specified in its YAML configuration, and the nominal fractional-scale test pair is included among the training pairs. Claims of arbitrary-scale or zero-shot scale generalization therefore require new experiments with strict train/test scale separation.

### Major 6 — scientific positioning

The most scientifically important aspect of the dataset is that coarse- and fine-resolution flood fields come from independent hydrodynamic simulations rather than synthetic downsampling. The paper should be reframed around geography-guided correction of resolution-dependent hydraulic discrepancies, which is stronger and more defensible than treating the task as generic image super-resolution.

---

# Part XVII. 最终推荐

### 现在的状态

**具备论文基础，但尚不适合直接投稿。**

### 修完 P0 后

如果能够做到：

- global + macro 指标重算；
- seed-level inference；
- spatial block uncertainty；
- A4/A6 消融修复；
- 至少 3 个强学习基线；
- 文献与论文结构补齐；
- 收缩因果/泛化表述；

则这项工作会从“较完整的研究报告”变成一篇证据链较扎实的 urban-flood super-resolution / physics-guided SR 方法论文。

### 若还能完成 V1

若进一步把：

- multiscale trainer 修好；
- 5→2 等 unseen scale 完全隔离；
- arbitrary-scale 对 LIIF/SRNO 做公平比较；

则“continuous-query / arbitrary-scale”可以成为真正的第二条方法创新，而不是架构层面的潜在能力。

---

# Appendix A. 建议直接修改的文件清单

| 文件 | 优先级 | 修改 |
|---|---:|---|
| `metrics/aggregation.py` | P0 | 增加 domain pooled accumulator；保留 macro 并改名 |
| `metrics/flood_metrics.py` | P0/P1 | empty CSI、area/volume、固定 PSNR range、标准 SSIM |
| `configs/v0_10m2m_hmax_ablation.yaml` | P0 | 重写正交消融 |
| `models/hydrogeo_srno.py` | P0/P1 | direct head、mask-aware base、接入 scale_dim |
| `losses/flood_loss.py` | P1 | focal alpha、valid-stencil boundary、per-sample quantile |
| `scripts/train_multiscale.py` | P0 | 完整 epoch/val/checkpoint/config pipeline |
| `configs/v1_arbitrary_scale.yaml` | P0 | train/seen/unseen pairs 分离 |
| `scripts/train_fixed.py` | P1 | NumPy/DataLoader seed、run provenance |
| `models/diffusion/residual_ldm.py` | P2 | timestep-aware denoiser；否则保留 stub 标签 |
| `requirements.txt` | P1 | 完整依赖/锁版本 |
| `report.md` | P0 | 收敛成 manuscript，补 references，改统计与结论 |
| `tests/*` | P1 | 增加科学语义测试 |

---

# Appendix B. 推荐提交前验收门槛

在论文进入投稿前，建议逐项通过：

- [ ] 主表至少包含 3 个学习型基线；
- [ ] proposed 模型 ≥5 seeds，强基线 ≥3 seeds；
- [ ] test 只在所有选择规则冻结后读一次；
- [ ] 无 epoch-as-replicate 显著性检验；
- [ ] spatial-block CI 已完成；
- [ ] domain pooled 与 tile macro 区分；
- [ ] A4 direct-depth 真正可训练；
- [ ] wet head 有非零 gradient 测试；
- [ ] `[5,2]` 不出现在 arbitrary-scale 训练 pairs；
- [ ] multiscale trainer 真正执行 YAML epochs/AMP/optimizer；
- [ ] nodata-aware interpolation；
- [ ] boundary loss 不受 nodata 伪边界污染；
- [ ] focal alpha 语义修复或改名；
- [ ] 标准 SSIM 或明确 proxy；
- [ ] 论文只声称 within-Wellington spatial generalization；
- [ ] Related Work + References 完整；
- [ ] 原始数据/权重/派生结果的 Availability 声明准确；
- [ ] 每个论文数字可追溯到一个冻结 JSON/CSV + run manifest；
- [ ] release tag + commit hash + environment lock 已冻结。

---

# Appendix C. 本审查使用的主要公开来源

仓库：  
https://github.com/Coucou2016/20260921-JiaChenchen-Paper

仓库关键文件：

- `FILE_INDEX.md`  
  https://github.com/Coucou2016/20260921-JiaChenchen-Paper/blob/master/FILE_INDEX.md
- `report.md`  
  https://github.com/Coucou2016/20260921-JiaChenchen-Paper/blob/master/report.md
- `dataset__DATASET.md`  
  https://github.com/Coucou2016/20260921-JiaChenchen-Paper/blob/master/dataset__DATASET.md
- `models__hydrogeo_srno.py`  
  https://github.com/Coucou2016/20260921-JiaChenchen-Paper/blob/master/models__hydrogeo_srno.py
- `losses__flood_loss.py`  
  https://github.com/Coucou2016/20260921-JiaChenchen-Paper/blob/master/losses__flood_loss.py
- `metrics__flood_metrics.py`  
  https://github.com/Coucou2016/20260921-JiaChenchen-Paper/blob/master/metrics__flood_metrics.py
- `metrics__aggregation.py`  
  https://github.com/Coucou2016/20260921-JiaChenchen-Paper/blob/master/metrics__aggregation.py
- `scripts__train_fixed.py`  
  https://github.com/Coucou2016/20260921-JiaChenchen-Paper/blob/master/scripts__train_fixed.py
- `scripts__train_multiscale.py`  
  https://github.com/Coucou2016/20260921-JiaChenchen-Paper/blob/master/scripts__train_multiscale.py
- `configs__v0_10m2m_hmax.yaml`  
  https://github.com/Coucou2016/20260921-JiaChenchen-Paper/blob/master/configs__v0_10m2m_hmax.yaml
- `configs__v0_10m2m_hmax_ablation.yaml`  
  https://github.com/Coucou2016/20260921-JiaChenchen-Paper/blob/master/configs__v0_10m2m_hmax_ablation.yaml
- `configs__v1_arbitrary_scale.yaml`  
  https://github.com/Coucou2016/20260921-JiaChenchen-Paper/blob/master/configs__v1_arbitrary_scale.yaml
- benchmark table  
  https://github.com/Coucou2016/20260921-JiaChenchen-Paper/blob/master/resultdata__benchmarks__table_test.json

相关方法/统计参考：

- Wei, M., Zhang, X. Super-Resolution Neural Operator. CVPR 2023. DOI: 10.1109/CVPR52729.2023.01750.  
  https://openaccess.thecvf.com/content/CVPR2023/html/Wei_Super-Resolution_Neural_Operator_CVPR_2023_paper.html
- Chen, Y., Liu, S., Wang, X. Learning Continuous Image Representation With Local Implicit Image Function. CVPR 2021. DOI: 10.1109/CVPR46437.2021.00852.  
  https://openaccess.thecvf.com/content/CVPR2021/html/Chen_Learning_Continuous_Image_Representation_With_Local_Implicit_Image_Function_CVPR_2021_paper.html
- Choi, H. et al. FLO-SR: Deep learning-based urban flood super-resolution model. Journal of Hydrology, 2025. DOI: 10.1016/j.jhydrol.2025.133529.  
  https://www.sciencedirect.com/science/article/pii/S0022169425008674
- LarNO official repository, Journal of Hydrology 2026.  
  https://github.com/holmescao/LarNO
- Valavi, R. et al. blockCV. Methods in Ecology and Evolution, 2019. DOI: 10.1111/2041-210X.13107.  
  https://doi.org/10.1111/2041-210X.13107

---

## 一句话结论

**最值得保留并强化的科学核心，不是泛泛的“把洪水图超分辨率”，而是“在粗、细网格来自独立水动力模拟的条件下，用高分辨率地形/城市形态约束连续残差算子，学习分辨率引起的水动力差异”；最需要立即修的不是网络结构本身，而是统计重复单位、指标聚合、消融有效性和 arbitrary-scale 训练证据。**
