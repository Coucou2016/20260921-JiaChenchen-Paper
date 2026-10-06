# 城市洪水深度超分辨率重建中的深水欠估问题

HydroGeo-SRNO — 地理引导的任意尺度神经算子，用于城市洪水水深超分辨率重建。
以新西兰惠灵顿城区洪水数据集为例，研究对象是十米到二米、最大淹没水深 `h_max`
的重建。

本仓库是一份**扁平化的公开镜像**，把代码、文档、报告、图件与核心结果数据
全部放在同一个目录下，便于自动化阅读与交叉审查。没有子目录是刻意的安排，
原因与命名约定见 `START_HERE_FLAT_LAYOUT.md`，完整清单见 `FILE_INDEX.md`。

---

## 一句话结论

把粗网格水深重建到细网格时，模型系统性地低估深水。本报告定位了成因，
给出了修正，并做了配对显著性检验。

## 报告的三份主产物

| 文件 | 说明 |
|---|---|
| `report.html` / `report.md` / `report.pdf` | 完整报告，图 52、表 32 |
| `report_brief.html` / `report_brief.md` / `report_brief.pdf` | 精简版，只保留核心图表，图 21、表 10，正文约六千六百字 |
| `reportbackup__report.*` | 早前一轮的报告快照，仅作对照 |

## 项目结构（原始目录，镜像中以前缀编码）

```
configs/     V0、消融、多尺度、扩散模型的 YAML
dataset/     WellingtonSRDataset 与各适配器
models/      HydroGeo-SRNO、Galerkin 算子、各类基线
losses/      FloodLoss 与深水加权变体
metrics/     CSI / F1 / RMSE_wet 等
engine/      训练器、评估器、检查点
scripts/     训练、评估、推理、作图、报告构建、审计脚本
tests/       单元检查
```

## 环境

```bash
pip install torch pyyaml netCDF4 numpy
```

命令都在项目根目录执行。主要实验 V0 是十米到二米、目标 `h_max`，
在二米静态特征栈条件下的确定性残差超分辨率。

## 复现顺序

```bash
# 1) 双线性基线
python scripts/evaluate.py --config configs/v0_10m2m_hmax.yaml --model bilinear --split val
# 2) 冒烟训练
python scripts/train_fixed.py --config configs/v0_10m2m_hmax_smoke.yaml
# 3) 完整 V0 训练
python scripts/train_fixed.py --config configs/v0_10m2m_hmax.yaml
# 4) 前置分析（跨分辨率、空间模式相似性）
python scripts/premodel_10_consistency.py
python scripts/premodel_11_pattern_extra.py
# 5) 重建三份产物
python scripts/build_report.py && python scripts/build_markdown.py && python scripts/build_pdf.py
python scripts/build_brief_report.py && python scripts/build_brief_markdown.py && python scripts/build_brief_pdf.py
```

## 数据与权重的说明

原始数值模拟场与完整检查点合计约 41 GB，超出 GitHub 承载能力，未随本镜像上传。
**这不影响任何报告数值的可追溯性**，因为每张图、每个表格背后的结果文件都已在
本目录中。缺什么、多大、如何再生成，见 `DATA_AND_BINARY_NOTICE.md`。

## 给自动化审查者的入口

- `chatgpt__00_TASK_BRIEF.md` — 项目背景与审查任务
- `chatgpt__01_WHERE_TO_LOOK.md` — 最值得核对的结论与对应文件
- `_manifest.json` — 逐文件的大小与 SHA-256

*快照日期 2026-10-06。*
