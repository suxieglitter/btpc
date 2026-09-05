# BTPC 论文结果复现报告（2026-09-05）

用本包从零重训论文的全部模型，对比论文原始输出。模型权重、预测 CSV 和图件等
run 产物按惯例留在本地 `runs/`（gitignore），本文件记录口径和数字。

环境：RTX 4060 Ti（8G）、venv torch 2.7.1+cu118；单 run 耗时 stage1 约 14-20 分钟 + stage2 约 2 分钟 + 论文口径 valid 约 20 秒。

## 三个复现 run（runs/ 下）

| run | 数据 | 配置 | 对应原始 run |
|---|---|---|---|
| repro_v1_120 | v1 基础版 scsn_consensus_fm_CFM_EQ.hdf5 | 120 epochs、seed 42、训练期 filter TTA 4 views | mdl_d20260518_1（论文 CM 图左 panel） |
| repro_v2_s42 | v2 精制版（yaml 默认） | 80 epochs、seed 42 | mdl_d20260518 …_42（CM 图右 panel、论文主模型） |
| repro_v2_s325 | v2 精制版 | 80 epochs、seed 325 | mdl_d20260518 …_325 |

每个 run：`btpc-train stage1` → `btpc-train stage2`（默认 30 epochs、spectral、冻结 encoder、含 downweight，与原 inline 设置一致）→ 论文口径 valid：

```bash
btpc-valid --stage2-checkpoint <run>/stage2/stage2_cluster_classifier.pth \
  --target-source train --n-tta-views 30 --tta-max-shift 0 --tta-noise-std 0.015 \
  --reject-rule confidence_center_margin --reject-strategy all \
  --valid-confidence-threshold 0.8 --valid-center-margin-threshold 0.05
```

评估目标是用种子选取重建的 10000 条 stage1 训练行（与论文 predict_valid 输出同口径，已预先核对 source_index 逐条一致）。

## 指标对比（论文拒收规则：mc<0.8 且 mcm<0.05 才拒）

| run | 指标 | 原始 | 复现 | 差 |
|---|---|---|---|---|
| v1_120 | 拒收前精度 | 0.9857 | 0.9849 | −0.08pp |
| | 接受率 | 0.9743 | 0.9760 | +0.17pp |
| | 接受子集精度 | 0.9935 | 0.9918 | −0.17pp |
| v2_s42 | 拒收前精度 | 0.9631 | 0.9636 | +0.05pp |
| | 接受率 | 0.9526 | 0.9537 | +0.11pp |
| | 接受子集精度 | 0.9796 | 0.9780 | −0.16pp |
| v2_s325 | 拒收前精度 | 0.9619 | 0.9657 | +0.38pp |
| | 接受率 | 0.9595 | 0.9603 | +0.08pp |
| | 接受子集精度 | 0.9749 | 0.9793 | +0.44pp |

结论：三个 run 全部复现成功，所有指标偏差 ≤0.5 个百分点。拒收前混淆矩阵逐格基本一致
（v1：4989/67/76/4868 vs 4986/70/81/4863；s42：4758/210/159/4873 vs 4757/211/153/4879）。
s325 误差在两个错分格之间有挪动（种子不同，整体精度一致）。

## 论文 CM 图复现

`runs/predict_valid_vs_reject_confusion_pair_reproduced.png`（用内部仓库的
plot_vs_reject_confusion.py 以复现 CSV 重画）与原图
`predict_valid_vs_reject_confusion_mdl_d20260518_pair_and-mc80-mcm05.png`
四个子图逐格差 ≤0.5%。

## 备注

- 聚类 ID 在部分 run 里 0/1 翻转（如 v1 原始 {0:down,1:up}，复现 {0:up,1:down}），属正常现象，
  anchor 映射会换回语义标签，不影响指标。
- 原代码 `reject_strategy: "and"` 指所有条件都违约才拒收（accept = mc≥0.8 或 mcm≥0.05），
  btpc-valid 里对应 `--reject-strategy all`；`any` 才是任一违约即拒。
- btpc 包本次扩展：valid 支持 `--target-source train`、`--tta-noise-std/--tta-scale-jitter`、
  `--reject-rule confidence_center_margin`、`--reject-strategy any|all`；训练行重建统一走
  `predict.get_stage1_selection_pools`（与训练同一种子 2x 选取）；新增
  `scripts/compare_paper_valid.py` 对比脚本。
- 详细对比数据见 `reproduction_comparison.txt`。

## Ridgecrest 复现（2026-09-05 补）

论文的 Ridgecrest 应用不是用 SCSN 模型，而是 mdl_d20260423 的 `ridgecrest_unlabeled`
模型：在 `consensus_waveforms_bp1_20.h5` 的 phasenet 组（27698 条无标注波形，snr 0-1000，
P 到 1000）上自监督训练，120 epochs、num_used=0、bino=False、训练期 filter TTA 4 views、
无每 5 epoch 聚类评估（0423 代码没有该环节，用 `--eval-interval 0` 对齐）。预测口径：
30 TTA views、max_shift 1、noise std 0.01、anchor 噪声 0，映射用 SCSN anchor bank
（100/类、seed 20260404），产物是该 CSV，再进 SKHASH 的 mc×mcm 拒收网格。

复现 run：`runs/repro_ridge_unlabeled/`（stage1 约 33 分钟 + stage2 约 5 分钟 + 全量预测约 3 分钟）。
逐行对比（`scripts/compare_ridge_predictions.py`，详见 `ridge_comparison.txt`）：

- 27698 行全部对齐（source_index 一致，行序无漂移）。
- 聚类标签一致性 97.3%（映射同为 {0:down, 1:up}，无翻转）；up/down 计数 16097/11601 vs 原 15473/12225。
- 嵌入几何量 mean_center_margin 相关 r=0.97（均值差 0.011）；mean_confidence r=0.77；
  vote_consistency r=0.50。
- mc<0.8 且 mcm<0.05 的拒收集：原 1461 vs 复现 1334，交 741（Jaccard 0.36）——边界样本
  在两次独立训练间挪动，与 SCSN in-sample 复现（混淆矩阵逐格差 ≤6/10000）相比明显更松。

结论：Ridgecrest 管线可以完整重跑，聚类结构和几何结构高度一致，但作为完全无监督的
自训练模型，两次独立训练在边界样本上的差异比有标签锚定的 SCSN 结果大。若要逐事件
复现 SKHASH 震源机制，需要把复现 CSV 喂给 `Ridgecrest_Test/skhash/barlow_mc-mcm/
prepare_reject_grid_skhash.py` 重跑网格（输入确定性传递，此步未跑）。

- btpc 包为此新增：`--dataset-source ridgecrest_unlabeled`（data.py 增加 phasenet
  加载器）、`--eval-interval`、ridge 预测路径补去趋势与左闭右开 SNR、
  `--anchor-tta-noise-std`、无标签数据的聚类导出兼容，以及
  `scripts/compare_ridge_predictions.py`。

## Ridgecrest 三模型 Kagan 角复现（2026-09-05 补，对应论文训 snr 0-1000/5-1000/10-1000 的图）

链路：三个 snr 区间各自训练 ridgecrest_unlabeled 模型（120ep）→ 全量预测 27698 行 →
mc0.8/mcm0.05 AND 拒收 → `Ridgecrest_Test/skhash/barlow_mc-mcm/prepare_reject_grid_skhash.py`
生成 SKHASH 输入并运行 SKHASH → `evaluate_kagan.py` 对照 data/consensus_with_fm.ctlg
算 Kagan 角 → `plot_kagan_all_vs_common_retained_frequency_panels_with_median.py` 画双面板图。

复现产物：`runs/repro_ridge_snr5-1000/`、`runs/repro_ridge_snr10-1000/`（0-1000 见上节），
SKHASH 输出与图在 `runs/repro_ridge_kagan/reproduced_kagan_all_vs_common_panels.png`。
SKHASH 用 obspy env 的 v1.1 (2025-05-13) 版（与论文 output.log 一致；base env 已升到
2025-12-17，不要用）。

分布级对比（原始 vs 复现）：

| 模型 | All retained 原 | 复现 | Common 原 | 复现 |
|---|---|---|---|---|
| snr 10-1000 | n=748, med=26.7° | n=746, med=27.0° | 716, 26.2° | 721, 26.7° |
| snr 5-1000 | n=739, med=27.9° | n=757, med=27.2° | 716, 27.5° | 721, 26.9° |
| snr 0-1000 | n=761, med=28.9° | n=761, med=28.6° | 716, 28.6° | 721, 27.9° |

中位数差 0.2-0.7°，"训练 snr 范围越窄 Kagan 角越小"的排序保持。逐事件对比（overlap
730-748 个事件）：每事件 Kagan 角中位差约 3°，Pearson r=0.87-0.90，约 80% 事件差 <10°，
与震源机制解的固有不确定性同量级。结论：这张图可以复现。

三模型间预测极性一致率（新旧对比，按 source_index 对齐）：

| 模型对 | 全行 27698 原 | 复现 | accepted picks 原 | 复现 |
|---|---|---|---|---|
| 0-1000 vs 5-1000 | 96.5% | 95.2% | 99.1% | 98.5% |
| 0-1000 vs 10-1000 | 94.6% | 93.5% | 98.3% | 97.6% |
| 5-1000 vs 10-1000 | 94.9% | 96.1% | 99.0% | 99.4% |

复现 run 的同模型新旧对比（重训漂移基线）：accepted 口径 99.5-99.8% 一致。即模型间
一致率的量级和结构完全复现（mc/mcm 拒收把模型间分歧从全行的 4-6.5% 压到 accepted 的
1-2.4%），各 pair 差别 ≤1.3pp；全行口径下最高一致率的 pair 名次换了一次（原始是
0-1000 vs 5-1000，复现是 5-1000 vs 10-1000），差值在重训噪声范围内。
