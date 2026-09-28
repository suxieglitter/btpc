[English](README.md) | 中文

# BTPC — Barlow Twins P 波初动极性分类

BTPC 对地震 P 波的初动极性（up / down）做分类，**全程不需要任何人工标注的训练样本**。把研究区已经拾取好的 P 波波形喂给它，它自己完成训练，输出的极性可以直接接 SKHASH 做初动震源机制反演。

流程分三步：

1. **自监督编码器（Stage 1）。** 一个带注意力的一维残差编码器，用
   [Barlow Twins](https://arxiv.org/abs/2103.03230) 损失在每条 0.32 s P 波
   窗口的两个增广视图上训练。每隔 `filter_interval` 个 epoch，一套无监督
   可靠性评分（波形质量、增广一致性、kNN 邻居一致性、epoch 稳定性、
   聚类间隔）对不可靠样本降权或剔除。
2. **伪标签分类器（Stage 2）。** 编码器特征做二类聚类（默认谱聚类），
   在保留样本上、冻结编码器，训练一个线性分类头。
3. **锚定预测。** 预测时用一小批有标注的 SCSN 锚定波形（随仓库发布在
   `anchors/`）把聚类结果固定到 up/down 语义上；测试时增强（TTA）投票
   加间隔判据把不稳定的预测拒收为不确定（label 2 = uncertain）。

在自有数据上从头到尾的完整工作流：

```
拾取表（CSV 或 SAC）──make_pwave_dataset.py──►  my_data.h5
my_data.h5            ──btpc-train stage1+stage2──►  模型
my_data.h5 + 模型     ──btpc-predict──►  极性 CSV（up/down/uncertain）
极性 CSV              ──export_skhash.py──►  SKHASH 输入 ──SKHASH──►  out.csv
```

每个箭头都是一条命令，见[在自有研究区使用 BTPC](#在自有研究区使用-btpc)。

本包是论文模型所用代码的整理发布版（实验 `mdl_d20260518`，seed-42
run；见 `configs/btpc_0518.yaml`），论文结果可由本包复现（见
[复现论文](#复现论文)）。

## 安装

```bash
git clone <repository-url>
cd btpc
pip install -e .
```

需要 Python 3.10+ 和 PyTorch 2.0+，完整依赖见 `requirements.txt`。两个
可选依赖只在特定步骤需要：**obspy**（`make_pwave_dataset.py` 读
SAC/mSEED 波形用）和 **SKHASH**（`pip install skhash`，震源机制反演用）。

提示：Linux 上直接 `pip install torch` 会拉整套 CUDA 组件（下载
2–3 GB）。只用 CPU 训练的话装 CPU 版即可（约 200 MB）：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## 快速上手（无需任何数据）

`scripts/make_toy_data.py` 生成一个小的合成数据集和配套锚定波形库，
CPU 上一分钟内就能把整条流程跑通。动真实数据之前先用它检查安装：

```bash
python scripts/make_toy_data.py --output-dir runs/toy

# Stage 1（冒烟测试用的小参数）
btpc-train stage1 --data-path runs/toy/toy_scsn.hdf5 \
    --num-used 512 --batch-size 64 --epochs 6 \
    --warmup-epochs 2 --filter-interval 2 --n-tta-views 3 \
    --save-path runs/toy/stage1

# Stage 2（伪标签分类器；输出落在 runs/toy/stage2）
btpc-train stage2 --stage1-dir runs/toy/stage1 --stage2-epochs 2 --batch-size 64

# 在训练文件的非训练划分上验证
btpc-valid --stage2-checkpoint runs/toy/stage2/stage2_cluster_classifier.pth \
    --n-tta-views 2 --chunk-size 1024

# 在训练文件中未使用的行上预测
btpc-predict --stage2-checkpoint runs/toy/stage2/stage2_cluster_classifier.pth \
    --target-source train_unused \
    --anchor-bank-path runs/toy/toy_anchor_bank.npz \
    --max-samples 256 --n-tta-views 2
```

## 在自有研究区使用 BTPC

这是本包的日常用法：在研究区的无标注波形上重新训练，再把得到的极性
拿去反演。你需要准备四样输入：

- **已拾取好的 P 震相**——拾取表 CSV，或 pick 写在 header 字段里的
  SAC 文件；
- **事件目录**——每个事件一行，含位置和震级；
- **台站表**——每个台站一行，含坐标；
- **研究区的一维速度模型**。

### 1. 准备拾取表

每条 P 震相一行的 CSV。`waveform_path` 和 `p_time` 必需；
`record_id`、`event_id`、`station`、`channel` 可选，但
**`event_id` 和 `station` 后面 SKHASH 要用**，建议一并写上：

```csv
waveform_path,p_time,record_id,event_id,station
data/2019-07-04/CI.CCA.2019-07-04.mseed,2019-07-04T17:35:13.078,0,0,CI.CCA
data/2019-07-04/CI.CCC.2019-07-04.mseed,2019-07-04T17:35:06.138,1,0,CI.CCC
```

`waveform_path` 指向任何 obspy 能读的文件（mSEED、SAC……），多条拾取
可以共用一个文件；`p_time` 是任意 ISO 时间格式。如果 pick 在 SAC
header 里，可以不写 CSV，第 2 步直接用 `--sac-dir` +
`--sac-pick-field`——该模式会自动填台站名，但 `event_id` 会缺省，对
第 3、4 步的训练和预测没有影响，第 5 步的 SKHASH 导出则用不了，所以
打算一路走到底的话还是用 CSV 拾取表。

### 2. 截取波形窗口

```bash
python scripts/make_pwave_dataset.py --picks picks.csv --output my_data.h5
# 或：--sac-dir sac/ --sac-pick-field a
```

脚本按论文口径处理——垂直分量（HHZ > BHZ > EHZ > 任意 `*Z`，或拾取表
的 `channel` 列）、重采样到 100 Hz、1–20 Hz 四阶带通、
SNR = max|P..P+0.5 s| / max|P−0.5 s..P|——写出的 `my_data.h5` 里 P
到时在第 1000 个采样点。窗口长度、采样率和滤波拐角都可调
（`--pre-sec`、`--post-sec`、`--freqmin/--freqmax`、`--no-filter`）；
保持默认即与论文处理口径一致。细节见 `docs/data.md` 的
*Preparing your own dataset* 一节。

### 3. 无标注训练

```bash
btpc-train stage1 --dataset-source unlabeled --data-path my_data.h5 \
    --num-used 0 --epochs 120 --n-tta-views 4 --eval-interval 0 \
    --save-path runs/mine/stage1
btpc-train stage2 --stage1-dir runs/mine/stage1
```

`--num-used 0` 用文件里的全部记录；全程不读任何极性标注。模型落在
`runs/mine/stage2/stage2_cluster_classifier.pth`。

### 4. 预测极性

```bash
btpc-predict --stage2-checkpoint runs/mine/stage2/stage2_cluster_classifier.pth \
    --target-source ridge --data-path my_data.h5 \
    --anchor-bank-path anchors/anchor_bank_scsn_50_1000.npz \
    --n-tta-views 30 --tta-max-shift 1 --tta-noise-std 0.01 --anchor-tta-noise-std 0
```

锚定波形库随仓库发布（10,000 条有标注的 SCSN 波形，预测时每类抽
100 条）。论文的 Ridgecrest 应用跨区用的正是这同一个库。
`--target-source ridge` 是读取 `phasenet` 组布局的模式（名字承自论文
的 Ridgecrest 数据，对第 2 步产出的任何文件都有效）。预测结果、置信
度和间隔写在
`runs/mine/stage2/predict_out/predictions_anchor_mapped.csv`。

### 5. 用 SKHASH 反演

准备三样研究区输入（列名别名见 `docs/skhash.md`）：

```csv
# events.csv: event_id,lat,lon,dep,mag,time
0,35.6434,-117.566081,6.6,4.28,2019-07-04T17:35:01

# stations.csv: station,lat,lon,ele        （station 写 NET.STA 或裸台站代码）
CI.APL,35.34149,-116.87464,959.0

# vel.txt: depth_km,Vp_km_s
0.0,1.82
0.52,5.00
15.0,7.20
```

然后导出并运行：

```bash
python scripts/export_skhash.py \
    --predictions runs/mine/stage2/predict_out/predictions_anchor_mapped.csv \
    --data-path my_data.h5 \
    --events events.csv --stations stations.csv --vmodel vel.txt \
    --output-dir skhash_run
SKHASH skhash_run/control_auto.txt
```

导出脚本按论文的拒收规则筛选（平均置信度 < 0.8 且平均中心间隔 <
0.05，两者都可调），同一事件在同一台站被拾取多次时保留置信度最高的
一条，并写出台站表、事件表、极性表和带论文 SKHASH 参数的控制文件。
震源机制解落在 `skhash_run/output/out.csv`。要与独立目录对比，用任意
标准的 Kagan (2007) 双力偶旋转角实现算 Kagan 角即可——该评估步骤不
在本包范围内。完整细节和可调的控制文件参数见 `docs/skhash.md`。

## 用预训练的论文模型直接预测

`models/0518_42/` 随仓库发布论文的 SCSN 训练模型（SCSN 精制数据、
seed 42、80 epochs，即 `configs/btpc_0518.yaml` 那个 run）。
`stage2_cluster_classifier.pth` 是预测用的 checkpoint（encoder + 分类
头 + 聚类中心）；`stage1_final_pwave_model.pth` 是裸的 Stage 1 编码
器，留给分析用。它经本包复测复现了论文验证数字（拒收前精度 0.9630、
接受率 0.9528、接受样本精度 0.9793；见 `docs/reproduction.md`）。

不重训直接预测——例如对第 2 步产出的 `phasenet` 布局文件：

```bash
btpc-predict --stage2-checkpoint models/0518_42/stage2_cluster_classifier.pth \
    --target-source ridge --data-path my_data.h5 \
    --anchor-bank-path anchors/anchor_bank_scsn_50_1000.npz \
    --n-tta-views 30 --tta-max-shift 1 --tta-noise-std 0.01 --anchor-tta-noise-std 0
```

注意：这个模型在 SCSN 数据上训练和验证；不重训直接迁移到其他区域的
效果没有在论文里验证过。新研究区仍推荐上面的自训练路线——预训练模型
的定位是 SCSN 式数据、快速试用，以及给自己重训的模型当对照基线。

### 完整演示命令（可整块照抄）

在 100 个事件的 Ridgecrest 演示子集上实测（2,222 条拾取，CI 台网，
横跨 M6.4 与 M7.1 两次主震）：截波形 8 秒、预训练模型预测 22 秒、
SKHASH 12 秒——CPU 全程约 45 秒。先把四个输入放进工作目录
（`picks.csv`、`events.csv`、`stations.csv`、`vel.txt`；格式见
[docs/data.md](docs/data.md) 与 [docs/skhash.md](docs/skhash.md)）。

```bash
# 1) 截 P 波窗口 -> my_data.h5（论文处理口径，P 在采样点 1000）
python scripts/make_pwave_dataset.py --picks picks.csv --output my_data.h5

# 2) 用仓库自带论文模型预测（无需训练）
btpc-predict --stage2-checkpoint models/0518_42/stage2_cluster_classifier.pth \
    --target-source ridge --data-path my_data.h5 \
    --anchor-bank-path anchors/anchor_bank_scsn_50_1000.npz \
    --n-tta-views 30 --tta-max-shift 1 --tta-noise-std 0.01 \
    --anchor-tta-noise-std 0 --save-dir demo_predict

# 3) 导出 SKHASH 输入（默认论文拒收口径：mc<0.8 且 mcm<0.05）
python scripts/export_skhash.py \
    --predictions demo_predict/predictions_anchor_mapped.csv \
    --data-path my_data.h5 \
    --events events.csv --stations stations.csv --vmodel vel.txt \
    --output-dir skhash_run

# 4) 反演 -> skhash_run/output/out.csv（附 beachball 图）
SKHASH skhash_run/control_auto.txt
```

要走自训练路线，把第 2 步换成：

```bash
btpc-train stage1 --dataset-source unlabeled --data-path my_data.h5 \
    --num-used 0 --epochs 120 --n-tta-views 4 --eval-interval 0 \
    --save-path runs/mine/stage1
btpc-train stage2 --stage1-dir runs/mine/stage1
btpc-predict --stage2-checkpoint runs/mine/stage2/stage2_cluster_classifier.pth \
    --target-source ridge --data-path my_data.h5 \
    --anchor-bank-path anchors/anchor_bank_scsn_50_1000.npz \
    --n-tta-views 30 --tta-max-shift 1 --tta-noise-std 0.01 \
    --anchor-tta-noise-std 0 --save-dir demo_predict
```

同一份 100 事件子集上，2,222 条 × 120 epochs 训练约 10 分钟（CPU）。

## 复现论文

真实数据就位后（见 [docs/data.md](docs/data.md)），参考配置可以复现
论文模型。论文结果的全套从头复现（SCSN 模型、混淆矩阵图、Ridgecrest
预测和 Kagan 角对比）见
[docs/reproduction.md](docs/reproduction.md)。

```bash
btpc-train stage1 --config configs/btpc_0518.yaml --save-path runs/0518_42/stage1
btpc-train stage2 --stage1-dir runs/0518_42/stage1
btpc-valid --stage2-checkpoint runs/0518_42/stage2/stage2_cluster_classifier.pth
```

YAML 里的任何值都可以在命令行覆盖，如 `--epochs 40 --batch-size 128`。
要严格复现论文的 in-sample 验证数字，需在重建的 Stage 1 训练行上、
用论文的 TTA 和拒收设置评估（仅当平均置信度 < 0.8 且平均中心间隔 <
0.05 时拒收），再用 `scripts/compare_paper_valid.py` 对比：

```bash
btpc-valid --stage2-checkpoint runs/0518_42/stage2/stage2_cluster_classifier.pth \
    --target-source train --n-tta-views 30 --tta-max-shift 0 --tta-noise-std 0.015 \
    --reject-rule confidence_center_margin --reject-strategy all
```

### 论文的 Ridgecrest 应用

论文在 Ridgecrest 共识文件的无标注 `phasenet` 组（27,698 条波形，P
到时在第 1000 个采样点）上自训练，对同样的行做预测，再把接受的极性
喂给 SKHASH——就是上面那条自有区域流程在论文数据上的实例：

```bash
btpc-train stage1 --data-path /path/to/consensus_waveforms_bp1_20.h5 \
    --dataset-source ridgecrest_unlabeled --no-bino --num-used 0 \
    --epochs 120 --n-tta-views 4 --eval-interval 0 --save-path runs/ridge/stage1
btpc-train stage2 --stage1-dir runs/ridge/stage1
btpc-predict --stage2-checkpoint runs/ridge/stage2/stage2_cluster_classifier.pth \
    --target-source ridge --data-path /path/to/consensus_waveforms_bp1_20.h5 \
    --anchor-bank-path anchors/anchor_bank_scsn_50_1000.npz \
    --n-tta-views 30 --tta-max-shift 1 --tta-noise-std 0.01 --anchor-tta-noise-std 0
```

`--dataset-source ridgecrest_unlabeled` 和 `unlabeled` 是别名：两者读
的都是 `make_pwave_dataset.py` 写出的 `phasenet` 组布局。

## 参考超参数

Stage 1 默认值（也在 `configs/btpc_0518.yaml` 里）：

| 参数 | 取值 | | 参数 | 取值 |
|---|---|---|---|---|
| `snr_range` | [0, 1000) | | `aug_shift` | 1 |
| `num_used` | 10000 | | `aug_noise_std_range` | [0.05, 0.2] |
| `resize` | 32 | | `aug_scale_range` | [0.8, 1.2] |
| `shift` | 0 | | `enable_periodic_filtering` | true |
| `bino` | true | | `warmup_epochs` | 20 |
| `base_cha` | 16 | | `filter_interval` | 10 |
| `projector_dims` | 512 | | `n_tta_views` | 30 |
| `batch_size` | 256 | | `tta_max_shift` | 2 |
| `lr` | 0.001 | | `knn_k` | 10 |
| `lambda_param` | 0.005 | | `low/high_score_threshold` | 0.45 / 0.7 |
| `epochs` | 80 | | `low_score_patience` | 2 |
| `cluster_feature` | encoder | | `w_q/w_aug/w_knn/w_stab/w_margin` | 0.15/0.2/0.2/0.25/0.2 |
| `norm_mod` | max | | `downweight_loss_scale` | 0.35 |
| `eval_source` | clean | | `stability_ema_alpha` | 0.5 |
| `stage1_cluster_method` | spectral | | `seed` | 42 |

Stage 2 默认值：`stage2_cluster_method` spectral、`stage2_epochs` 30、
`stage2_lr` 0.001、`stage2_weight_decay` 1e-4，降权样本保留参与训练，
编码器冻结，线性分类头。验证/预测默认值：`n_tta_views` 4、
`tta_max_shift` 2、投票阈值 0.8、间隔阈值 0.15。

## 输出

- `stage1/`：`final_pwave_model.pth`、`stage1_config.json`、
  `stage1_summary.json`、`filtering/`（逐 epoch 筛选 CSV/NPZ 和时间
  线）、`cluster_results/`（CSV、t-SNE/PCA 图、混淆矩阵、示例波形）、
  `train_out/`。
- `stage2/`：`stage2_cluster_classifier.pth`（encoder + 分类头 + 聚类
  中心 + 特征归一化统计）、`pseudo_labels.csv`、`stage2_summary.json`、
  `cluster_results/`。
- `valid_out/`：`valid_predictions.csv`（逐样本指标和最终标签）、
  `valid_summary.json`、拒收前后的混淆矩阵、指标直方图。
- `predict_*_out/`：`predictions_anchor_mapped.csv`（逐记录极性、置信
  度、间隔）、`predict_summary.json`、聚类/极性分配的 t-SNE 图。
- SKHASH 运行目录（`export_skhash.py` 产出）：`input/sk_sta.csv`、
  `input/sk_ctlg.csv`、`input/polarities.csv`、`control_auto.txt`、
  `input/predictions_for_skhash.csv`、`input/rejected_predictions.csv`、
  `export_summary.json`，以及 SKHASH 的 `output/out.csv`。

## 许可证

MIT——见 [LICENSE](LICENSE)。
