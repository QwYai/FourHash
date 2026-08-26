# DCMH-F-SemInit：固定特征协议下的监督哈希锚点

实现：`new_hash_work_2026/mixed_gallery/encoders/dcmh_feature.py`

## 当前结论

首轮 **DCMH-F-RandomInit** 的 MIRFLICKR 产物无效，不能作为论文中的编码器证据：20 个 epoch 后 pairwise NLL 仍约为 `log(2)=0.6931`，跨模态 buffer agreement 约 0.5，码没有学到语义。该失败产物应删除或永久标记为 failed diagnostic。

审计后保留的候选版本必须称为 **DCMH-F-SemInit**：固定 512-D 特征和固定 split 上，先用训练标签派生的共享语义码 warm-start 两个独立分支，再执行未改变相对权重的 DCMH alternating objective。它仍不是作者官方复现。

如果 MIR 64-bit 重跑不能通过脚本内置质量门，则将 DCMH-F 判定为 **no-go**，不再扩展数据集/码长，第三锚点改用 UCCH-F。

依据：Qing-Yuan Jiang and Wu-Jun Li, *Deep Cross-Modal Hashing*, CVPR 2017，DOI `10.1109/CVPR.2017.348`。

- 论文页面：<https://openaccess.thecvf.com/content_cvpr_2017/html/Jiang_Deep_Cross-Modal_Hashing_CVPR_2017_paper.html>
- 作者代码：<https://github.com/jiangqy/DCMH-CVPR2017>
- 对照过的常用 PyTorch 实现：<https://github.com/WendellGul/DCMH>

## 失败原因与缩放审计

代码采用行优先记号，`F,G,B` 的形状为 `n_train × bits`。DCMH 目标是：

\[
\mathcal{J}=\sum_{i,j}
\left[\operatorname{softplus}(\Theta_{ij})-S_{ij}\Theta_{ij}\right]
+\gamma(\lVert B-F\rVert_F^2+\lVert B-G\rVert_F^2)
+\eta(\lVert\sum_iF_i\rVert_2^2+\lVert\sum_iG_i\rVert_2^2),
\]

\[
\Theta_{ij}=\tfrac12F_i^\top G_j,qquad
S_{ij}=\mathbb{1}[L_i^\top L_j>0],qquad
B=\operatorname{sign}(F+G).
\]

常用 PyTorch 实现把一个 mini-batch 的三项总和一起除以 `batch_size × n_train`。这是整个 batch surrogate 的共同乘数，没有改变 pairwise、quantization、balance 的相对权重。本实现继续保留这一做法；没有通过“分别求 mean”偷偷重设 `gamma/eta`。

首轮失败来自优化起点与固定特征头尺度的组合：

1. 常用 fork 以互相独立的随机 `F/G` buffer 开始，但论文算法本身并未把这一 buffer 初始化规定为核心方法。
2. 固定特征小头的初始连续输出 RMS 约为 `1e-2`，远小于原始深网常见激活尺度。
3. 当随机 buffer 被这些小输出逐步覆盖时，`theta≈0`；双线性 pairwise 梯度又依赖另一分支的输出，因此语义梯度一起衰减。
4. 在真实 `n_train` 下，量化梯度经共同除数后约带有 `bits/n_train` 的尺度，20 epoch 不足以从随机 `B` 拉出语义结构。
5. 结果就是 NLL 停在 `log(2)`、正负 pair 的 Hamming distance 无差异，甚至全体样本得到同一码。

本地复现的 64-bit 随机初始化失败诊断为：NLL `0.6945`、正负 Hamming gap `0`、图像/文本各只有一个 unique code row，质量门全部失败。这与首轮 MIR 现象一致。

## DCMH-F-SemInit 的修复

修复没有改变 DCMH alternating objective，而是更换起点：

1. 只取 `labels[train_idx]`，每行按正标签数归一化，再减去训练集均值。
2. 使用固定 seed 的 Gaussian projection 投影到目标 bit 数并取 sign，得到共享训练码 `B0`。
3. 图像和文本小 MLP 分别用 Adam/MSE 拟合同一个 `B0`；默认 20 个 warmup epoch。
4. 用 warm-start 后的真实分支输出初始化 `F/G` buffer，并置 `B=sign(F+G)`。
5. 切换为 SGD，按原 DCMH 次序更新图像分支、文本分支和离散 `B`。

SemInit 没有使用新标签：DCMH 的 `S` 本来就由同一训练标签重叠矩阵产生。但“把标签投影码用于初始化”是明确的算法偏离，所以：

- 表格与正文必须写 **DCMH-F-SemInit**，不能只写 DCMH。
- metadata 会记录 target hash、投影 seed、target bit balance、warmup loss 全历史与初始 buffer 尺度。
- `--initialization random` 只用于失败机理消融，名称为 `DCMH-F-RandomInit-ablation`。

## 保留和偏离原工作的边界

保留：

1. 自然正率下的跨模态 Bernoulli negative log-likelihood。
2. 共享离散训练码及 `||B-F||² + ||B-G||²`。
3. 全训练集 bit-balance 项。
4. 交替图像/文本 SGD，随后 `B=sign(F+G)`。
5. 冻结后独立导出 `sign(f(x))` 与 `sign(g(y))`。
6. `sign(0)=+1`。

偏离：

| 项目 | DCMH 2017 | 本实现 |
|---|---|---|
| 图像输入 | 原始图像与 CNN | 固定 512-D 图像特征 |
| 文本输入 | tag/BOW 与文本网络 | 固定 512-D 文本特征 |
| 网络 | 原始深网 | 权重独立的小 MLP/线性头 |
| split | 作者协议 | 项目固定 `indT/indQ/indD` |
| 初始化 | 论文未规定 SemInit | 训练标签共享码 warm-start |
| 数值实现 | 2017 MATLAB/TensorFlow | PyTorch 2.x、稳定 softplus |
| 默认训练量 | 常用 fork 500 epoch | SemInit 20 + DCMH 20 |
| 码长 | 主要报告 16/32/64 | 128 是协议扩展 |

推荐论文表述：

> DCMH-F-SemInit is a controlled feature-based implementation that retains the DCMH alternating objective but replaces the raw-input networks with two fixed-feature heads and uses a train-label-only semantic warm start. It is not an official reproduction of DCMH.

## 防泄漏约束

优化入口只能接收：

- `image[train_idx]`
- `text[train_idx]`
- `labels[train_idx]`

SemInit 和 DCMH pair 构造均局限于这些数组。query/database 的非训练标签不会传入训练函数；两个分支冻结后才编码全集。逐样本 L2 normalization 不估计跨样本统计量。

loader 支持 MSCOCO、MIRFLICKR、NUS-WIDE 的固定 split，以及只作 negative control 的 CIFAR-10；会检查 512-D 对齐、标签格式、索引基数、重复/越界和 split 覆盖关系。

## 强制质量门

训练完成后，脚本在**训练集全部 `n_train²` 有序跨模态 pair**上计算：

- 自然正 pair 比率；
- pairwise NLL 与 `log(2)`；
- 正/负 pair 的平均 theta；
- 正/负 pair 的平均 Hamming distance；
- 两模态 unique code rows、bit balance 和 paired agreement。

默认必须同时满足：

1. `pairwise_nll <= log(2)-0.05`；
2. `negative_HD-positive_HD >= max(1, 0.03×bits)`；
3. 两模态都不允许全同码；
4. 两模态平均绝对 bit mean 均小于 `0.95`。

训练门通过后，脚本冻结两分支并在固定 `query_idx → database_idx` 上计算完整图库的 I2T/T2I mAP。每个 query 的随机排序期望 AP 使用其真实相关项数的解析期望计算，不需要抽取“幸运随机 seed”。held-out 门要求：

1. I2T 与 T2I 的 mAP 均至少高于各自随机排序期望 `0.05`；
2. 两个方向均满足相关 pair 平均 Hamming distance 小于不相关 pair。

held-out 指标只作冻结后验收，不回传训练，也不能据此调整 warmup、epoch 或网络。训练门或 held-out 门任何一个失败时，脚本默认**不生成 NPZ**。`--allow-failed-quality-gate` 仅供保存失败诊断，产物 metadata 会保留 `passed=false`，不得用于论文证据。

## 建议 MIR 重跑命令

先重新同步修复后的脚本，再只跑 MIR 64 bit：

```bash
python new_hash_work_2026/mixed_gallery/encoders/dcmh_feature.py train \
  --dataset mirflickr \
  --data-root /path/to/Data/ProcessData \
  --output /path/to/work/codes/dcmh_f_seminit_mirflickr_64_seed20260805.npz \
  --bits 64 \
  --initialization semantic \
  --warmup-epochs 20 \
  --warmup-lr 0.003 \
  --epochs 20 \
  --batch-size 128 \
  --hidden-dim 256 \
  --lr 0.0316227766 \
  --min-lr 0.000001 \
  --gamma 1 \
  --eta 1 \
  --seed 20260805 \
  --device cuda
```

不要加入 `--allow-failed-quality-gate`。只有命令正常生成 NPZ，且输出同时显示 `quality_gate_passed=true`、`heldout_quality_gate_passed=true`、`overall_usable=true`，才进入 mixed-gallery 评测。

停止规则：若 MIR 仍失败，不增加 warmup、反复改 seed 或依据 query mAP 调参，直接将 DCMH-F 标为 no-go，转向 UCCH-F。若 MIR 通过，先检查冻结码的常规 I2T/T2I 检索有效性，再决定是否扩到 COCO/NUS 和其它 bit。

## `.npz` 输出

| key | dtype | shape | 含义 |
|---|---|---|---|
| `image_codes` | `int8` | `N × bits` | 全体独立图像码，值域 `{-1,+1}` |
| `text_codes` | `int8` | `N × bits` | 全体独立文本码，值域 `{-1,+1}` |
| `labels` | `uint8` | `N × C` | 评价用 multi-hot 标签 |
| `train_idx` | `int64` | `N_train` | 零基训练下标 |
| `query_idx` | `int64` | `N_query` | 零基查询下标 |
| `database_idx` | `int64` | `N_database` | 零基图库下标 |
| `metadata_json` | Unicode scalar | scalar | 方法边界、初始化、目标、split/hash、质量门、训练历史与运行版本 |

输出禁止隐式覆盖。加载时必须使用 `allow_pickle=False`。

## 新的可学习性 smoke

```powershell
python new_hash_work_2026/mixed_gallery/encoders/dcmh_feature.py smoke --device cpu --epochs 10 --warmup-epochs 15 --seed 20260805
```

2026-08-05 实际结果：

| bit | train NLL | `log(2)` | train 正 HD | train 负 HD | I2T mAP | T2I mAP | 随机 mAP 约值 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 0.2690 | 0.6931 | 3.30 | 9.35 | 0.898 | 0.900 | 0.29 |
| 32 | 0.1228 | 0.6931 | 6.88 | 19.22 | 0.989 | 0.987 | 0.29 |
| 64 | 0.0448 | 0.6931 | 13.37 | 38.33 | 0.998 | 0.997 | 0.29 |
| 128 | 0.1850 | 0.6931 | 26.70 | 77.24 | 0.998 | 1.000 | 0.29 |

四种码长均同时通过：双分支非零梯度、参数更新、NLL 显著低于 `log(2)`、相关 pair 明显更近、双向 mAP 超过 20 次随机码均值，以及无 pickle 的 NPZ round-trip。

该测试证明修复后的优化链路能学习已知跨模态语义；真实 MIR 是否通过质量门仍是保留该锚点的必要条件。
