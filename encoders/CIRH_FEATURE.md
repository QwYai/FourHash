# CIRH-F 受控特征适配审计

状态：**代码、单测和最小 smoke 已通过；正式 MIRFlickr-64 训练命令已冻结，但未启动。** 资产门禁已降轨，须等待 UCCH-F 质量门结果后再决定是否消耗正式训练资源。

## 1. 来源与计数口径

- 论文：Lei Zhu et al., *Work Together: Correlation-Identity Reconstruction Hashing for Unsupervised Cross-Modal Retrieval*, IEEE TKDE 35(9), 2023，DOI `10.1109/TKDE.2022.3218656`。
- 官方仓库：`https://github.com/XizeWu/CIRH`
- 固定审计 commit：`0f6439d3cad0240ea4a9924ff168aa2cef3b3b1e`
- 本实现报告名必须写成 **CIRH-F**：它是固定 CLIP512 特征协议上的受控适配，不是官方数值复现，也不是作者发布的 frozen artifact。

已逐文件审计 `main_mir.py`、`my_opt.py`、`models.py`、`load_data.py`、`utils.py`。文件 SHA-256 已固化在 `cirh_feature.py` 的 `OFFICIAL_SOURCE_SHA256` 中，并会进入每个导出 NPZ 的 metadata。

## 2. 保留的官方核心 objective

CIRH-F 保留官方实现的完整双阶段训练逻辑。

第一阶段首先只用训练图像/文本特征构造 collaborated similarity：

```text
S1 = a1 cosine(I,I) + (1-a1) cosine(T,T)
将每行最小 K 项置零
S2 = symmetrize(2 sigmoid(S1)-1 + I)
S  = a2 S1 + (1-a2) S2
```

每个 batch 按官方代码选取第 4、第 3 强邻居，并构造 `0.7 self + 0.2 third + 0.1 fourth` 的增强特征。联合图网络的损失原样保留三部分：

```text
lambda1 * [MSE(decoded_image, image) + MSE(decoded_text, text)]
+ lambda2 * [MSE(cos(HI),S) + MSE(cos(HT),S) + MSE(cos(H),S)]
+ MSE(H, sign(H))
```

第二阶段再训练独立 image/text hash functions：

```text
MSE(BI,B) + MSE(BT,B) + MSE(BI,BT)
+ beta * [MSE(cos(BI,BT),S) + MSE(cos(BI,BI),S) + MSE(cos(BT,BT),S)]
```

正式默认值保持官方 MIR 配置：`lambda1=10`、`lambda2=1`、`beta=.01`、`K=3000`、`a1=a2=.6`、联合网络学习率 `.001`、图像/文本 hash net 学习率 `.0001`、batch `512`、epoch `60`；图像 hash net 仍为 `512→4096→64`，文本 net 为 `512→512→64`。

## 3. 必要适配和 fidelity caveat

1. 官方数据文件替换为项目固定的 512-D image/text features 和公共 `indT/indQ/indD`；训练不重抽样。
2. 硬编码 `.cuda()`、旧 `Variable` 和旧数据接口替换为现代、显式 device 的 PyTorch。
3. **修复官方第二阶段的索引错位**：官方先把 feature/B 按 shuffled `record_index` 排列，却仍用位置切片 `S[kb:(k+1)b,kb:(k+1)b]`。CIRH-F 明确使用 `S[item_ids][:,item_ids]`。这会使数值不同，但避免把错误样本间相似度塞给 objective。
4. 官方全排序只为把每行最小 K 项置零；适配用等价 `topk(largest=False)` 降低内存。若边界存在完全相等的相似度，具体被置零的同值 item 可能不同。
5. `torch.sign(0)=0` 会产生非二值码；导出时固定 `sign(0)=+1`。
6. 官方每 10 epoch 查看 query mAP 并按 query label 保存 best checkpoint。该路径被完全删除：CIRH-F 固定跑 60 epoch，只用最终 epoch。因而它不会复现官方的 test-selected 最优数值。
7. 手工确定性 permutation 代替 DataLoader shuffle；网络、dropout、BatchNorm 和两阶段更新顺序不变。不同 PyTorch/CUDA 版本仍可能产生小数值差异。

因此论文中只能说“CIRH objective under a controlled fixed-feature protocol”，不能写“official CIRH reproduction”。

## 4. 泄漏防火墙

- 训练入口签名只有 `train_cirh_f(train_image, train_text, config)`，根本不接收 label、query 或 database。
- collaborated graph 只由 `image[indT]` 与 `text[indT]` 构造。
- epoch 数、超参数和最终 checkpoint 在任何 held-out row 被编码前已经冻结。
- query/database labels 只在最终冻结后打开一次，用于资产接受审计和随后导出；不能选择 epoch、checkpoint、超参数、阈值或变体。
- 默认质量门失败时不导出 NPZ；诊断性强制导出必须显式使用 `--allow-failed-quality-gate`，且不能计入 encoder evidence。

## 5. 质量门

先跑无标签 structural gate：

- history 全部有限；
- joint/image/text 三个网络参数均真实更新；
- 两个模态的训练码均有多于一个 unique row；
- 两个模态的 `mean(abs(bit mean)) < .95`，排除全局塌缩。

冻结后再跑一次 held-out audit：I→T 与 T→I full-database mAP 都必须超过各自精确随机排序期望至少 `.05`，并且正对平均 Hamming 距离小于负对。此 mAP 的稳定 item-order tie 处理只用于 encoder gate；论文正式指标仍应使用项目的 closed-form tie-aware evaluator。

## 6. 可行性与停止条件

MIR 训练行数为 5,000。单个 `float32 5000×5000` 矩阵约 100 MB；默认 `5000×3000` 的 int64 top-k 索引约 120 MB。加上模型和临时张量，当前 GPU 足以容纳，故“一周内因全矩阵不可行”的停止条件没有触发。

但该判断只覆盖 MIR。NUS 的单个 `21000×21000` float32 矩阵约 1.76 GB，完整 transient memory 与运行时须单独预检，不能由 MIR 结论外推。

## 7. 已通过验证

```text
python -m unittest discover new_hash_work_2026/mixed_gallery -p "test_*.py" -v
52 tests OK（其中 CIRH-F 新增 8 tests）

python new_hash_work_2026/mixed_gallery/encoders/cirh_feature.py smoke \
  --device cuda --epochs 2 --seed 20260805
PASS；32×16 独立 image/text codes；值域严格 {-1,+1}；三网络均更新；structural gate PASS
```

正式命令位于 `runs/cirhf-mir64-train-seed20260805/command.sh`。当前状态是 `PREPARED_NOT_STARTED`，不得在 UCCH-F gate 决策前启动。

