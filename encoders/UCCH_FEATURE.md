# UCCH-F：固定 512-D 特征的受控适配

## 当前判定

UCCH-F 是一个 **science gate**，不是官方复现，也不是用 UCCH 名义包装的新算法。它回答一个窄问题：在完全相同的 MIRFLICKR 512-D image/text 特征和 common split 上，保留 UCCH 的无监督对比哈希核心目标，能否产出一套独立、非坍塌、跨模态有效的 64-bit 双分支码。

本地状态：

- `ucch_feature.py` 已完成；
- 6 个目标/结构/无泄漏/训练单元测试通过；
- 3 epoch CPU smoke 通过：16 bit 下 paired Hamming gap 为 `4.8213`，image/text 分支及 momentum memory 均发生更新，image/text unique code rows 为 `127/126`（总 128）；
- 正式证据只认 MIRFLICKR、64 bit、seed `20260805` 的预注册配置；不据此扩 COCO/NUS。

## 官方源码锚点

- 论文：Peng Hu et al., *Unsupervised Contrastive Cross-modal Hashing*, TPAMI 2023，DOI `10.1109/TPAMI.2022.3177356`。
- 官方仓库：<https://github.com/penghu-cs/UCCH>
- 审计 commit：`0c20e62b99875cd2ec9d7a496eae80b3ab8ba61b`
- 对照文件：`UCCH.py`、`nets/ImageNet.py`、`nets/TextNet.py`、`NCE/NCEAverage.py`、`NCE/NCECriterion.py`、`src/utils.py`、`utils/config.py`。

官方 README 的 MIRFLICKR feature-mode 命令固定了：20 epochs、batch 256、Adam、lr `1e-4`、image/text depth `3/2`、`alpha=0.7`、`margin=0.2`、`shift=0.1`。其余保留 parser 默认：hidden width 8192、weight decay `1e-6`、K=4096、T=0.9、memory momentum=0.4、warm-up 1 epoch。

## 保留的核心

1. image 与 text 是独立参数分支；网络结构与官方 feature-mode 一致：Linear/ReLU 堆叠，最后 `tanh`，逐样本 L2 normalize。
2. `L_c` 保留双向实例级 contrastive hashing：候选 memory 在相似度计算前取 `sign`，正样本固定在候选第 0 列，logit 除以 `T*sqrt(bit)`。
3. memory 使用同一 paired item 的 image/text 平均表征做 momentum 更新；首 epoch 使用 in-batch 候选且 momentum=0，随后进入 K=4096 memory negatives。
4. `L_r` 保留官方 Cross-modal Ranking Learning 的 margin/shift 掩码与双向 log-sum-exp。
5. 总目标仍为 `alpha*L_c + (1-alpha)*L_r`；Adam、weight decay 和 norm-1 gradient clipping 不变。
6. official `drop_last=True` 行为保留；5000 train rows、batch 256 时每个 epoch 19 个完整 batch，随机遗漏 136 行，下一 epoch 重新 shuffle。

## 必须明示的差异

1. 输入换成项目已经固定的 512-D CLIP-style image/text 特征；不使用 UCCH 自带 VGG/tag 特征。
2. split 换成项目固定 `indT/indQ/indD`，且 optimizer 只读 `indT` 的 paired features。
3. `train_ucch_f(train_image, train_text, config)` 的接口根本没有 labels 参数；训练不读 train/test semantic labels。
4. 官方脚本每个 epoch 用 retrieval labels 算 mAP 并保存 best checkpoint；该行为会把选择信号带入模型。本适配版取消它，预先规定 **最后一个 epoch**，并在打开 held-out labels 之前先写死 final checkpoint 和 SHA256。
5. 官方 `AliasMethod` 的 unigram 全为 1；适配版用 seeded uniform integer sampling，抽样分布相同，但 RNG 序列不宣称逐 bit 一致。
6. 显式 `softmax -> log` 和 `exp -> sum -> log` 分别换为数学等价的 cross-entropy 与 logsumexp，避免数值溢出。
7. 单 GPU、现代 deterministic PyTorch 替代硬编码 CUDA 假设；这不是逐行/逐随机数复现。
8. 导出时极小概率的 exact zero 映射到 `+1`，确保码域严格为 `{-1,+1}`。

因此论文表格和正文必须使用名称 **UCCH-F (controlled adaptation)**，不能写“official UCCH result”或与官方论文表格数字做复现误差承诺。

## 无泄漏顺序

固定执行顺序如下，不允许调换：

1. 读取完整特征、labels、split，只把 `image[indT]` 与 `text[indT]` 传给训练函数；
2. 按预注册配置跑满 20 epochs；不评估任何 mAP，不选择 checkpoint；
3. 用 train pairing（同 item ID）做 label-free gate；
4. label-free gate 通过后保存 final checkpoint，并计算 checkpoint SHA256；
5. 冻结两分支并编码全体 rows；
6. 此后才打开 `labels[indQ]`/`labels[indD]`，只运行一次 post-freeze I→T/T→I acceptance；
7. held-out gate 通过才导出可用于 mixed-gallery 的 NPZ。失败 checkpoint 只能作为 diagnostic，不能进论文主实验。

## 质量门

### Gate A：训练健康（不使用 semantic labels）

必须全部满足：

- 每个 epoch loss 有限；
- image/text 每个 epoch 都有非零梯度，最终参数均改变；
- memory bank 改变；
- paired item 的平均跨模态 Hamming 距离至少比 32 个循环错配 offset 低 `max(1, 0.03*bit)`；64 bit 门槛为 `1.92`；
- image/text unique code rows 均大于 32；
- image/text `mean(abs(bit mean)) < 0.95`。

### Gate B：冻结后 held-out 接受

只运行一次，并沿用 DCMH-F 的同级接受规则：

- I→T 与 T→I full-database mAP 均至少高于相应随机排序期望 `0.05`；
- 两个方向 relevant pairs 的平均 Hamming 距离均低于 irrelevant pairs。

Gate B 不用于选择超参或 checkpoint。即使失败，也不得再根据 test 结果改 `alpha/T/K/depth/epoch`；本 seed 直接判 NULL/BLOCKED。

## 本地验证

```bash
cd new_hash_work_2026/mixed_gallery/encoders
python -m unittest -v test_ucch_feature.py
python ucch_feature.py smoke --epochs 3 --device cpu
```

正式命令已锁定在 `runs/ucchf-mir64-train-seed20260805/command.sh`。

