# A-CFR: Anchored-Regularization CFR

首个具有 last-iterate 线性收敛理论的 model-free 深度 CFR：删除 DeepCFR
脉络的平均策略网络/蓄水池缓冲/历史网络存储。目标 ICLR 2027。

核心更新（逐信息集乘性形式，锚 μ 每 K 步快照，温度 τ 固定不退火）：

    sigma^{t+1}(a|I) ∝ sigma^t(a|I)^{1/(1+ητ)} · μ(a|I)^{ητ/(1+ητ)}
                       · exp( η·q̃(I,a) / (1+ητ) )

采样版 = λ-估计器（DREAM↔ESCHER 连续插值）+ 小批量 + 超相位调度；
神经版 = 上述 + 三联动调度（η/K/lr）+ Q-anchor + Q-replay
（统一原则："锚定一切慢变量"）。

## 目录

| 路径 | 内容 |
|---|---|
| `01..20_*.md` | 按编号的研究日志：每轮"结果判读 + 下一阶段设计"（含全部失败记录） |
| `code/` | 自含博弈引擎 + 算法 + 各阶段实验驱动（`run_phase*.py`） |
| `results/` | 全部实验 CSV 与图 |
| `paper/` | 论文 LaTeX 骨架（P2.5）与拼装笔记 |

## 代码地图（code/）

| 文件 | 内容 |
|---|---|
| `games.py` | EFG 树引擎；Kuhn / Leduc / Liar's Dice(sides)；节点带 meta 供特征编码 |
| `exploitability.py` | 精确 best response 与 NashConv |
| `algorithms.py` | CFR / CFR+ / 表格 A-CFR |
| `sampling.py` | 采样版 A-CFR（λ-估计器）+ OS-MCCFR 基线 |
| `neural_acfr.py` | 神经版（最终配方开关：`lr_couple, q_anchor, q_replay, encoding`） |
| `features.py` | 泛化特征编码（Kuhn/Leduc/Liar's Dice），替换 one-hot |
| `run_phase0*..4a*.py` | 各阶段实验（详见对应编号文档） |

## 运行

```bash
pip install -r requirements.txt
cd code
python run_phase0.py --quick      # 自检 + 冒烟
python run_phase3f.py             # 神经最终配方的 400k 验证（示例）
```

## 状态（2026-06-12）

- 表格/采样层：理论预测 15 项中 13 项定量命中（见 10 号文档全景表）。
- 理论：Thm 1 / Thm 3 / Prop 4 证完；Thm 2(b)（线性率全文）为收尾主项。
- 神经层：P3 系列收口，误差三分解（拟合偏差可忽略 / Q 漂移 / SGD 噪声）
  + 三个锚修复，Kuhn 上神经 min 0.0088 < tabular 0.0181（P3f）。
- 进行中：Leduc 神经版（one-hot vs 真特征）；下一步 Liar's Dice 与
  OpenSpiel 对比（Ubuntu 侧）。

## 迁移说明（Windows 初验 → Ubuntu 主战场）

本仓库在 Windows 上仅做初步验证；正式对比实验（OpenSpiel、DREAM/ESCHER/
VR-DeepPDCFR+/MMD 基线复现、大博弈）在 Ubuntu 上进行。博弈引擎与算法
解耦：Ubuntu 侧只需补一个 OpenSpiel state→infoset 适配层，算法零改动。
