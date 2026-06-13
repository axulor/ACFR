# Phase 0 代码：A-CFR 核心 idea 验证

## 运行

```
pip install numpy matplotlib
cd /d D:\DESKTOP\CFR\CFRforICLR\code
python run_phase0.py --quick     # 冒烟测试（~1-2 分钟），先跑这个
python run_phase0.py             # 完整实验（约 20-40 分钟，E5 Leduc 最慢）
```

输出到 `../results/`（5 张图 + 5 个 CSV）。

## 文件

| 文件 | 内容 |
|---|---|
| `games.py` | 通用 EFG 树引擎；Kuhn（12 信息集）与 Leduc（含两轮下注、加注 2/4、每轮最多 2 次加注） |
| `exploitability.py` | 精确 best response（按深度降序解信息集）与 NashConv |
| `algorithms.py` | CFR、CFR+、**A-CFR**（核心更新式见文件头注释；锚模式 fixed / periodic / ema） |
| `run_phase0.py` | 自检 + 实验 E1–E5 |

## 自检（运行时自动断言）

Kuhn 信息集数 == 12；CFR 平均策略博弈值 ≈ −1/18；NashConv < 0.05。任一失败会直接报错。

## 实验与预期结果（对应 02 文档的理论预测）

| 实验 | 验证内容 | 成功判据 |
|---|---|---|
| E1 (fig1) | last-iterate 总览 | CFR/CFR+ 的 **last** 曲线振荡不收敛；A-CFR 移动锚的 last 曲线持续下降 |
| E2 (fig2) | **预测(a)** 固定锚线性收敛 | 半对数图上距离衰减为直线（Thm 1） |
| E3 (fig3) | **预测(a)** gap 底 ~ O(τ) | log-log 图上 floor-τ 斜率 ≈ 1（Lemma 2） |
| E4 (fig4) | **预测(b)** 移动锚 → 精确 Nash | τ 固定不退火，NashConv(last) 持续下降趋 0（Thm 2）；K 太小可能失稳（理论允许，ε_k 条件） |
| E5 (fig5) | Leduc 上复现 | 同 E1 趋势在更大博弈上保持 |

## 失败模式应对（02 文档 §5）

- E4 中 K=10 发散而 K=200 收敛属预期（内环精度 ε_k 条件）→ 已实现 `anchor_mode="adaptive"`。
- Leduc 上未归一化 A-CFR 收敛慢（O1 障碍，首轮已实测）→ 已实现 `normalize=True`（反事实值按信息集对手×机会 reach 归一化，锚相位内冻结）。

## Phase 0b（首轮结果判读后的跟进实验，见 04 文档）

```
python run_phase0b.py --quick   # 冒烟
python run_phase0b.py           # 完整（~15-30 分钟，B4 最慢）
```

B1 收缩率 vs 理论值对照表；B2 τ 底加密到 0.01；B3 自适应锚打 1e-3；B4 Leduc 归一化修复验证。
