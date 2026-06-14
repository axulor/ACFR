# 工作报告：Ubuntu 迁移 + 神经-vs-神经正面对比（2026-06-13/14，约 12h 自主）

> 目标（用户定）：把 A-CFR 与众多基线正面对比，**最优对最优**（我方
> last-iterate vs 基线 avg-iterate）在真效率指标上拿出**突出优势**才达 ICLR 标准。
> 本报告诚实记录已建设施、已得结果、根因分析、与目标的差距、下一步。

---

## 0. 一句话结论

**部分达标，未完全达标。** A-CFR 在**早期/中等精度样本效率**上有真实优势
（小博弈 Kuhn 全程胜、Leduc 到 exp≤0.5 快 3.9×），但在**高精度**上其
last-iterate floor（Prop-4 带）高于基线的平均策略渐近，且**wall-clock 偏慢、
last-iterate 有漂移不稳定**。在争议中博弈 Leduc 上**未在主指标（最终
exploitability）胜过 SOTA**。根因清楚（见 §4），下一步算法方向已提出（§6）。

## 1. 已建设施（可复用）

- **环境**：`deeppdcfr`（py3.9 + open_spiel 1.4 + TF 2.15 + torch 2.8）跑基线
  框架 + 我方代码（同环境对比）；`acfr`（torch 2.11）我方开发。
- **OpenSpiel 适配层** `code/os_adapter.py`：算法零改动跑标准博弈 + 精确
  NashConv；规模与 SOTA 论文 Table 1 完全吻合（Leduc 936 / LiarsDice6 24576）。
- **对比管线** `code/comparison/`：`run_ours`（我方 last-iter，OpenSpiel
  exploitability=NashConv/2 口径）、`run_baseline`（跑 DeepPDCFR 7 基线、解析
  曲线同格式）、`campaign`（矩阵编排、并发、幂等）、`analyze`（seed 平均、多
  阈值样本效率、图）。
- **我方神经版接入 OpenSpiel 泛化特征**（info_state_tensor）：I_enc=当前玩家
  tensor（跨信息集泛化），H_enc=双方 tensor concat（区分历史）。
- 代码已整理：core 扁平 / experiments / diagnostics / comparison + README。
- 基线全部跑通：NFSP/OSDeepCFR/DREAM/VRDeepDCFR+/VRDeepPDCFR+（=AAAI'26 SOTA）。

## 2. 对比结果（1M episodes，同口径，exploitability=NashConv/2）

### Kuhn（12 信息集）—— 我方样本效率全程胜
| 方法 | final | 到 exp≤0.1 | 到 exp≤0.05 |
|---|---|---|---|
| **A-CFR neural onehot (ours)** | 0.0074 | **19200** | **23927** |
| A-CFR sampled (ours) | 0.0080 | 19200 | 23927 |
| VRDeepPDCFR+ (SOTA) | 0.0048 | 170484 | 227216 |
| DREAM | 0.0165 | 113752 | 198850 |
| OSDeepCFR | 0.0276 | 109025 | 170484 |

**我方达标样本数比基线少 1.5×(exp0.5) → 5.7×(exp0.1) → 7.1×(exp0.05)**，优势随
精度提高而增大。final 与 SOTA 同量级（略高）。**清晰的样本效率优势。**

### Leduc（936 信息集）—— 早期胜、高精度负（交叉）
| 方法 | seeds | final | min | 到 exp≤0.5 |
|---|---|---|---|---|
| VRDeepPDCFR+ (SOTA) | 4 | **0.147** | 0.147 | 175212 |
| VRDeepDCFR+ | 2 | 0.173 | 0.154 | 165756 |
| DREAM | 4 | 0.233 | 0.232 | 203577 |
| OSDeepCFR | 4 | 0.357 | 0.355 | 402138 |
| **A-CFR neural FEATURES (ours)** | 4 | 0.441 | **0.330** | **42838** |
| A-CFR neural onehot (ours) | 2 | 0.526 | 0.499 | 241398 |
| A-CFR sampled (ours) | 2 | 0.667 | 0.667 | — |

- **特征编码的价值确认**：把到 exp≤0.5 从 onehot 的 241k 提到 **42838（5.6×
  快于 onehot、3.9× 快于最佳基线）**，min 从 0.50 降到 0.33。→ 泛化对中博弈必要。
- **但 floor 在 ~0.27–0.33（4 种子 min），且 final 漂回 0.34–0.53**；基线平均
  策略到 0.15–0.23。**高精度区我方输 2×。**

## 3. 效率与稳定性（诚实短板）

- **Wall-clock（1M Leduc 单核）**：我方 neural **6400–7470s** vs 基线
  1265–3029s（**慢 2.5–3×**）。根因=每 B=64 episodes 刷新全表 sigma/Q（15625
  次全前向），实现层低效，可优化（增量刷新/更大 B）。我方 sampled 仅 115–199s
  （表格，但 exploitability 差、仅小博弈可用）。
- **last-iterate 漂移不稳定**：features 在 **Kuhn s0 发散**（final 0.499，900k–1M
  钉死 ~0.5），s1 正常 0.008——1/2 种子崩；Leduc 4 种子 min 后均漂上（0.27→0.44）。
  与 [[neural-headlines-dont-reproduce-torch211]] 同一 Prop-4 带 + init 依赖漂移。

## 4. 根因分析（本质，理论一致）

**A-CFR 的 last-iterate 快速收敛到一个邻域（带），但带半径=Prop-4 的非零 SA 噪声
floor，且带内有 init 依赖随机游走漂移。** 平均策略方法相反：收敛慢（早期迭代污染
平均）但渐近到更低 exploitability（平均消噪）。这解释了**交叉**：
- 早期/中精度：我方快（无平均滞后）→ 样本效率胜。
- 高精度：基线低（平均把噪声磨到更小）→ 我方被带 floor 卡住。

即：**速度（last-iterate）vs 精度（平均）的根本权衡**。这不是 bug，是
last-iterate 范式的内在特性；features 改善了"代表能力/泛化"（降带、提早期速度）
但未改变 floor 的存在，且引入更多漂移方差。

## 5. 与 ICLR 目标的差距（诚实裁决）

- ✅ 真实优势：早期/中精度**样本效率**（理论根=无平均滞后），小博弈全胜、
  中博弈早期 3.9× 快；**无平均策略网络**（删 SD-CFR 称"最大误差源"）+ 小缓冲。
- ❌ 未达突出：**高精度 exploitability 输 SOTA**（Leduc 0.33 vs 0.15）；
  **wall-clock 慢 2.5–3×**；**last-iterate 漂移不稳**。
- 结论：**当前形态不足以单凭"我们更快到中等精度"达 ICLR 对 SOTA 的突出优势**，
  尤其高精度与 wall-clock 是硬伤。按用户授权 → 须改算法（§6）。

## 6. 下一步算法方向（基于根因，待验证）

1. **尾平均（tail-averaging）破带 floor**（最有前景）：A-CFR 的 last-iterate
   **已在 Nash 邻域**（不像 CFR 末迭代振荡），对**收敛后的近期策略**做短窗
   平均即可消掉带内 SA 噪声、把 floor 降一截，**同时保留早期快**——且**不需平均
   策略网络**（只维护近期策略表的 EMA，仍删 SD-CFR 的大误差源）。这是
   "last-iterate 速度 + 平均精度"的合一，直击 §4 权衡。理论上 floor 应 ↓ ~√窗口。
2. **修 wall-clock**：增量刷新（只刷访问到的 iset）/ 更大 batch 摊薄全前向；
   目标追平基线单核成本。
3. **稳住漂移**：features 引入的方差（Kuhn s0 崩）需 §1 的软锚/退火再调，或
   配合尾平均（平均天然抑制漂移）。
4. **扩展博弈**：补 LiarsDice5/6、Battleship、Goofspiel（适配层已支持），看
   优势/劣势的博弈依赖；补基线 4 种子 + CI。
5. 若尾平均仍不破高精度劣势 → 须重审理论主张（把"last-iterate 到精确 Nash"
   的神经版主张降级为"到有界邻域 + 尾平均到更低"，Prop-4 已支持）。

## 7. 数据/产物位置
- 曲线 CSV：`code/comparison/logs_ours/`、`logs_baseline/`；图 `figs/compare_*.png`。
- 运行日志：`code/comparison/logs_run/`、`campaign_phase1.log`。
- 记忆：`comparison-phase1-results`、`openspiel-integration-and-baselines`、
  `neural-headlines-dont-reproduce-torch211`、`p3f-rebound-is-estimator-variance`。
