# A-CFR：算法介绍、SOTA 对比与实验计划（2026-06-12）

> 本文档回答四个问题：我们提出了什么算法？比 SOTA 优势何在？对比实验怎么做？
> 当前在跑什么实验？（面向项目回顾与论文 Intro 的素材沉淀）

## 1. 我们提出了什么算法：A-CFR（Anchored-regularization CFR）

**问题背景**。整条深度 CFR 脉络——DeepCFR (NeurIPS'19) → SD-CFR → DREAM →
ESCHER → VR-DeepPDCFR+（AAAI 2026，当前 SOTA）——共享一个结构性缺陷：
**当前策略本身不收敛，必须额外维护"平均策略"** 才能逼近纳什均衡。代价：

- 平均策略网络是最大单一误差源（SD-CFR 的论证）；
- 需要蓄水池缓冲（reservoir buffer）或存储全部历史网络；
- 整条脉络**没有任何 last-iterate 收敛理论**。

**核心 idea**。把 CFR 的反事实分解（这条脉络效率的来源）与正则化博弈动力学
的锚定技术（last-iterate 的来源）接通。逐信息集的三因子乘性更新：

$$\sigma^{t+1}(a|I)\;\propto\;\underbrace{\sigma^t(a|I)^{\frac{1}{1+\eta\tau}}}_{\text{当前策略惯性}}\;\cdot\;\underbrace{\mu(a|I)^{\frac{\eta\tau}{1+\eta\tau}}}_{\text{向锚回拉}}\;\cdot\;\underbrace{\exp\Big(\tfrac{\eta\,\tilde q^t(I,a)}{1+\eta\tau}\Big)}_{\text{归一化反事实优势}}$$

- 锚 μ = 每 K 步对**自身策略**的快照；温度 τ 固定**不退火**（锚的移动替代退火）；
- 外环 = 单调 VI 上的非精确近点法（PPM）；
- 采样版用 **λ-估计器**：DREAM 无偏端点（λ=1）↔ ESCHER 零方差端点（λ=0）
  的连续插值，配小批量平均与超相位调度（η×½、K×2）；
- 神经版（P3 系列定稿）：**三联动调度（η/K/lr 同步）+ Q-anchor + Q-replay**，
  统一原则"**锚定一切慢变量**"——策略锚（理论核心）、Q 锚（target network
  的博弈论版本）、lr 锚（SGD 噪声控制）。

**理论链**（详见 11/12/15 号文档）：
Thm 1 内环线性收缩（速率不含强凸模 c，实测 2–5% 内命中）→
Thm 2 移动锚到**精确** Nash、整体线性率（Luque 形式 ρ(τ)，Kuhn 实测跨 13
个数量级直线至 1e-12 机器精度）→ Thm 3 采样版噪声底闭式 →
Prop 4 神经误差 uniform-in-time 封顶（误差几何级数封顶，不随轮数累积）。

## 2. 比起当前 SOTA 的优势

| 维度 | DeepCFR 脉络（含 VR-DeepPDCFR+） | A-CFR |
|---|---|---|
| 收敛对象 | 仅平均策略收敛，当前策略发散 | **当前策略 last-iterate 线性收敛到精确 Nash** |
| last-iterate 理论 | 无 | 有；每层误差闭式刻画且被实验逐项命中 |
| 系统组件 | 平均策略网络 + 蓄水池缓冲 + 历史网络存储 | **全部删除**（可量化的内存优势） |
| 误差累积 | 对 T 轮逼近误差求和 + 平均网络误差 | 几何级数封顶（uniform-in-time） |
| 采样噪声鲁棒性 | 隐式、不可控 | 可证；实测 vs CFR+ 剪刀差 **83×**（σ=0.05）/ **39×**（σ=0.2） |
| 神经误差可控性 | 完全隐式、无仪表 | 三分解（拟合偏差/Q 漂移/SGD 噪声），逐项可测、各配修复 |

**诚实边界（同样写进论文）**：
1. 以上验证目前限于 Kuhn / Leduc /（刚移植的）Liar's Dice；
2. 正面交手过的只有 OS-MCCFR 与 CFR+，**尚未与 VR-DeepPDCFR+/MMD/PPO 对局**——这是 P4 的任务；
3. Leduc 采样版绝对速度此前慢于 OS-MCCFR 平均策略（1.27 vs 0.27 @1M），
   赌注在神经泛化（正在跑的 A1 实验就是验证这一点）；
4. 我们的差异化定位不是"调参刷榜"，而是**理论-实验咬合密度**：四次失败模式
   全部被理论预言、定位并修复（效用尺度→锚频→Q 漂移→SGD 噪声）。

## 3. 对比实验计划（基准 = OpenSpiel，主战场 = Ubuntu）

按用户约束修订：**Windows 本机仅做初步验证（不装 WSL2）；项目经 GitHub 同步
到 Ubuntu 机器后做正式对比**（OpenSpiel 官方包仅支持 Linux/macOS，Ubuntu 上
`pip install open_spiel` 即用，CUDA 无虚拟化坑）。

**博弈阶梯**（小→大）：
1. Leduc poker（与现有结果对接，精确 NashConv）；
2. Liar's Dice(1,1) sides=6 —— 已原生移植进我们引擎（24,576 信息集，与
   OpenSpiel 拓扑一致），精确 NashConv 评估 0.95s/次；
3. Goofspiel、Battleship、Dark Hex（OpenSpiel 现成，大博弈档用近似 BR 或
   head-to-head 胜率）。

**基线**：OS-MCCFR（已有）、NFSP、DREAM、ESCHER、MMD（Sokota et al.）、
VR-DeepPDCFR+（复现 AAAI'26 公开代码）、PPO（RL 对照）。

**指标与口径**：
- 主指标：NashConv vs **采样局数**（全算法 model-free 同预算，公平口径）；
- 我们报 last-iterate，同时报对手的 avg-iterate（双口径，避免口径争议）；
- 系统指标：峰值内存（我们删平均网络与缓冲——可量化优势）、wall-clock；
- 大博弈：DQN/IS-MCTS 近似 best response + 交叉对战矩阵。

**迁移设计**：博弈引擎与算法已解耦（games.py 的树接口 ↔ 算法只见
infoset/advantage）。Ubuntu 侧只需写一个 OpenSpiel state→infoset 适配层，
算法代码零改动。

## 4. 当前在跑的实验（2026-06-12 深夜）

| 实验 | 配置 | 验证什么 | 判据 |
|---|---|---|---|
| P4a-A1 onehot | Leduc 神经版，one-hot 编码 + 完整配方（三联动+Q双稳定器），400k | 神经闭环在 288 信息集/3780 历史上成立否 | 接近/优于表格采样版 |
| P4a-A1 features | 同上，但用 **64/68 维真特征编码** | **P4 头号问题**：参数共享能否破解表格版数据饥饿 | 显著 < 基线 **1.394**；理想接近 0.27（OS-MCCFR avg 量级） |

刚完成的铺垫：Liar's Dice 移植（sides=4/5/6：1024/5120/24576 信息集），
表格版冒烟 20k→0.49 正常；GitHub 迁移件（README/.gitignore/requirements）
就绪，待指令即可 init+commit。

## 5. 阶段位置一览（项目弧线）

- **P0–P1**（表格+采样）：15 项理论预测 13 项定量命中（10 号文档全景表）；
- **P2**（理论）：Thm 1/Thm 3/Prop 4 证完；**Thm 2(b) 全文是收尾主战场**；
- **P3 系列**（神经 bridge，b–f 六阶段）：已收口。神经误差三分解 + 三锚修复，
  Kuhn 上神经 min 0.0088 **首次低于** tabular 0.0181（16–20 号文档）；
- **P4a**（特征编码）：Kuhn 验证过（min 好 3–5×），Leduc 运行中；
- **P4b**（下一步）：Liar's Dice → GitHub 同步 → Ubuntu 上 OpenSpiel 正式对比；
- **论文**：LaTeX 骨架与核心推论已同步至最新结论（paper/main.tex）。
