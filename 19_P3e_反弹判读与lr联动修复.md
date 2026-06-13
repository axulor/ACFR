# P3e/P4a-A0 判读：400k 反弹 → 最后一个底是 SGD 噪声（2026-06-12）

## 1. 结果

**P3e（Kuhn 400k，最终配方 = 退火+q_both，vs tabular）**

| 臂 | min | final(400k) | 判读 |
|---|---|---|---|
| tabular | — | **0.0181** | 阶梯下降继续，目标线 |
| neural seed0 | 0.0622@~200k | 0.2818 | **反弹** |
| neural seed1 | 0.0552@~200k | 0.4435 | **反弹** |

**P4a-A0（Kuhn 200k，features vs onehot，最终配方）**

| 臂 | min | final(200k) |
|---|---|---|
| onehot s0 | 0.0622 | 0.1621（=P3d D3 同配置复算 ✓） |
| onehot s1 | 0.0552 | 0.2570 |
| features s0 | **0.0184** | 0.4338（反弹更陡） |
| features s1 | 运行中 | — |

两个事实：(a) **所有神经臂在 ~150–200k 处达到优秀的 min 后反弹**——q_both 把崩溃从 70k（P3d none_const）推迟到 ~200k，但没有根除；(b) **特征编码的 min 比 one-hot 好 3 倍**（0.018，已到 tabular@400k 的量级）——泛化编码平滑了优化面，P4 的前提利好；但反弹也更陡。

## 2. 诊断：幸存的底是策略侧每更新 SGD 噪声，∝ 1/η

时间线对齐：反弹发生在第 5–6 次超相位（η≤0.0156，K≥64k）之后。机制：

- 蒸馏目标的**信号**部分（q 项）∝ η 缩小；
- 但每次蒸馏的 **SGD 干扰噪声**不随 η 缩小（P3c-C0 实测 interference ≈ 0.01–0.02/更新，与 ds 无关、不随 η 变）；
- 锚回拉强度 ∝ ητ ⇒ 策略随机游走的稳态方差 ∝ σ_w²/(ητ)——**η 越小底越高**；
- 这就是 Prop-4 的 δ 的真身：**不是拟合偏差（已测 ≈1e-4~1e-7，可忽略），而是每更新参数噪声（方差项）**。P3b 把 matched（×2 蒸馏步）设计为压偏差——方向就错了，步数越多某种意义上噪声注入越多（C0 中 interference 随 ds 升）。

修订后的神经底三项分解（最终版）：
$$\text{floor}(t)\;\asymp\;\max\Big(\underbrace{\tfrac{\eta\sigma^2(t)}{B\tau}}_{\text{估计方差（Q 漂移污染，q\_both 已治}},\;\underbrace{\tfrac{\sigma_w^2(\text{lr})}{\eta\tau}}_{\text{SGD 参数噪声（本轮）}},\;\underbrace{\tfrac{C\,\delta_{\text{bias}}}{\eta\tau}}_{\text{拟合偏差（实测可忽略）}}\Big)$$

## 3. 修复（P3f，运行中）：lr 与 η 联动退火

σ_w² ∝ lr²（Adam 噪声尺度）⇒ 每次 η 减半时 lr 同步减半 ⇒ SGD 底 ∝ lr²/η ∝ η 持续下降，与噪声底同步。跟踪能力不受损：目标每更新移动 O(η)，lr∝η 保持相对步长不变。实现：`lr_couple=True`（η 衰减时双优化器 param_groups lr ×= eta_decay，保留 Adam 状态）。

**P3f 臂**：lr-coupled × {onehot, features} × 2 种子，Kuhn 400k，与 P3e/A0 未联动曲线同图。判据：末四分位均值 ≤ min×1.5（无反弹）且 final ≤ 2× tabular(0.018)。

若通过 ⇒ 神经配方最终形式 = **超相位退火（η、K、lr 三联动）+ Q-anchor + Q-replay**，调度推论改写为"三联动"版本；启动 Leduc 神经臂（A1）。若仍反弹 ⇒ 剩余候选：蒸馏 deadband（目标变化小于阈值时跳过训练）、相位末重置 Adam 二阶矩；以及诚实降级选项——报告 min-iterate（带在线 NashConv 监控的停止规则在采样设定下不可用，需用代理信号如锚间距离）。

## 4. 方法论沉淀（写进论文 appendix 的 lessons）

1. 神经误差必须分解为 **偏差（可测可压）/ 估计方差（Q 漂移）/ 参数噪声（lr）** 三项，各有不同的 η 标度——单一 δ 模型三次误导了修复方向（P3b matched、P3c η*、P3d frozen）。
2. floor 测量必须带漂移检测与 ≥2× 视界余量（两次翻车：P1c-C1、P3c-C1）。
3. 每个"慢变量"都要锚定：策略锚（理论）、Q 锚（target net）、lr 锚（调度联动）——统一原则在算法上的第三次实例化。
