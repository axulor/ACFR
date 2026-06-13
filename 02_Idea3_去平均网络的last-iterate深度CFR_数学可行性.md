# Idea 3 细化：去平均策略网络的 Last-Iterate 深度 CFR —— 数学可行性完整推导（v1，2026-06-10）

**工作名**：A-CFR（Anchored-regularization CFR；最终命名待定）。
**一句话主张**：在每个信息集上做带"磁锚"正则的镜像下降更新（锚 = 周期性快照的当前策略网络），外环构成单调变分不等式上的非精确近点法，使**当前策略本身** last-iterate 收敛到纳什均衡——从而彻底删除平均策略网络、蓄水池缓冲与 SD-CFR 式的 T 个网络存储。温度 τ 不需要退火到零：**锚在走，温度不动**。

---

## 0. 结论先行

可行性结论：**理论上成立，且定理链的每一环都有已发表的支柱可引用或适配；真正需要新证明的是 3 个环节（标注 ★），难度可控；有一条路线（A2L 归约）经详细分析后必须排除。** 与最近邻工作（PG 线、MMD、Reg-CFR、APMD）的可区分性明确。下文给出全部推导。

---

## 1. 形式化设定与记号

二人零和、完美回忆 EFG。玩家 $i$ 的信息集集合 $\mathcal{I}_i$，动作集 $A(I)$。**序列形式（sequence form）**：策略表示为 realization plan $x\in\mathcal{X}\subset\mathbb{R}^{\Sigma_1}$，$y\in\mathcal{Y}\subset\mathbb{R}^{\Sigma_2}$，其中 $\mathcal{X},\mathcal{Y}$ 为 treeplex（嵌套单纯形的凸多胞形），收益双线性：$u(x,y)=x^\top A y$。纳什均衡 = 鞍点 $z^*=(x^*,y^*)$。可利用度（exploitability / 对偶间隙）：

$$\mathrm{gap}(x,y) := \max_{x'\in\mathcal{X}} x'^\top Ay - \min_{y'\in\mathcal{Y}} x^\top Ay'.$$

博弈算子 $F(z)=(A y,\,-A^\top x)$。零和双线性 ⇒ $F$ **单调**（$\langle F(z)-F(z'),z-z'\rangle\ge 0$）且 $L$-Lipschitz，$L=\|A\|_{\mathrm{op}}$。这是后文一切收敛论证的结构基础。

**Dilated（膨胀）Bregman 散度**：对 treeplex 上的 $x,x'$，

$$B_{\Psi}(x\,\|\,x') := \sum_{I\in\mathcal{I}_1} x_{p(I)}\;\mathrm{KL}\big(x(\cdot|I)\,\big\|\,x'(\cdot|I)\big),$$

其中 $x_{p(I)}$ 为到达 $I$ 的己方序列概率，$x(\cdot|I)$ 为行为策略。dilated entropy 是 treeplex 上的强凸正则子，强凸模 $c_Q>0$ 有显式常数（Kroer et al. 2020；与树深、分支因子有关）。**反事实值**：$q^{\sigma}(I,a)$ 为对手/机会 reach 加权的动作值，恰是 $\nabla_x u$ 在序列形式下对应坐标除以己方 reach 的局部化——这是 CFR 分解与 dilated 几何天然匹配的原因（见 §4.2）。

---

## 2. 负结果汇编：为什么"分析现有 CFR 的最后迭代"是死路

任何此方向的论文都需要这一节来论证"必须改动力学"：

**N1（RM 家族）**：simultaneous/alternating RM+、predictive RM+ 在一个 3×3 矩阵博弈上就没有 last-iterate 收敛（持续循环）；ExRM+ 与 Smooth PRM+ 有渐近 last-iterate、$O(1/\sqrt{t})$ best-iterate，加 restarting 后线性 last-iterate（[Cai, Farina, Grand-Clément, Kroer, Lee, Luo, Zheng — ICLR 2025, arXiv:2311.00676](https://arxiv.org/abs/2311.00676)）。技术障碍被明确指出：RM 类算子**不 Lipschitz 也不（伪）单调**——这宣判了"直接证 CFR/RM last-iterate"的死刑。

**N2（FoReL/MWU）**：零和博弈中 FoReL/MWU 的迭代呈 Poincaré 回归（循环不收敛，Mertikopoulos et al. 2018）。

**N3（乐观也不够）**：OMWU 虽有渐近 last-iterate，其速率可以任意慢；快速 last-iterate **需要"遗忘性"**（[Cai et al., NeurIPS 2024](https://arxiv.org/abs/2406.10631)：Fast Last-Iterate Convergence Requires Forgetful Algorithms）。设计指导：锚定/重启/折扣这类遗忘机制不是可选项，是必需品。

**N4（平均化的深度代价）**：平均策略网络的逼近误差是 DeepCFR 的主要误差源（SD-CFR 的论证）；蓄水池缓冲拟合的是非平稳混合目标；SD-CFR 的替代方案要存 $T$ 个网络。**当前全部深度 CFR 脉络（含 AAAI 2026 SOTA）都背着这个包袱。**

结论：必须注入{正则化、锚定、外推}之一。三条候选路线裁决如下。

---

## 3. 三条理论路线的裁决

### 3.1 路线 R3：A2L 归约 —— 经分析后**排除**

[Cai, Luo, Wei, Zheng 2025（arXiv:2506.03464）](https://arxiv.org/abs/2506.03464)的归约机制（已精读全文）：**玩运行平均 $\bar x^t$、利用收益线性性反解出"虚拟迭代"的反馈** $u^t = t\bar u^t-(t-1)\bar u^{t-1}$，喂给内部基算法。优点：黑盒、状态 $O(1)$（无需存历史策略）、全反馈下率漂亮（$O(\log d/T)$ last-iterate）。但对我们的目标有两个**结构性障碍**：

1. **播放的策略就是平均策略**。归约没有消灭平均——它把平均变成了你玩的东西（作者原话："inherently requires computing the running average of iterates"）。神经实现中这个被播放的 $\bar x^t$ 必须被一个网络表示，且其更新 $\bar x^t=(1-\frac1t)\bar x^{t-1}+\frac1t x^t$ 是**策略空间**的平均（非参数空间）——这等于换汤不换药地重新引入了平均策略网络。
2. **$t$ 倍噪声放大**。反解步骤把第 $t$ 轮反馈乘以 $t$ 再差分；任何采样/函数逼近误差 $\Delta^t$ 以 $t\Delta^t$ 进入遗憾界（其 Lemma 3）。论文自己的 bandit 版需要每轮批量 $B_t\sim t^4$ 增长 + 相位内冻结策略才能压住，率退化到 $\widetilde O(T^{-1/5})$。**对常数批量、带逼近误差的 model-free 深度算法，该归约无任何保证，且机制上自败。**

判决：作为相关工作引用并解释排除理由（这本身是论文里有价值的一段论证），不作为算法基础。

### 3.2 路线 R2：ExRM+/SPRM+ 作为 CFR 局部最小化器 —— 降级为"理论加分项"

N1 的正面结果只覆盖矩阵博弈。把 ExRM+ 塞进 CFR 的每个信息集后，**局部 last-iterate ⇏ 全局 last-iterate**：信息集 $I$ 看到的反事实值依赖所有其他信息集的当前策略，局部问题是非平稳耦合的；目前没有任何工具能把 2311.00676 的极限点几何分析提升到 treeplex 上（其证明严重依赖单纯形结构）。这是一个真正的开放问题——如果在做主线时顺手解决，可单独成文；不作为依赖项。

### 3.3 路线 R1（主线）：锚定正则化反事实动力学

四块已发表的支柱恰好拼出完整路径，每块只差"组合 + EFG/采样/神经实例化"：

| 支柱 | 已证内容 | 我们要补的 |
|---|---|---|
| [MMD（Sokota et al., ICLR 2023）](https://arxiv.org/abs/2206.05825) | 强单调正则化 VI 上磁镜像下降**线性** last-iterate 到 QRE；NFG+EFG(dilated) 理论 | 锚不固定而是移动；model-free 采样版 |
| [Reg-CFR / Reg-DOMD（Liu et al., ICLR 2023）](https://arxiv.org/abs/2206.09495) | **首个 CFR 型 last-iterate**：扰动 EFG 上 last-iterate $O(1/T)$（对偶间隙）；原 EFG 上 best-iterate $O(T^{-1/4})$ | 全反馈、表格、固定扰动中心；我们换成移动锚 + 采样 |
| [APMD（Abe et al., ICML 2024）](https://arxiv.org/abs/2305.16610)及其 [GABP（ICLR 2025）](https://arxiv.org/abs/2410.02388) | 收益扰动 + **锚周期性更新为当前迭代** ⇒ 无需正则强度退火即收敛到**精确** Nash（单调博弈，GABP 含噪声版更快率） | NFG/单调博弈层面；无 EFG 反事实分解、无深度版 |
| [PG 线（ICLR 2025, arXiv:2408.00751](https://arxiv.org/abs/2408.00751)；[NeurIPS 2025 全局收敛](https://www.mit.edu/~gfarina/2025/neurips25_policy_gradient/)) | 自博弈策略梯度 best-iterate 收敛到正则化 NE | 他们**放弃了反事实值**（用 Q 值）；只有 best-iterate |

**我们的组合即贡献**：反事实值估计（CFR 脉络的效率来源，AAAI 2026 消融已证明估计器质量决定收敛速度）× 锚定正则化（last-iterate 的来源）× bootstrap 蒸馏（单网络、无缓冲的来源）。每个相邻工作都恰好缺其中两样。

---

## 4. 主线定理链：完整推导

记 $\Psi(z)=\Psi_1(x)+\Psi_2(y)$ 为 dilated entropy，锚 $\mu=(\mu_x,\mu_y)$，温度 $\tau>0$ 固定。**锚定正则化博弈**：

$$f_\tau^\mu(x,y) := x^\top Ay - \tau B_{\Psi_1}(x\|\mu_x) + \tau B_{\Psi_2}(y\|\mu_y).$$

其鞍点记 $z^*_\tau(\mu)$，对应 VI 算子 $F_\tau^\mu(z) = F(z) + \tau(\nabla\Psi(z)-\nabla\Psi(\mu))$。

### Lemma 1（强单调性）【引用级，无风险】

dilated entropy 在 treeplex 上对适当范数强凸，模 $c_Q>0$ 显式（Kroer et al. 2020）。Bregman 项的梯度差 $\nabla\Psi(z)-\nabla\Psi(\mu)$ 关于 $z$ 是 $\Psi$ 的梯度平移，故 $F_\tau^\mu$ 是 $\tau c_Q$-**强单调**算子；$z^*_\tau(\mu)$ 存在唯一。

### Lemma 2（锚界：正则化解离纳什多远）【三行新证，已验证】

对任意 $x'$：由 $z^*_\tau$ 在正则化博弈中的最优性，
$$x'^\top Ay^*_\tau - \tau B(x'\|\mu) \le x^{*\top}_\tau Ay^*_\tau - \tau B(x^*_\tau\|\mu) \le x^{*\top}_\tau Ay^*_\tau,$$
取 $\max_{x'}$ 并对 $y$ 侧对称操作，相加得

$$\boxed{\;\mathrm{gap}(z^*_\tau(\mu)) \;\le\; 2\tau\,\max_{z\in\mathcal{Z}} B_\Psi(z\|\mu)\;=:\;2\tau D_\Psi(\mu).\;}$$

对 uniform 锚，$D_\Psi \le \sum_{I}\log|A(I)|$ 量级（reach 加权后通常远小于此）。**关键观察：当锚 $\mu$ 本身趋近 $z^*$ 时 $B_\Psi(z^*\|\mu)\to 0$，gap 上界自动消失——这就是"锚在走、温度不动"能到达精确 Nash 的数学原因。**

### Theorem 1（内环：全反馈线性收敛）【适配级，低风险】

固定锚 $\mu$。锚定镜像下降
$$z^{t+1} = \arg\min_{z}\;\eta\big\langle F(z^t),z\big\rangle + \eta\tau B_\Psi(z\|\mu) + B_\Psi(z\|z^t)$$
在 $\eta \le \tau c_Q/L^2$ 时以收缩率 $\rho=(1+\eta\tau c_Q)^{-1}$ 线性收敛：$B_\Psi(z^*_\tau\|z^t)\le \rho^t B_\Psi(z^*_\tau\|z^0)$。

*证明骨架*：强单调 + 相对平滑下 MMD 的标准三点引理推收缩（Sokota et al. Thm 3.4 的 treeplex 实例化；dilated 情形的相对平滑常数由 $L$ 与 $c_Q$ 给出）。可替代地引 Reg-DOMD 的 $O(1/T)$ last-iterate（Liu et al.）作为弱化版本——双保险。

**行为形式闭式更新（反事实分解的体现）**：上述 prox 在 dilated 几何下逐信息集分解。对每个 $I$（自底向上带 value-to-go 修正，或采用 Reg-CFR 式局部化），更新有闭式

$$\sigma^{t+1}(a|I)\;\propto\;\big[\sigma^t(a|I)\big]^{\frac{1}{1+\eta\tau}}\big[\mu(a|I)\big]^{\frac{\eta\tau}{1+\eta\tau}}\exp\!\Big(\frac{\eta\, \hat q^t(I,a)}{1+\eta\tau}\Big),$$

其中 $\hat q^t(I,a)$ 为**反事实动作值**（对手 reach 加权）。为什么必须是反事实值而不是 Q 值：序列形式梯度 $\nabla_x u = Ay$ 的第 $(I,a)$ 坐标恰为对手 reach 加权值；用 Q 值（己方 reach 也加权）会让更新依赖己方 reach，破坏与序列形式 VI 的对应（PG 线为绕开此问题付出了只有 best-iterate 的代价）。**这一段是论文的"可解释核心"：CFR 的反事实分解 = dilated 几何下镜像更新的精确局部化。**

### Theorem 2（外环：移动锚 = 非精确近点法 ⇒ 精确 Nash，无退火）【★ 新证明 1，中风险】

锚更新规则：内环跑到 $B_\Psi(z^*_\tau(\mu_k)\|z)\le \varepsilon_k$ 后置 $\mu_{k+1}\leftarrow z$。则外环序列是单调 VI $F$ 上步长 $1/\tau$ 的**非精确近点法**（PPM）：$\mu_{k+1}\approx \mathrm{prox}_{F/\tau}^{\Psi}(\mu_k)$。

*可证内容*：(i) 精确 PPM 对单调 VI 的 last-iterate 收敛是经典结果（Rockafellar 1976；Bregman 版 Eckstein 1993）；(ii) 非精确版在 $\sum_k\sqrt{\varepsilon_k}<\infty$（取 $\varepsilon_k = \varepsilon_0 4^{-k}$，由 Thm 1 线性收敛只需 $K_k = O(k)$ 步内环）下保持收敛；(iii) 对偶间隙率：结合 Lemma 2，$\mathrm{gap}(\mu_k)\le 2\tau B_\Psi(z^*_{k}\|\mu_{k-1})$，PPM 的标准能量不等式给出 $\min_{j\le k}\mathrm{gap}(\mu_j) = O(\tau B_\Psi(z^*\|\mu_0)/k)$，加 Halpern/锚定平均技巧可望升级为 last-iterate 率。

**【2026-06-10 升级，Phase 0b 实验驱动】**：B3 实验显示周期锚（K 固定）下整体 NashConv 呈跨 10 个数量级的**线性**衰减（直至机器精度）。严格化路径：双线性 EFG 的 VI 是**多面体**的 ⇒ 满足误差界条件（metric subregularity）⇒ 非精确 PPM 在几何内环精度调度下**线性收敛**；固定 K 的周期锚恰好自动提供几何 ε_k（每相位内环收缩 ρ^K）。同一工具已被用于 restarted ExRM+ 的线性 last-iterate（[arXiv:2311.00676](https://arxiv.org/abs/2311.00676) 正面结果部分）与 LP 的 restarted PDHG（Applegate et al.）。**主定理目标据此升级为：A-CFR（周期锚）以线性速率 last-iterate 收敛到精确纳什均衡。**
*风险点与靠山*：APMD（ICML 2024）已在单调博弈中证明同结构的"锚更新 ⇒ 精确 Nash"以及其 ICLR 2025 续作的率；我们要做的是 Bregman/treeplex 版本。难度：中。**Fallback**：若 last-iterate 率卡住，退守"$\mathrm{gap}(\mu_k)\to 0$ 渐近 + best-iterate $O(1/k)$"——仍然严格强于现有全部深度 CFR（零 last-iterate 理论）。

### Theorem 3（采样版：model-free 样本复杂度）【★ 新证明 2，中风险】

内环改用反事实值的无偏估计 $\hat q^t$（**与 Idea 1 在此汇合**：λ-估计器/DREAM 基线给出方差 $\sigma^2$ 的显式上界；λ-估计器的截断版给出与深度无关的 $\sigma^2$ 硬上界）。强单调随机 VI 的标准结果（Robbins-Monro 型，步长 $\eta_t=\Theta(\tfrac{1}{\tau c_Q t})$）：

$$\mathbb{E}\,B_\Psi(z^*_\tau\|z^t) \;=\; O\!\Big(\frac{\sigma^2}{\tau^2 c_Q^2\, t}\Big).$$

端到端：目标可利用度 $\varepsilon$ ⇒ 由 Lemma 2 取 $\tau=\varepsilon/(4D_\Psi)$（一次性分析；移动锚版更优）⇒ 样本复杂度

$$\widetilde O\!\Big(\frac{\sigma^2 D_\Psi^2 L_u^2}{\varepsilon^4}\Big),$$

其中 $L_u$ 为 gap 对策略距离的 Lipschitz 常数。**对比**：ICML 2025 bandit 反馈 last-iterate 论文（[iOAcVOHvEN](https://openreview.net/forum?id=iOAcVOHvEN)）只有 $\widetilde O(k^{-1/8})$（即 $\varepsilon^{-8}$）且无实用算法；我们的反馈模型更强（轨迹反馈 + 学习的值函数），换得 $\varepsilon^{-4}$ 与可实现性——定位诚实，不冲突。
*风险点*：锚相位间的统计相关性。解法：相位内冻结锚与值网络（AAAI 2026 对基线网络用同一技巧恢复条件独立）。

### Proposition 4（神经层：误差不随时间累积）【★ 新证明 3，这是工程卖点的理论化】

神经版用单策略网络 $\pi_\theta$ 蒸馏闭式目标：$\theta^{t+1}=\arg\min_\theta\;\mathbb{E}_{I\sim\text{当轮样本}}\,\mathrm{KL}\big(\sigma^{t+1}_{\text{closed-form}}(\cdot|I)\,\|\,\pi_\theta(\cdot|I)\big)$，每步蒸馏误差 $\le\delta$（KL 意义）。则误差进入收缩动力学：

$$B_\Psi(z^*_\tau\|z^{t+1}) \le \rho\, B_\Psi(z^*_\tau\|z^t) + C\delta \;\;\Rightarrow\;\; \limsup_t B_\Psi(z^*_\tau\|z^t) \le \frac{C\delta}{1-\rho} = O\!\Big(\frac{\delta}{\eta\tau c_Q}\Big).$$

**几何级数封顶，与 $T$ 无关。** 对比 DeepCFR 范式：平均策略由全部 $T$ 轮的优势网络误差线性叠加（DeepCFR Thm 的 $O(\varepsilon)$ 项是对每轮误差求平均，且平均网络再额外引入 $\varepsilon_{\text{avg}}$）。**收缩 + 注入误差 ⇒ uniform-in-time 稳定性**，这是删掉平均网络后反而获得的更好误差传播性质，是论文的核心卖点定理。

### 算法骨架（神经版，对照表见 §6）

```
组件：策略网络 π_θ；反事实值网络 Q_w（λ-估计器用）；锚 π_θ̄（θ 的周期快照/EMA，零训练成本）
每轮 t：
  1. 自博弈采样 K 条轨迹（ε-混合采样策略）
  2. λ-估计器算各访问信息集的反事实优势 q̂(I,·)（Q_w 做控制变量/纯值端点）
  3. 闭式目标 σ' ∝ π_θ^{1/(1+ητ)} · π_θ̄^{ητ/(1+ητ)} · exp(η q̂/(1+ητ))
  4. 当轮样本上蒸馏：θ ← θ - ∇ KL(σ'||π_θ)；Q_w 做 expected-SARSA 更新（同 AAAI 2026）
  5. 每 K 轮：θ̄ ← θ（锚前移）
输出：π_θ 本身（last iterate）。无平均网络、无蓄水池缓冲、缓冲只存当轮。
```

---

## 5. 主要数学障碍清单（诚实版）

**O1（dilated prox 的 reach 权重）**：精确的 dilated prox 逐信息集更新带 value-to-go 项且温度被己方 reach 调制（reach→0 处局部温度爆炸）。解法 a：用 Kroer 式 reach-加权（"balanced"）dilated entropy 吸收权重；解法 b：Reg-CFR 式局部化（每个信息集独立 OMD + 反事实值），其分析框架已发表。难度：低-中（有成熟先例）。

**O2（移动锚 × 采样的耦合）**：锚是历史数据的函数，破坏内环估计的条件无偏。解法：相位冻结（见 Thm 3 风险点）。难度：低。

**O3（Thm 2 的 last-iterate 率**）：非精确 Bregman-PPM 在 treeplex 上的率可能只到 best-iterate。Fallback 已述；另一条升级路径是 Halpern 锚定（与 ExRM+ 文献的 restarting 同源）。难度：中-高（这是三个新证明里最难的）。

**O4（神经容量）**：闭式目标是三个分布的几何混合，表示能力要求不高于现有方法的优势网络；蒸馏目标在单纯形上显式、有界、量纲友好（对比反事实遗憾的数量级问题——AAAI 2026 为此大费周章，我们的目标天然归一化）。难度：低。**这其实是又一个卖点：拟合目标从"无界累积遗憾"变成"有界概率分布"。**

**O5（τ 与 K 的调度）**：固定 τ + 锚步频 K 两个超参，对比 DCFR 的 (α,β,γ) + 退火调度并不更多；且 Lemma 2 给了 τ 的可解释选择公式。

---

## 6. 与最近邻工作的差异定位（审稿人视角）

| 工作 | 反事实值 | last-iterate | model-free 采样 | 无平均网络 | 神经实现 |
|---|---|---|---|---|---|
| DeepCFR/SD-CFR/DREAM/ESCHER | ✓ | ✗ | 部分✓ | ✗（SD-CFR 存 T 网络） | ✓ |
| VR-Deep(P)DCFR+（AAAI 2026 SOTA） | ✓ | ✗ | ✓ | ✗ | ✓ |
| MMD（ICLR 2023） | ✗ | ✓（固定锚→QRE） | 部分（无理论） | ✓ | 启发式 |
| Reg-CFR（ICLR 2023） | ✓ | ✓（扰动博弈） | ✗ | ✓（表格） | ✗ |
| APMD/Boosting（ICML24/ICLR25） | ✗ | ✓（NFG/单调博弈） | 部分 | ✓ | ✗ |
| PG 线（ICLR25/NeurIPS25） | ✗（Q 值） | best-iterate | ✓ | ✓ | ✓ |
| ICML25 bandit EFG | ✓ | ✓（$k^{-1/8}$） | ✓ | ✓ | ✗ |
| **A-CFR（本工作）** | **✓** | **✓（移动锚→精确 Nash）** | **✓（含方差界）** | **✓** | **✓（误差不累积）** |

故事线：「CFR 脉络给了最好的估计器，正则化脉络给了 last-iterate，但十五年来没人把两者在 model-free 深度设定下接通——接通后平均策略网络（DeepCFR 以来的最大误差源与内存包袱）整个消失。」

---

## 7. 验证路线（4060 约束）

**Phase 0（表格，CPU，1–2 周，先证伪核心假设）**：Kuhn/Leduc 上实现表格 A-CFR：(a) 固定锚 → 验证线性收敛到 QRE（对照 Thm 1 的收缩率预测）；(b) 移动锚 → 验证 gap→0 无退火（对照 APMD 行为）；(c) 锚步频 K、τ 扫描；(d) 对照组 CFR+、MMD（退火版）、Reg-CFR。判据：last-iterate 可利用度曲线单调下降且无 CFR 的振荡。
**Phase 1（采样表格）**：OS 采样 + λ-估计器注入，验证 Thm 3 的 $O(\sigma^2/\tau^2 t)$ 标度律（log-log 斜率 −1）。
**Phase 2（神经，4060 够用）**：3×64 MLP，Kuhn/Leduc/Battleship(2)/Liar's Dice(5)，对照 OS-DeepCFR、DREAM、（复现的）VR-DeepDCFR+；额外汇报**内存占用与缓冲大小**（我们的结构性优势项）。
**Phase 3（投稿规模，届时借算力）**：FHP head-to-head + Rudolph et al. 套件对 MMD/PPO。

---

## 8. 撞车监控与时间敏感性

最危险的三个邻居及其与本工作的"一步之遥"：PG 线（NeurIPS 2025 已做到全局收敛，若他们下一步加反事实值结构就撞）；Abe 组（APMD 作者，若下一步做 EFG 实例化就撞）；Sokota/McAleer 系（MMD 作者，若做 MMD 的 model-free EFG 理论版就撞）。三组都活跃，**建议 Phase 0 立即启动、3 个月内出 workshop 版占位**。每月扫一次 arXiv cs.GT 关键词：anchored/magnetic + extensive-form、last-iterate + CFR、proximal point + games。

## 9. 参考文献

[MMD](https://arxiv.org/abs/2206.05825) · [Reg-CFR](https://arxiv.org/abs/2206.09495) · [RM+ last-iterate 负结果](https://arxiv.org/abs/2311.00676) · [APMD](https://arxiv.org/abs/2305.16610) · [遗忘性必要](https://arxiv.org/abs/2406.10631) · [A2L 归约](https://arxiv.org/abs/2506.03464) · [PG best-iterate](https://arxiv.org/abs/2408.00751) · [PG 全局收敛 NeurIPS25](https://www.mit.edu/~gfarina/2025/neurips25_policy_gradient/) · [ICML25 bandit EFG](https://openreview.net/forum?id=iOAcVOHvEN) · [R-NaD/DeepNash](https://arxiv.org/abs/2206.15378) · [Perolat et al. 2021 正则化](https://arxiv.org/abs/2002.08456) · [VR-Deep(P)DCFR+](https://arxiv.org/abs/2511.08174) · [Kroer et al. dilated entropy 强凸常数](https://arxiv.org/abs/1702.04849) · [Farina-Kroer-Sandholm dilated DGF 局部分解](https://arxiv.org/abs/1910.10906)（后者证明了 dilated 几何下镜像下降精确分解为逐信息集局部更新——§4.2 论断的直接文献支柱）
