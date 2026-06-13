# HANDOFF — 接手指南（给 Ubuntu 上接续的智能体）

本仓库是 **A-CFR**（Anchored-regularization CFR）研究项目的完整快照。
此前在一台 Windows 机器上做初步验证（conda env `InforMARL`，Python 3.8 +
torch 1.12+cu113，RTX 4060），现同步到 GitHub，由你在 Ubuntu 服务器上接续。
**Python 环境已不同**——见下方"环境"一节，先把环境弄对再跑任何东西。

---

## 0. 接手第一步（按顺序做）

1. **装记忆**：仓库根目录 `memory/` 是上一阶段沉淀的持久记忆（项目全景、
   用户偏好、环境约定）。把这些 `.md` 复制到你自己的 Claude 记忆目录
   （形如 `~/.claude/projects/<本项目的-slug>/memory/`），其中 `MEMORY.md`
   是索引、每轮加载。**至少先读一遍 `memory/` 全部 4 个文件。**
2. **读文档建立全局**（中文研究日志，按编号）：
   - `02` 数学可行性与相关工作定位（idea 的理论根）
   - `10` idea 与验证全景（15 项预测 vs 实验的总表）
   - `21` 算法介绍 + SOTA 对比 + 实验计划（最适合快速理解"我们在做什么"）
   - `23` **最新收口**：三博弈记分牌、最终配方、开放项清单 ← 从这里看现状
   - 理论细节：`11`（定理骨架）、`12`/`15`（证明全文两批）
   - 神经版完整弧线（失败-修复闭环）：`16`→`17`→`18`→`19`→`20`→`22`
3. **核对环境后**跑 `code/run_phase0.py --quick` 自检（含断言），通过即可上手。

---

## 1. 一句话定位

整条深度 CFR 脉络（DeepCFR→…→VR-DeepPDCFR+，AAAI'26 SOTA）都靠"平均策略"
收敛、且无 last-iterate 理论。A-CFR 把 CFR 的反事实分解与锚定正则化动力学
接通，得到**当前策略 last-iterate 线性收敛到精确 Nash** 的 model-free 深度
CFR，删除平均策略网络/蓄水池缓冲/历史网络存储。目标 ICLR 2027。

核心更新（逐信息集乘性，锚 μ 每 K 步快照，温度 τ 固定不退火）：

    σ^{t+1}(a|I) ∝ σ^t(a|I)^{1/(1+ητ)} · μ(a|I)^{ητ/(1+ητ)} · exp(η·q̃(I,a)/(1+ητ))

---

## 2. 环境（Ubuntu，务必先弄对）

- 上一台机器用 conda env `InforMARL`，**那个路径在 Ubuntu 上无效**。
- 在 Ubuntu 上：`conda env list` / `which python` 探测，或直接问用户用哪个 env。
- 依赖见 `requirements.txt`：numpy + torch(带 CUDA) + matplotlib（+ pandas 仅分析用）。
  torch 版本按服务器 CUDA 选；代码是 Python 3.8 写法，向上兼容。
- **OpenSpiel** 只在 Ubuntu 需要：`pip install open_spiel`（官方仅支持 Linux/macOS，
  这是当初不在 Windows 上装的原因）。
- 跑长实验的习惯：后台进程 + 输出重定向到文件，脚本支持 `--quick` 冒烟。

---

## 3. 代码地图（`code/`）

| 文件 | 内容 |
|---|---|
| `games.py` | EFG 树引擎；`build_kuhn` / `build_leduc` / `build_liars_dice(sides)`；节点带 `meta` 供特征编码 |
| `exploitability.py` | 精确 best-response 与 NashConv（全程可用，无需近似 BR） |
| `algorithms.py` | CFR / CFR+ / 表格 A-CFR |
| `sampling.py` | 采样版 A-CFR（λ-估计器）+ OS-MCCFR 基线 |
| `neural_acfr.py` | 神经版；最终配方开关：`encoding='features'`, `lr_couple`, `q_anchor`, `q_replay`, `eta_min`, `lam`（可='adaptive'，但见开放项） |
| `features.py` | 泛化特征编码（Kuhn/Leduc/Liar's Dice），替换 one-hot |
| `run_phase0*..4b.py` | 各阶段实验驱动；对应编号判读文档 |

**新增博弈/编码的方式**：在 `games.py` 加 builder 并给 Decision 挂 `meta`，在
`features.py` 的 `_ENC` 注册 `(iset编码, hist编码)` 两个函数即可，算法零改动。

---

## 4. 当前记分牌（last-iterate NashConv，400k episodes，seed 0）

| 博弈 | 神经 A-CFR | 表格采样 A-CFR | OS-MCCFR last / avg | 结论 |
|---|---|---|---|---|
| Kuhn | **0.0088**(min) | 0.0181 | — | 神经胜 |
| Leduc | **0.702** (λ=0.5+三冻结) | 1.394 | 1.779 / 0.378 | 神经大胜同口径；对 avg 1.9× 且仍降 |
| Liar's Dice(5) | 0.387 (λ=0.75) | **0.238** | 0.173 / 0.117 | 神经**输**表格（适用边界，见下） |

**适用边界（重要、诚实）**：神经泛化只在**方差大/数据饥饿**的博弈划算
（Leduc：效用尺度 13、表格 Q 数据饥饿）；在结构温和、表格本就高效的博弈
（liars5）泛化开销无红利偿还。真正大博弈无表格选项，对比是神经-vs-神经
（DREAM/ESCHER/VR-DeepPDCFR+）——**这是下一阶段的真正裁决场**。

---

## 5. 最终配方（迁移基线，详见 `23` 号文档 §5）

```
表格/采样: 归一化+效用缩放, η=0.5, τ=0.1, K=⌈7/ln(1+ητ)⌉, B=16, ξ=0.4,
          超相位 (η×½, K×2) 每 4 相位, λ 按博弈 100k 廉价扫描 {0.5,0.75,1}
神经:     + encoding='features' + distill 20 + lr 三联动(lr_couple)
          + q_anchor + q_replay(2万/512)
          + 噪声大的博弈 eta_min=0.0625 冻结(η/K/lr 同冻), 温和博弈持续退火
```

**神经误差三分解**（论文 Cor，全部仪表验证过，见 `20` 号 §2）：
拟合偏差（可忽略）/ Q 漂移方差（修复=Q-anchor+Q-replay，单用皆败）/
SGD 参数噪声（修复=lr 与 η 联动）。统一原则"**锚定一切慢变量**"。

---

## 6. 开放项 / 下一步优先级（Ubuntu 阶段）

1. **OpenSpiel 适配层** + 与 DREAM/ESCHER/MMD/VR-DeepPDCFR+ 的神经-vs-神经
   正面对比（公平口径：NashConv vs 采样局数；同时报内存优势——我们无平均网络）。
2. **大博弈**：Battleship、Dark Hex、liars_dice 标准 sides=6（≈24.6k 信息集，
   我们引擎已支持，精确 NashConv 0.95s/次）。
3. **理论收尾**：Thm 2(b) 线性率证明全文（路径见 `11`/`15`，主战场）。
4. **adaptive λ 开放问题**：λ* 博弈依赖（Leduc 0.5 / liars5 0.75 / Kuhn ~1，
   由 λ²·Var_IS vs (1−λ^D)²ε_Q² 平衡决定）。当前 κ·EMA|corr|/u_max 映射双向
   失灵（机制见 `23` 号 §2）——正确归一化需要 ξ/深度信息，待解。
5. P3f 残余轻微爬升（1/4 种子）；论文拼装（`paper/main.tex` 骨架 + `refs.bib` 已就位）。

---

## 7. 仓库里没有的东西（被 .gitignore 忽略）

- `__pycache__/`、LaTeX 编译产物。
- `results/p3c_c0_diag_full.csv`（~1MB 的逐更新诊断，可由 `run_phase3c.py` 重生）。
- 其余所有 `results/*.csv` 与 `*.png`（实验证据）都已入库。
