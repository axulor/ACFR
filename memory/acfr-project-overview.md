---
name: acfr-project-overview
description: A-CFR 研究项目（ICLR 投稿）的核心 idea、文档/代码组织方式与当前进展位置
metadata: 
  node_type: memory
  type: project
  originSessionId: 8982562f-7c50-4497-a31f-4e46b5dc1fd0
---

项目：A-CFR（Anchored-regularization CFR）——首个 last-iterate 线性收敛的 model-free 深度 CFR，删除 DeepCFR 脉络的平均策略网络。目标 ICLR 投稿。

**Why:** 根目录按编号 markdown 文档（01–16+）记录每轮"判读+下一步设计"，代码在 code/（run_phaseX*.py 驱动实验），结果在 results/（csv+png）。每轮工作流程 = 跑实验 → 写判读文档（含诚实失败记录）→ 设计下一 phase 脚本。

**How to apply:** 继续推进时先读最新两份编号文档掌握状态，沿用该编号与"判读→设计"文档风格。表格版定稿配置：归一化+效用缩放+周期锚，η=0.5、τ=0.1、K=⌈7/ln(1+ητ)⌉；采样版加 B=16–32、λ-估计器、超相位调度、ξ=0.4。截至 2026-06-12 深夜：P0–P2 完成；P3 系列（b–f）收口，神经配方定稿 = 超相位三联动（η/K/lr）+ Q-anchor + Q-replay（开关 lr_couple/q_anchor/q_replay），P3f 上神经 min 0.0088 首次低于 tabular 0.0181，残余轻微爬升记为开放工程项；神经误差三分解（拟合偏差可忽略 / Q 漂移方差 / SGD 参数噪声）写入论文 Cor，统一原则"锚定一切慢变量"。P4（2026-06-13 收口，"初步满意"判定已下）：λ* 博弈依赖（Leduc 0.5 / liars5 0.75 / Kuhn ~1，内部最优，按 λ²Var vs (1−λ^D)²ε_Q² 平衡），adaptive λ 双向失灵记开放问题；三博弈记分牌——Kuhn 神经胜（0.0088）、Leduc 神经大胜（0.702 vs 表格 1.394/OS-last 1.779）、liars5 神经输表格（0.387 vs 0.238，适用边界：泛化只在方差大/数据饥饿博弈划算）。最终配方见 23 号文档 §5。下一步=GitHub 同步→Ubuntu（OpenSpiel 对比、Thm 2(b) 证明、大博弈神经-vs-神经）。教训：floor 测量需漂移检测+2× 视界；单一 δ 模型曾三次误导修复方向。相关 [[informarl-python-env]]
