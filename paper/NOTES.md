# P2.5 论文拼装笔记（2026-06-12）

## main.tex 状态
骨架已立：全部定理陈述为论文级形式（源自 11 号文档 + 12/15 号修正），
含 P3b/P3c 新增的 Cor (η* 冻结推论)。`%% TODO[...]` 标记待写块。

## 证明移植清单（中文文档 → Appendix）
| Appendix | 来源 | 状态 |
|---|---|---|
| Lemma 2 (anchor bound) | 12 号 | 直接翻译 |
| Thm 1 五步证明 | 12 号 | 直接翻译 |
| Thm 2(a) | 15 号核对清单 | 余 1 项：treeplex 边界 ∇Ψ 爆炸相容性 |
| Thm 2(b) | 15 号路径 + D3 | **主剩余工作**：Luque+准则B 全文 |
| Prop 3.1 / Thm 3 | 12/15 号 | 直接翻译；Prop 3.2 截断版待写 |
| Prop 4 + Lemma N′ + Cor η* | 12/15 号 + P3c | 等 C0 实测 δ(s,t) 形式后定稿 |

## 图表映射（results/ → 论文图）
- Fig 1 ← figP1A2_scissors.png（83×/39× 剪刀差）
- Fig 2 ← fig2_linear_rate.png / figC2_rate_vs_K.png（13 个数量级直线 + K* 规则）
- Fig 3 ← 拼图：figB1(内环率) + figB2(τ底) + figD1(外环∝1/τ) + D3(Luque a≈37)
- Fig 4 ← figP1B2_sampled.png + figP1C2_leduc.png（λ 谱双博弈反转）
- Fig 5 ← figP3c_C3_final.png（naive 病理 vs η* 冻结修复；P3b_B1 为前身）
- Fig 6 ← figP3c_C1_ucurve.png + figP3c_C0_delta.png（U 曲线 + δ 标度）

## P3c/P3d 落地后的修订（2026-06-12 已完成）
- Cor 已改写为"Q 稳定化退火"形式：δ_policy 可忽略（实测），σ²(t) 含
  Q 误差时变项；配方 = 退火（永不冻结）+ Q-anchor + Q-replay。
- 神经节图表更新：Fig 5 ← figP3d_D3_final.png（四臂）+ figP3e_long.png
  （400k 确证，跑完后替换）；Fig 6 ← figP3d_D1_stabilizers.png（消融）
  + figP3c_C1_ucurve.png（floor(η) 非 U 形 + 爆炸前瞬态的教训）。
- C0 的 1M 行逐更新诊断（p3c_c0_diag_full.csv）可出 qRMSE-vs-floor
  的机制图（appendix）。

## P3f 定稿后的状态（2026-06-12 深夜）
- Cor 已改写为三分解最终形式（fit bias / estimator variance / SGD
  parameter noise，各自 η 标度 + 三个锚修复）。
- 神经节图：Fig 5 ← figP3f_lrcouple.png（带 P3e 未联动对照）；
  Fig 6 ← figP3d_D1_stabilizers.png + figP3d_D3_final.png；
  appendix ← figP3c_C0_delta.png（机制仪表）。
- 失败方案 ablation 表素材见 20 号文档 §2。

## 剩余动作
1. P4a-A1 Leduc 结果 → 实验节神经小节定稿。
2. 实验节每图配 predicted-vs-measured 小表。
3. refs.bib 建库（main.tex 末尾已列文献清单）。
4. Thm 2(b) 全文（理论收尾主战场）。
