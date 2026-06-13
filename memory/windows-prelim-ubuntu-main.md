---
name: windows-prelim-ubuntu-main
description: 用户指示：Windows 本机仅做初步验证，不装 WSL2；正式实验经 GitHub 同步到另一台 Ubuntu 机器上做
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 8982562f-7c50-4497-a31f-4e46b5dc1fd0
---

用户不建议用 WSL2 跑实验（Windows 虚拟化下 CUDA 坑多）。当前 Windows 项目仅用于初步验证；之后项目同步到 GitHub，在另一台 Ubuntu 系统上继续推进（OpenSpiel/基线复现/大规模对比都放那边）。

**Why:** 用户 2026-06-12 明确指示，纠正了我先前"装 WSL2 跑 OpenSpiel"的提议。

**How to apply:** 不在 Windows 上装 WSL/重型依赖；保持代码可迁移（博弈引擎与算法解耦，OpenSpiel 适配层留到 Ubuntu 侧写）；适时准备 GitHub 仓库结构（.gitignore、requirements、README）便于同步。相关 [[acfr-project-overview]] [[informarl-python-env]]
