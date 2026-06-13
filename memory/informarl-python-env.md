---
name: informarl-python-env
description: 本项目所有代码必须用 InforMARL conda 环境运行（用户明确要求）
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 8982562f-7c50-4497-a31f-4e46b5dc1fd0
---

本项目代码须在一个专用 Python 环境里跑；环境名/路径随机器而变。

**Why:** 用户 2026-06-12 明确指示。Windows 初验机用 conda env `InforMARL` = `D:\Anaconda\envs\InforMARL\python.exe`（Python 3.8.20、torch 1.12.1+cu113、CUDA 可用 4060）。2026-06-13 起项目同步到 GitHub（github.com/axulor/ACFR），正式实验在另一台 **Ubuntu 服务器**上跑，环境会不同。

**How to apply:**
- **Windows 初验**：`/d/Anaconda/envs/InforMARL/python.exe <script>`。
- **Ubuntu（当前主战场）**：环境名未知——先问用户或探测（`conda env list` / `which python`），确认有 numpy+torch(CUDA)+matplotlib 后再跑（见 requirements.txt）。**不要**假设还是那个 Windows 路径。
- 代码兼容 Python 3.8 写法（无 match、无 3.9+ 类型语法），可向上兼容更高版本。
相关 [[acfr-project-overview]] [[windows-prelim-ubuntu-main]]
