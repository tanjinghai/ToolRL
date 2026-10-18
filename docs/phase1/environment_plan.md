# AutoDL 环境、GPU 与成本方案

核验日期：2026-10-18。以下是候选部署方案，**未在 GPU 上验证**，没有租机/训练动作。Mac 仅使用 `/tmp/toolrl-phase1-venv` 的 numpy/pyarrow 做数据读取，未安装 CUDA 训练栈。

## 依赖事实

README 的示例为 torch=2.4.0/cu121、vLLM=0.6.3、未固定 ray/flash-attn。`requirements.txt` 要求 transformers<4.48、tensordict<0.6、vllm<=0.6.3；`pyproject.toml` 的依赖列表遗漏 pandas/flash-attn/wandb 且未限制 tensordict，与 setup.py 读取 requirements 不一致。没有可直接复用的完整 lockfile。

本地 vendored veRL 版本字符串为 **0.1**，是项目内一份代码，不应另外 `pip install -U verl` 覆盖。`verl/third_party/vllm/__init__.py` 只接受四个精确版本。vLLM=0.6.3 的 [PyPI 元数据](https://pypi.org/pypi/vllm/0.6.3/json) 明确依赖 torch==2.4.0、numpy<2、transformers>=4.45、ray>=2.9、Linux x86_64 的 xformers==0.0.27.post2。

官方 [vLLM 0.6.3 安装说明](https://docs.vllm.ai/en/v0.6.3/getting_started/installation.html) 要求 Linux，并强调 CUDA/PyTorch 构建的二进制兼容性；`nvidia-smi` 的 CUDA 数字只是驱动能力上限，不是当前 torch/vLLM 实际 runtime。

## 两条路线

| 项目 | A：保留原栈 | B：受控迁移 |
| --- | --- | --- |
| Python / OS | Linux x86_64、Python3.10 | Linux x86_64，随选定官方训练容器/lock 固定 Python（例如官方当前 uv 路径为3.12） |
| 核心栈 | torch2.4.0+cu121、vLLM0.6.3、本仓库 veRL0.1 | 选**固定 release/commit + image digest**的现代 veRL/vLLM/torch 配套环境，不能混装 latest |
| 其他候选约束 | transformers4.47.1、numpy1.26.4、tensordict0.5.0、ray2.40.0；FlashAttention2.6.x 为待编译测试候选 | 使用所选 release 的官方 lock/容器；FlashAttention、Ray、Transformers、Triton 作为整组验证 |
| 忠实度 | 主循环与原代码最接近；还要决定论文无KL或代码默认KL | 移植数据和 reward 契约，逐项对齐 GRPO 语义，不能默认现代 GRPO 等价 |
| 工作量 | 中：解依赖、SFT adapter、轻量 bug 修复 | 高：配置、reward API、rollout/权重同步、checkpoint、loss 差异 |
| 风险 | 老包源编译、依赖无上限、老私有 API；新 GPU 架构支持不足 | 同名参数语义变更、batch/advantage/loss/reward 聚合改变导致实验不可比 |
| 显存 | 原 actor FP32 参数+AdamW 状态重；FSDP/手工 offload | 可能支持更好的 offload/LoRA，但必须用实际峰值验证，不保证自动省显存 |

A 中非核心候选版本**不是验证过的 lockfile**。安装前在 Linux CPU/无卡环境做 resolver dry-run，补齐 pandas、pyarrow、hydra-core、accelerate、wandb 等实际 import；按既定 torch 版本编译扩展，`pip check` 后固化 freeze/wheel hashes。不要把 Mac 审计用 numpy2.0.2 带入要求 numpy<2 的旧训练环境。

B 的兼容底线：官方 [PyTorch2.7 发布说明](https://pytorch.org/blog/pytorch-2-7/) 引入 Blackwell/CUDA12.8 支持；这不等于 torch2.7 能与任意 vLLM/FlashAttention 组合。查询到的 [现代 veRL 安装页](https://verl.readthedocs.io/en/latest/start/install.html) 要求 Python≥3.10、CUDA≥12.8，并列出新的 rollout 与锁文件工作流，和本地分叉差距很大。本轮不把移动的 latest 页面当已验证版本组合，更不直接覆盖仓库。

迁移必须对齐：group n、uid 分组、std correction/epsilon、sequence reward 与token KL、clip loss token/sequence reduction、entropy、ref 起点、EOS mask、temperature、额外样本丢弃、优化器更新次数。用相同 synthetic batch 对比 reward/advantage/loss，再检查权重同步和导出。若这些变化不可避免，单列“迁移实验”，不可冒称严格复现。

## GPU 选择与显存

粗估采用 P=1.5B/3B（实际参数量以模型 config 为准）。原 actor 默认 FP32 参数，保守按参数4+梯度4+Adam两状态8≈16 bytes/P：1.5B 约24GB、3B约48GB（十进制），**尚未包括**激活、混合精度副本、vLLM权重、reference、KV cache、NCCL和内存碎片。BF16 单份模型约3/6GB。FSDP 单卡不能分摊参数，多卡与offload才能改变这个账。训练阶段峰值与rollout阶段峰值不同，不能简单把 gpu_memory_utilization 当整进程上限。

| GPU | A/B 建议 | 预算策略 |
| --- | --- | --- |
| RTX4090 24GB | A：推理/LoRA SFT合适；1.5B全参GRPO需CPU offload和小token预算，3B更高风险 | 若已验证 offload 吞吐足够再选；不能宣称1.5B一定直接放得下 |
| RTX5090 32GB | 选择B的已验证Blackwell栈；不推荐把A的cu121/torch2.4直接搬过去 | 便宜时可用于独立推理或LoRA SFT；迁移调试成本可能超过省下的卡费 |
| A800/A100 80GB | A优先候选；1.5B较宽裕，3B仍需控制激活/KV峰值 | 可以短时间用大显存卡完成训练，比较**每有效update费用**，不只比较小时价 |
| 其他/多卡 | 按架构、显存、PCIe/NVLink、CPU内存和已验证镜像判断 | 原论文两张80GB不是硬性要求；多卡同步与两倍单价需实测合理 |

推荐先争取 A + 单张80GB + 1.5B 的低调试风险闭环；若报价不合适，再测4090 offload。3B仅在同预算吞吐测试能通过时替代主模型，不默认两模型都跑 E0–E4。可用小卡推理、大卡训练，但最终各模型的主榜推理精度/backend/解码配置要一致，GPU差异记录在案。

**可行性判断（待实测）**：500元下，1.5B、几百到数千训练prompt、有限有效update的E0–E4探索实验有条件可行；3B也可能通过短租80GB卡实现，但更易消耗调试和训练预算。不能承诺两种模型、多seed、论文15epochs和全部reward消融同时完成。有意义的标准是固定对照、可靠评测和可解释坏例，并不要求复现论文每一个规模。

## 正式训练之前必须通过

1. 用户先确认租机型号、小时价、最大开机时长和本次费用上限。依赖准备尽量在无卡模式/预制镜像完成；无卡也可能收费，仍需授权。
2. 保存 GPU/driver、torch.version.cuda、实际 device capability、编译架构、nvcc、CPU/RAM/磁盘、全部包版本；CUDA矩阵乘、BF16、FlashAttention前后向、vLLM短生成、Ray/FSDP1卡初始化分别冒烟。
3. 16–32 prompts、n=4、小token预算做10–20次有效更新，必须包含**最慢长prompt样本**。记录每阶段用时、峰值 allocated/reserved与nvidia-smi、CPU峰值、权重是否更新、奖励方差与零adv组比例。
4. 导出再加载；确认预算期限内确实有checkpoint。短任务显式save_freq较小并验证最后步语义；必要修复另开小commit。
5. 用稳定区间测得 `s/update`（剔除初始化但单列初始化费），按正式token长度/样本覆盖估算总费用；预留至少30%时间波动。超过阶段上限则停机/减规模，不临时扩卡或追加费用。

## 500元预算草案（全项目总额）

GPU单价未登录市场核实，不提供“当前报价”。以下金额为**支出上限分配**，非保证完成时长；讨论 r=2/4/8 元·卡小时只是敏感性假设。

| 阶段 | 上限（元） |
| --- | ---: |
| 环境与性能冒烟 | 55 |
| E0与评测闭环 | 45 |
| E1小规模SFT | 40 |
| E2 cold-start GRPO | 95 |
| E3 SFT→GRPO | 95 |
| E4一个主要reward改动 | 55 |
| 统一最终评测/关键复核 | 45 |
| 磁盘/服务等非GPU费用 | 20 |
| 故障与复现储备 | 50 |
| **总计** | **500** |

成本公式：`C=Σ(实例开机小时×实际整实例小时价)+存储+API+传输/其他费用`；多卡若报价按卡须乘卡数，不重复乘已是整机价的报价。假定可用于GPU的总额430元，对r=2/4/8对应215/107.5/53.75**卡小时**，包含安装等待和初始化，不是纯训练时间承诺。

例：某 run 计划100 updates，实测60s/update，初始化20min，验证/保存15min，r=4元/h，则135分钟约9.00元；加30%波动约11.70元。这个算例不代表ToolRL实测速度。先有实测再批准正式step预算，实际上下文和输出长度可能使时间成倍变化。

最大风险是反复修依赖时GPU仍开机、长think和prefill、KV/optimizer OOM重启、全量15epochs照抄、没有可恢复checkpoint、付费搜索/judge与磁盘残留费。根据 [AutoDL计费](https://api.autodl.com/docs/price/)，按量按实例开关机计时，不按GPU利用率；[扩容数据盘](https://www.autodl.com/docs/local_disk/) 可在关机后继续计费。每次结束保存日志/模型到持久存储并确认实例关机；预算不够优先保留同模型E0/E1/E2/E3和一个E4，小样本结果如实标记探索性，不承诺论文规模显著性。
