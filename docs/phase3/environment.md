# 候选 AutoDL 环境与迁移边界

截至 2026-10-18，未租 GPU。下面是锁定的**候选版本组合**，不是已验证训练环境或完整传递依赖 hash lock。

首选 Linux x86_64 / Python 3.10 / ≥64GiB主机RAM（优先128GiB）/ A100 或 A800 80GB / CUDA toolkit 12.1 与兼容 driver；仓库内 veRL 0.1、torch 2.4.0、vLLM 0.6.3、xformers 0.0.27.post2、Transformers 4.47.1、tensordict 0.5.0、NumPy 1.26.4、Ray 2.40.0、FlashAttention 2.6.3。直接依赖见 `scripts/autodl/requirements-oldstack.txt`；FlashAttention 在 torch 安装后用 `--no-build-isolation` 构建。实际 resolver 和 GPU ABI 验收后才可将 `pip freeze`、wheel SHA256、driver、镜像 digest 固定为正式环境。根目录旧 requirements 不用于这条安装流程。

vLLM 0.6.3 官方安装文档说明 Linux、Python 3.8–3.12、预编译 CUDA 12.1，且不同 CUDA/PyTorch 构建存在二进制兼容限制：[官方安装文档](https://docs.vllm.ai/en/v0.6.3/getting_started/installation.html)。该版本发布元数据明确绑定 torch 2.4.0 和 xformers 0.0.27.post2：[PyPI 发布元数据](https://pypi.org/pypi/vllm/0.6.3/json)。这些来源支持候选兼容判断，不证明本项目 FSDP/Ray/vLLM 已运行通过。

备选为同一软件组合的双卡 A100/A800 80GB：需统一把 SFT micro 从 1 改成 2、GRPO micro/logprob_micro 从 1 改成 2，DP 整除检查通过后重新固定 configs/plan/approval；TP 保持 1，FSDP 做数据并行。它是新的实验配置，不能临时在已批准 run 中更改。4090 24GB 只能作为需另建 offload/短上下文训练子集的探索候选，不把 1.5B 全参数 actor + Adam + reference + vLLM 默认塞进 24GB。

5090/Blackwell 不进入当前旧栈候选。PyTorch 2.7 才引入 Blackwell/CUDA 12.8 支持：[PyTorch 官方发布说明](https://pytorch.org/blog/pytorch-2-7/)。当前仓库依赖特定 vLLM 0.6.3 私有接口；升级 torch/vLLM 不只是修改 requirements。未来若迁移，需要重新接入 hybrid sharding/weight sync、rollout API、checkpoint、advantage/有效 batch、KL/entropy/reward 语义，并重跑 E0 对照与数值测试，当前未实施迁移，不估算未经实测的收益或小时价格。

`check_env.py --gpu` 验证 Linux/Python/磁盘、版本、pip check、CUDA12.1、设备型号/数量/BF16、CUDA matmul 和相关 import。它**不**代表完成分布式或模型训练冒烟。付费环境安装经 `bootstrap.py` 的固定 commit/approval、费用、超时闸门；安装日志与实际 freeze 位于被忽略的 `runs/bootstrap/`。

Mac CPU 环境独立在 `/tmp`，没有 CUDA/vLLM。用于 Dataset、objective 和 optimizer/RNG 单步恢复测试；不用于 AutoDL 二进制兼容判定。
