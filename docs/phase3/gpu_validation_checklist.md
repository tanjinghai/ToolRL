# GPU 尚待验证：先冒烟，再决定正式范围

这份清单是后续真实租期的验收条件，不是已通过记录。

1. 用批准的整机报价与实例实际开机时间启动预算账本；核对历史实际项目账单、其他待扣费用。环境安装计入本次租期。
2. 确认 torch/CUDA/driver/FlashAttention/xformers/vLLM/Ray/FSDP 的真实 ABI、单卡 NCCL、checkpoint FULL_STATE_DICT/optim_state 的内存峰值。
3. 本地完整 Qwen snapshot 与 tokenizer 同 revision，模型 SHA256 seal、Phase 2 全部生成物 hash、最长 RL prompt 4293 token、SFT 总长 2706 token。训练不读取 Test。
4. SFT 10 有效更新：有限 loss/grad、首个 assistant token 与 EOS 监督、权重改变、逐步 loss 记录；导出在独立进程加载 tokenizer/权重生成，统一 validation 评测成功。当前主路径全参数，未实现 LoRA。
5. GRPO 32 prompt 子集、16 prompt/rollout、n4、10 updates，实际每次64条回答，单卡 micro4、累积16、一次 optimizer step。记录 prompt/response 实际长度、显存峰值、初始/稳态/验证/保存耗时和 s/update。
6. 检查 UID 归组打乱后仍正确；组 advantage 有限、std correction=1；clip objective、entropy=.001 保持；reference 参数冻结且无梯度。paper-style reward KL=0、显式 KL false；legacy-default reward KL=.001。数值检验实际 KL 路径，不只看配置。
7. 检查 reward 只解码 completion，不受 prompt 中标签字面量影响；末有效 response token 写入四分量。原版类型满分假阳性已知，只有显式诊断批准可运行，不能宣称通过安全闸门。
8. SFT 相同 world size/config/data、optimizer/scheduler/RNG/epoch/next_batch 恢复后与不中断训练比较至少下一更新；GPU 尚未验证。GRPO 无精确 Ray/vLLM/sampler 恢复，只能新 run 从导出权重重启 optimizer，E2/E3 对照保持同样恢复策略。
9. 校验目标更新数恰好达到、周期和最终保存、导出后重新加载、不可覆盖 step、失败保留旧 checkpoint；模拟一次超时/失败，确认本地子进程退出并检查 `nvidia-smi`。
10. 基于实测吞吐再固定 E2/E3 相同配置；E3 regression 门槛已预登记，E4 必须真实 validation 坏例与候选 reward 审查。Final Test 在候选和配置冻结且另行批准前不打开。
11. 独立持久备份目录回读 hash；本地第二份不等于异地备份。无验证过的 AutoDL 停机 API，必须在平台控制台停止实例并核对停止计费状态，不能只相信进程退出。

遇 NaN/Inf、梯度异常、3 个连续 batch >90% 同分 prompt 组、hash/版本不符、OOM、权重同步错误、超时/预算不足立即结束相关阶段。默认训练重试为0；改变 batch/长度/模型/预算必须重建计划和审批，不现场自动扩大实验。
