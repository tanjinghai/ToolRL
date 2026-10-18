# SFT 可行性与最小接入方案

基线同 [仓库审计](repository_audit.md)。本阶段仅分析，不实现训练系统。

## 已验证现状

- `dataset/rlla_4k_raw/rlla_sft.json` 是 **400 条** JSON 数组，每条只有 `instruction`、`input`、`output` 三个字符串。
- instruction 包含工具清单和协议，input 是已有对话历史，output 包含 `<think>` 和 `<tool_call>` / `<response>`。README 和论文附录说明 think 由 DeepSeek-R1 蒸馏；文件本身没有 teacher revision、生成参数、审核记录，无法独立验证 teacher 来源。
- RL 4k 的 think 是模板占位，不应直接当高质量推理 SFT 标签。SFT/RL 不仅条数不同，训练目标也不同：SFT 学完整目标序列，RL 的 think 不参与正确性标签匹配。
- 实测SFT与RL有37个完全相同prompt，其中1个落在现有RL test；不能照用原划分声称无泄漏，见 [数据审计](data_audit.md)。另有17条think包含协议标签字面量，需处理parser歧义。
- 存在 `verl/trainer/fsdp_sft_trainer.py::FSDPSFTTrainer`，配置为 `verl/trainer/config/sft_trainer.yaml`，理论入口 `torchrun --standalone --nnodes=1 --nproc_per_node=1 -m verl.trainer.fsdp_sft_trainer ...`。没有 ToolRL 专用 SFT shell 脚本。

## 为什么不能直接运行

1. `SFTDataset` 默认读 Parquet 的 prompt/response 字符串，原文件是 instruction/input/output JSON，需要转换。
2. `sft_dataset.py:59–105` 把 key 变成列表；`dataframe[self.prompt_key]` 返回 DataFrame，之后调用 `.tolist()`，默认无 dict keys 路径会抛 `AttributeError`。已在本地用 pandas 最小复现，未加载 torch。不是“只改配置即可用”。
3. `__getitem__` 把 prompt 包成单个 `user`，而 RL 是 `system=instruction, user=input`。把两个文本简单拼接虽然能做 SFT，却引入 SFT/RL 模板差异，应提供最小 messages 适配，并确保仅应用一次模型原生 chat template。
4. trainer 明确拒绝 `data.chat_template` 非空；这个配置项不能解决上述角色结构问题。
5. 默认 max_length=1024、batch=256、micro=16，且训练/验证都有 drop_last。400 条和长蒸馏回答可能丢样本、截断、甚至验证 loader 为空。先统计真实 tokenizer 长度，使用 truncation=error，禁止默默截掉 JSON/EOS。左截断后的 loss mask 也需验证。
6. 默认 checkpoint 指向 HDFS；AutoDL 必须显式 `trainer.default_hdfs_dir=null` 并指定持久本地目录。`resume_path` 有配置字段，当前 fit 没有完整恢复逻辑。

## 推荐接入顺序（后续阶段实现）

先确定无泄漏 split 和 tokenizer revision，再做一个小转换器：保留稳定 sample_id、provenance、原始 hash；输出 `messages=[system(instruction),user(input)]`、`response=output`。为现有 SFTDataset 增加最小 messages 路径，修复列读取；不复制整个 veRL，也不修改 GRPO 算法。

训练前用 2–8 条验证：渲染后的 prompt token IDs 与 RL 数据路径一致；assistant 起始符、EOS 只出现一次；prompt/padding 的 label 全部屏蔽；首个回答 token 和 EOS 有正确监督；每条有效 label 数 >0。10–20 optimizer updates 检查 loss 有限、权重变化、导出后可重新加载及生成。

保留完整蒸馏 output 为论文风格 SFT 主对照，同时把“简短/去 think SFT”作为有预算时的独立消融；不能混淆目标、数据与算法三个变量。SFT 价值假设是学协议/降低格式错误，不预设能提升泛化。

## 全参、LoRA 与 QLoRA

| 方案 | 当前支持 | 决策 |
| --- | --- | --- |
| 全参 FSDP SFT | 主干已有，先修复数据适配 | 最接近原实现；1.5B 单卡仍需量显存，3B 风险更大 |
| LoRA SFT | trainer 没有 PEFT 接入 | 预算紧时可另建轻量 HF/PEFT 入口；要验证 target_modules、mask、导出与合并 |
| QLoRA SFT | 没有 bitsandbytes/4-bit 路径 | 不是本轮默认；Linux GPU 兼容、量化精度和 FSDP 配置增加工作量 |
| LoRA/QLoRA GRPO | 当前 hybrid worker 没有完整 adapter 同步链 | 不可因为 vLLM 源码出现 LoRA 类型就视为支持；需另行验证 |

若使用 LoRA SFT，导出合并后的 HF 模型给现有全参 GRPO 是可研究路线；必须记录合并精度，并让 E1/E3 共用同一 SFT checkpoint。主实验应一致选择 SFT 方法，避免每组随意切换 full/LoRA。

SFT400 的正式训练/验证划分在 Phase 2 冻结；若去重后全部可用，建议按对话/近重复分组分为 360/40，并与 RL train/validation/test、外部 benchmark 交叉去重。不能拿 SFT 自己的 train loss 代替工具能力评测。
