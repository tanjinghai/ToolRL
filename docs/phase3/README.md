# ToolRL Phase 3：训练工程与连续实验准备

本轮从 `feature/phase2-data-eval@abd6689` 创建 `feature/phase3-training-pipeline`。已读 Phase 1 的训练、奖励、环境审计及 Phase 2 报告；没有重复构建/改写冻结数据，没有合并 PR #1/#2。实现以现有 veRL 最小修复和独立 CPU 控制模块为主。**未租 GPU、运行正式训练/真实模型推理、下载模型权重、调用付费 API 或打开 Final Test 评测。**

默认入口：

```bash
python run_experiments.py --plan configs/experiment_plan.yaml
```

此命令只检查计划并打印 dry-run，不启动模型、安装环境或付费操作。实际执行必须显式 `--execute --approval /path/approved.json`，并满足批准文件、clean Git、数据/模型/config hash、实例/GPU、预算和墙钟限制。

## 交付与实现边界

| 交付 | 实现 |
| --- | --- |
| SFT | 修复 veRL Dataset/Trainer；`sft_small.json`，全参数 FSDP、完整 think、无截断、真实 assistant-only labels、按全 batch token 归一化累积 |
| GRPO | `grpo_smoke/small/expanded/legacy_kl.json`；旧 Ray/FSDP/vLLM 入口，step/validation/最终保存/KL/冻结 ref/有效 batch 修复 |
| Reward | `toolrl_training/rewards.py`，`legacy` 与 `strict_matching`，原 `rlla.py` 不改；反例证据 `evidence/reward_probes.json` |
| Checkpoint | `checkpoint.py` 原子不可覆盖目录、hash、model/tokenizer/config/state；SFT 精确恢复候选，GRPO 诚实限定 weights-only restart |
| 评测与决策 | `stage.py` 复用 Phase 2 generate/score/Bad Case/API-Bank；`decision.py` 输出 JSON/Markdown |
| 调度与预算 | `executor.py` / `approval.py`，DAG、六种状态、进程隔离、失败重试、恢复、产物核验、预算/时限/授权闸门 |
| 部署 | `scripts/autodl/` 候选依赖、受预算约束安装、环境/数据/模型检查、冒烟、外部 timeout、候选冻结 |
| 验证 | `scripts/phase3/validate.py`、`tests/phase3/`；[CPU 结果](evidence/validation.json)、[环境](evidence/cpu_environment.txt) |

## A. SFT 数据能否正确进入 Trainer？

实际 Dataset 使用 Phase 2 Parquet 的 `prompt=[system,user]`、`response`，修复了 DataFrame `.tolist()` 错误；不把消息展平为单一 user。共享 `adapt_sft` 调用 Phase 2 `encode_sft`，原生 Qwen 模板仅一次，不添加第二个 BOS。未 shift 的 labels 中 prompt/padding=-100，assistant 首 token 到一个 EOS 全部监督；Trainer 的 causal shift 只做一次。原始完整 think 保留。3072 容纳现有 train 最大2706与 validation 最大2693；超长在模型加载前报 row index，禁止 left/right truncation；若未来长度增长，调整全配置或另建带 ID/原因的 train-only 子集，不改 frozen 数据。

全参数 FSDP 是本阶段唯一训练实现：已有 optimizer/导出路径，1.5B 80GB 的首条路径更容易控制变量。LoRA 可减 SFT 显存，但需要新增 PEFT、target modules、merge 精度与导出一致性验证；本轮没有实现，不伪称支持 LoRA GRPO。E1 评测与 E3 初始化均读取同一 E1 `model.json` 的 checkpoint hash。

累积不是简单平均不同长度 micro loss：先求全 optimizer batch 的有效 assistant token 总数并跨 DP 归并，各 micro 的 CE sum 用同一分母。尾 batch 不丢失；单卡训练245、验证37均保留。多卡 DistributedSampler 为保证同构迭代可能 padding 重复末样本，训练/内部 val loss 需记录此边界；最终独立 validation 不重复。gradient checkpointing 开启，use_cache=False；异常梯度在 optimizer.step 前拒绝。真实 FSDP 数值和显存仍待 GPU。

## B. GRPO 能否读取 Phase 2 数据？

沿用 `RLHFDataset`，读 `grpo/train.parquet`2511与`validation.parquet`307，工具 schema 和 `extra_info.call_order` 随非 tensor batch 保留。prompt_length=4608，response_length=1024，实际 engine context=5632（上限预算6144）；train 最长 prompt4293，禁止左截断。冒烟在原 train 前32条的明确子集上运行，不改原文件；正式使用全部 train。验证按4条分批，末批不丢、不 shuffle。原训练内部的 reward validation 只做诊断，checkpoint 决策以 Phase 2 独立 verifier 为准。

| 配置 | prompt batch / n | 实际单卡 micro / 累积 | 有效 updates | 范围 |
| --- | --- | --- | --- | --- |
| smoke |16 /4|4 /16|10|32条 train prompt，10 epochs 为容量上限，达到10即停 |
| small |16 /4|4 /16|64|2511 train，2 epochs 容量上限 |
| expanded |16 /4|4 /16|128|可选，需另行批准；不默认执行 |

这是旧 worker 先按 DP 除、再乘 rollout.n 后的实际值，不是 micro=1 条回答。每 rollout64条 completion，mini64，因此每 rollout 一次 optimizer update。传入 batch 必须在除 DP 前整除；不允许整除为0。显存未知时不能把这个数字误当单序列 micro。E2/E3 同一 config hash、数据、生成预算、步数；E3 被预授权降为10步 diagnostic 时明确不是等预算正式对照。

## C. 修改了哪些原 veRL 代码？

`fsdp_sft_trainer.py` / `sft_dataset.py` 接入 labels/非丢弃 loader/token 加权累积/恢复保存。`ray_trainer.py` 修复1..N最后更新、按配置 validation batch、强制最终保存、固定 train subset/seed、健康闸门与 progress。`main_ppo.py` 只解码 completion 给版本化 reward，不从实验名猜响应边界；四分量写末有效 response token，空响应拒绝，learned RM 四元组不兼容路径明确拒绝。`fsdp_workers.py` 增加 batch 归一化检查、reference requires_grad=False、optimizer/scheduler/RNG 捕获和不可覆盖导出。`core_algos.py` 用同设备 stack 求组均值/std，保留 sample std correction=1；`dp_actor.py` 拒绝非有限梯度，显式 KL loss 从错误的减号改为正号。rollout 将 max_num_seqs/max_num_batched_tokens/seed 真正传到 engine；RL 无 template 备用分支修正遍历空字符串错误。

主候选 `paper-style`：reward KL=0，explicit KL loss=False；`legacy-default`：reward KL=.001，explicit KL loss=False。组 sample std、entropy=.001仍为原代码设置，不伪称与论文全部公式一致。ref 仍加载起始策略，E2=Base、E3=SFT；即便 KL=0仍保留 ref 计算便于验证/控制一致性，可能有额外成本。未迁移 veRL/vLLM；旧顶层大 batch shell 保留历史用途，不作为 Phase 3 启动入口。

## D. 原始 Reward 与严格变体的区别？

原文件逐字保留。legacy wrapper 固定默认环境开关、指定 Qwen 包装，仅传 completion，返回 total/format/correctness/length，不继承 shell 隐式实验开关。严格变体使用严格协议/JSON/schema；bool/int 区分，schema 明确 number 时才允许 int/float 数值等价。工具名多重集 Jaccard、参数 key Jaccard和严格值匹配提供部分分；parallel 同名调用用独立 Hungarian 全局最优配对，sequential 保留位置；额外调用通过覆盖因子扣减全部正向证据。length 默认0，无工具自由文本正确性0，仅可惩罚错误工具，不虚构语义判断。它不是直接调用 verifier EM 当唯一 reward。

反例涵盖 bool/int、错误值/参数、额外调用、同名多调用/顺序、JSON错误、no-tool。legacy 已知满分假阳性不会因 wrapper 消失；正式默认不能越过它，运行 legacy E2/E3 需单独 `allow_legacy_reward_diagnostic` 批准，标注诊断。严格候选在当前反例无满分假阳性，只是离线验收，不保证真实 reward hacking 已消除。E4 会对 E2 已保存 validation 预测离线重算候选奖励与独立EM，工具样本出现满分假阳性即停止并输出 audit；还必须有 E2 真实 validation scored hash、人工审查 JSON 和批准 hash，才能从同一 Base、同样预算单独训练 strict；不会接着 E2 多训练几步冒充改进。

## E. Checkpoint 是否完整恢复？

step 目录位于唯一 run/stage/attempt 下，原子临时目录→rename，拒绝覆盖，记录所有文件 SHA256。正常训练结束必保存目标 step；失败不覆盖上一个完整模型。每个训练 stage 默认仅保留最新一个完整状态，新 checkpoint 完整验 hash 后才删除本 stage 的旧 step，避免上百 GB 的周期快照累积；不同实验的最终产物不会被此保留策略删除。临时发布需要新旧同时容纳；preflight要求128GiB可用，最终仍需按实测状态大小和独立备份容量复算。模型、tokenizer、HF config、训练 config、optimizer、scheduler、Python/NumPy/torch/CUDA RNG 一并保存。原子 rename 不等于已验证断电耐久性；未做分布式故障容错。

SFT 实现同 world size、相同训练签名（data hash、mask/长度/batch、optimizer、seed、epochs/steps）、相同 checkpoint 权重的恢复，保存 epoch/next_batch；独立 DataLoader generator 防止重建 iterator 消耗 dropout RNG。手动恢复需将 `model.partial_pretrain` 和 `trainer.resume_path` 都设为该完整导出，输出到新的 run；不能拿不同 batch/data/config 叫精确续训。CPU toy optimizer/RNG 下一步可复现不代表真实 FSDP 恢复已通过。

GRPO 保存 optimizer/scheduler/RNG材料及 driver progress，但**未实现 Ray/vLLM/sampler 精确恢复**。当前支持的重新启动语义是读取导出权重，新 run 重建 optimizer/reference 和采样进度；必须标记 `weights_only_optimizer_restart`，不可混叫 resume，也不能与不中断 E2/E3 随意比较。未来恢复整个 RL 运行还要恢复随机采样引擎、data cursor、KL/健康窗口状态。

## F. 已实际 CPU 验证什么？

最终实测：Phase 2 **35项** + Phase 3 **41项**，共76项全部通过、零跳过；223个Python文件编译、7个shell语法与pip check通过。可复现证据见 `evidence/validation.json`、两份测试日志和 `source_hashes.json`。隔离环境Python3.9.6、Mac ARM64、torch2.4.0（CUDA=null）、Ray2.40.0仅导入使用，没有启动集群；AutoDL候选仍为Python3.10。测试覆盖真实 Qwen282条 SFT mask/EOS/template/无截断、实际 veRL Parquet Dataset、全部2511 RL prompt、Hydra 原配置解析、有效 batch/KL、reward反例、真实 CPU tensor loss/micro梯度等价、GRPO advantage/clipping、CPU optimizer/scheduler/RNG下一更新恢复、immutable checkpoint、DAG/依赖/重试/安全停止/费用拒绝/授权/数据/模型隔离、完整344 validation与597 API Mock评测/比较/Bad Case/决策/备份和重复运行。Mock Final阶段跳过，未生成或评分 Test。全部 Mock 分数均不是模型成绩。

```bash
# 隔离 Mac CPU 环境；不安装根目录 CUDA 依赖
python3 -m venv /tmp/toolrl-phase3-venv
/tmp/toolrl-phase3-venv/bin/python -m pip install -r scripts/phase3/requirements-cpu-mac.txt
# 实际 CPU 完整版本见 evidence/cpu_environment.txt
/tmp/toolrl-phase3-venv/bin/python scripts/phase3/validate.py --output /tmp/new-phase3-cpu-evidence

# CPU Mock 独立 run；相同 run 重跑跳过已验收阶段
python run_experiments.py --plan configs/phase3/mock_plan.yaml --mock --run-dir runs/my-mock
```

## G. 哪些必须 GPU 验证？

[GPU 待验证清单](gpu_validation_checklist.md)：旧栈 ABI/NCCL/FSDP、模型加载与 vLLM 权重同步、实际 loss/advantage/KL/entropy 数值、显存/吞吐、最终导出真实生成、全状态保存内存和真实恢复。AutoDL 候选环境不称“已验证”；BFCL 官方执行未跑，Bamboogle付费搜索/judge未跑。

## H–I. 首选环境与单卡80GB成功预期？

优先试 A100/A80080GB、Linux/Python3.10 的固定旧栈，[环境分析与官方来源](environment.md)。1.5B FP32 actor/grad/Adam约24GB，还要加 reference、vLLM KV和长序列激活，采用 ref offload、gradient checkpointing、eager rollout、vLLM利用率.25。**单卡80GB有合理尝试依据，没有峰值显存和速度实测保证。** 单卡4090不是当前全参数默认；5090需新栈迁移，当前不自动升级。报价和每有效 update 成本尚未知，用平台真实报价与冒烟 s/update决定，不能凭卡名预设最便宜。

## J. 租 GPU 后第一条命令？

先完成用户租赁与明确批准、固定代码/计划、数据复制、模型 snapshot 的人工核验；实例从实际开机时间开始计费。第一条检查命令：

```bash
python scripts/autodl/check_env.py --host-only
```

随后在 Linux Python3.10 环境（不是 Mac）执行：

```bash
python scripts/autodl/prepare_approval.py --plan configs/phase3/smoke_plan.yaml --output /root/smoke-proposal.json
# 人工填真实 GPU/整机或卡时价、开机时间、历史账单、caps、允许阶段等；显式批准后保存 approved.json。
# 本地已准备模型后 seal_model.py 固定文件 hash；来源需操作人核验，本脚本不自动下载。
python scripts/autodl/seal_model.py --path /root/autodl-tmp/toolrl/cache/base --revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306
bash scripts/autodl/install.sh configs/phase3/smoke_plan.yaml /root/approved-smoke.json
python scripts/autodl/preflight.py --expected-commit <approved-full-SHA> --output-root /root/autodl-tmp/toolrl --plan configs/phase3/smoke_plan.yaml
bash scripts/autodl/smoke.sh /root/autodl-tmp/toolrl/runs/smoke001 /root/approved-smoke.json
```

本轮不提供自动租赁或自动权重下载。先在固定快照目录备好模型、将 `MODEL_MANIFEST.json` 的hash填入批准文件；同一租期复用，缺失或hash不符拒绝运行。GPU端需自行传入冻结生成物（rsync/持久卷复制），preflight全量核hash，不重新构建数据。

## K. 怎样连续执行 E0–E4？

`configs/experiment_plan.yaml` 登记 E0→E1→E1_eval→decision→E2→E2_eval→E3→E3_eval→E4→E4_eval→final→backup。调度器按依赖和登记顺序执行，阶段是结构化 handler，不是 shell 串；日志、每次 attempt、结果和 hash持久化。E2只依赖E0/reward_gate，SFT失败仍执行已授权E2；E3若E1 EM下降≥3个百分点按预登记规则skip，规则也支持预授权10步diagnostic/full/manual。这个3pp是工程门槛，不是显著性结论。

E4遇缺失真实 Bad Case 审查会写 stopped 决策并结束本地付费调度，不等待用户空转。需要新人工决定时先保留模型/日志，平台控制台停机。相同代码/计划/数据/模型身份允许显式续签 approval，保留旧批准记录和已完成 stage；新配置/模型/Git须新 run。运行中崩溃时先检查旧PID和本地/远端Ray子进程；仍存活拒绝恢复；确认旧进程不存在后只在登记重试范围内新 attempt，完成任务不会重跑。

每个模型自动同解码 validation+API-Bank（完整597/有效GT479），并输出 E0→E1/E2、E1/E2→E3、E2/E3→E4 配对比较，保留失败固定分母。E4若 E3已跳过，示例 DAG 对 E3比较分支也会跳过；要改为仅E2对照需另行固定/批准计划，不隐式换口径。

Final Test需先 `freeze_candidates.py` 固定 real validation 已完成的 candidates，再为同计划签发含 `allow_final_test` 和 freeze hash 的批准。所有 candidate 在开 Test 前核验。`FINAL_TEST_STARTED.json` 及固定 freeze 文件旁的跨run标记以独占写创建，一旦开始不允许自动重试第二轮；中途失败需人工审计，不用结果重新调参。未批准时安全停止。BFCL次要兼容不阻塞主链，Bamboogle不在计划中。

## L. 何时必须停止？

代码/data/model/config/产物hash变化、GPU或价格口径不符、无批准/过期/超范围、未知历史账单、余额或预计剩余成本不足、墙钟/阶段费用上限、NaN/Inf/梯度异常、3个连续batch >90%零方差组、OOM/导出/权重同步失败。默认无自动训练重试；每次 retry独立目录并计费。需人工决定的 stage写报告后退出。父进程有本地 process group超时和预算检查，另提供GNU timeout外部watchdog；它们均未在GPU上验证可清理所有Ray远端worker，实际必须核查平台和设备状态。

`stopped`/预算拒绝不启动新付费阶段，模型/日志保留本地；正常流程执行独立目录完整备份并回读hash。安全停止时如果无法在已批准费用/时间内完成备份，不伪称已异地备份；应按预留窗口人工复制有效产物并立即平台停机。`STOP_AND_BILLING.md`明确进程退出≠停计费。没有验证过的AutoDL关机接口，**必须人工在平台控制台停止实例并核对账单**。

## M. 如何尽量守住500元？

总帽≤500，先扣历史实际账单和其他已承诺费用；本轮未运行付费操作，报告项目GPU支出0并不证明账户没有其他费用。授权文件 `historical_spend_rmb=null` 时不能execute。`price_basis=instance`不再乘卡数；`per_gpu`才乘GPU数。按实际实例开机时间累计运行费用，包含安装/空闲/重试和跨executor重启间隔。每阶段启动需预计剩余墙钟×整机价×1.3+备份/停机储备≤可用预算，同时满足run/阶段费帽与最大墙钟。阶段重试累计计入本阶段runtime，不能通过重启executor清零；实际账单仍需人工核对。

先用10步冒烟测加载/生成/update/验证/save时间，计算本配置 RMB/update 和完整计划估价，再批准small；expanded默认不跑。实际价格、历史费用、剩余预算未知，不能声称一定500元跑完E0–E4，更不能为了填满实验表自动扩大GPU或模型。

## N. 尚未实现或未验证什么？

未实现 LoRA/QLoRA、GRPO精确分布式恢复、远端Ray可靠故障接管、自动租赁/计费查询/平台关机、自动模型权重获取、BFCL官方打分、真实工具交互式RL、自由文本语义判断、统计显著性校准或自动发现E4最优reward。候选直接依赖已固定，完整wheel/hash与镜像锁需GPU环境resolver成功后生成。SFT FSDP exact resume、权重同步、吞吐/显存、真实指标均未GPU验证。控制器不会隐式批准E4或Test；后续用户可在CPU准备审查/续签文件，必要时先停机再决定。

Phase 3 到此停止。下一阶段应先审阅CPU证据、填写实际预算/设备审批，再安排真实GPU冒烟与连续实验；本轮没有执行该阶段。
