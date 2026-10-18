# ToolRL Phase 1 阶段报告

日期：2026-10-18。代码审计基线：`8cee13ec0ca72f0461da372a93a6fd8140dbb840`，开发分支：`feature/phase1-audit`。
本阶段仅做审计、轻量CPU验证和文档；没有下载大型模型、租GPU或运行SFT/GRPO，也没有自动开始Phase2。

## 本阶段完成了什么

| 交付 | 内容 |
| --- | --- |
| [仓库审计与架构图](repository_audit.md) | Git初始检查、10个核心调用环节、已有/缺失能力、训练完整性 |
| [数据审计](data_audit.md) | raw/Parquet全量读取、逐条一致性、分布、重复和跨split污染证据 |
| [SFT方案](sft_plan.md) | 400条蒸馏格式、现有Trainer阻塞、最小适配、LoRA/QLoRA取舍 |
| [Reward分析](reward_analysis.md) | 公式、变体、论文/实现差异、反例与优化方向 |
| [Benchmark审计](benchmark_audit.md) | BFCL/API-Bank/Bamboogle依赖、评分偏差与统一独立指标 |
| [环境与500元预算](environment_plan.md) | 原栈/迁移两案、GPU显存与性能验证、分阶段成本帽 |
| [实验和Phase2计划](experiment_plan.md) | E0–E4假设、RL适用性门槛、小规模验证及下一步顺序 |
| [实验记录模板](experiment_record_template.yaml) | Git/模型/数据/config/GPU/用时/费用/指标/失败记录 |
| [机器证据](evidence/audit.json) | 数据hash、环境版本、分布和重叠索引、Reward小样例结果 |
| `scripts/phase1/audit.py`、`requirements.txt` | 可重复的只读审计；不依赖torch/vLLM，不生成训练集 |

## 七个关键问题的结论

**A. 属于哪种Agentic RL？** 工具选择/参数生成的规则奖励后训练：静态历史输入、单次completion GRPO。训练中没有真实环境step或工具执行；Bamboogle多轮执行发生在独立评测中。

**B. GRPO流程是否完整？** 源码的采样→reward→group advantage→policy update→周期验证/权重导出闭合，但尚未做GPU运行验证；原脚本不是可直接按低预算执行的完整实验系统，缺独立评测、稳定SFT适配、精确resume和可靠最终保存。

**C. 怎样加入SFT？** 先清洗/隔离SFT400，修现有SFTDataset列读取和messages适配，保留system/user角色、assistant-only loss和EOS；在小样本验证后训练。全参已有主干，LoRA需额外PEFT接入；本轮不实现整个训练系统。

**D. 怎样可靠比较Base/SFT/GRPO？** 同一Instruct起点、冻结无泄漏split/模板/解码预算、逐样本输出和独立verifier；工具名/参数名/参数值/完整调用/格式/无工具分项报告。官方benchmark分数与内部诊断分开，不能拿reward代替准确率。

**E. Reward优化点？** 已复现Python bool/int宽松相等、贪心同名多调用顺序依赖、无工具正确性空缺；应先严格schema/type、全局匹配与额外调用约束，再按坏例做单因素消融。论文称无KL，默认代码仍施加reward KL=0.001，需显式冻结实验语义。

**F. 500元风险？** 付费开机调环境、旧栈与5090不兼容、全参actor/优化器/KV显存、长think和重试、无可靠resume、磁盘/API支出。先1.5B主线，优先已验证旧栈+大显存短租候选；4090offload/5090迁移均须测吞吐后决定。金额为全项目上限分配，不是当前GPU报价或完成保证。

**G. Phase2先做什么？** 去重与schema审计→冻结train/validation/test→独立verifier→修评测路径/固定分母→token长度与环境方案→提交具体费用方案供确认→获批后冒烟与E0。先不进入训练阶段。

## 最重要的数据证据

- RL=4000、SFT=400；Parquet=train3920/test80，seed31415内存复算完全一致。
- RL482条无工具、1823条单调用、1695条多调用；完全重复冗余9条。
- SFT/RL有**37个完全相同prompt**，其中**1个在RL test**；现有test不可直接作为SFT→GRPO的未见测试。
- 13条RL/1条SFT的label工具名不在声明中；未知参数key涉及112条RL/8条SFT，需回溯，不自动猜修。
- 17条SFT的think含协议标签字面量；回答主体JSON均合法，但原split解析器会失败。
- 本地API-Bank/Bamboogle精确问题检查未发现交集；BFCL、近重复、历史前缀、源ID和预训练污染尚未排除。

## 修改范围与验证

修改仅包括本目录文档/证据、只读审计脚本及其CPU依赖清单、README追加入口、`.gitignore`补充、Bamboogle凭据占位文件改名为示例。原始数据、核心训练/Reward逻辑和LICENSE保持原样。

验证记录见 [`evidence/validation.json`](evidence/validation.json)。审计只证明已检查的静态/CPU性质；GPU兼容、吞吐、收敛、最终模型质量和真实费用全部待AutoDL验证。

复现只读检查（Mac/Linux，独立轻量环境）：

```bash
python3 -m venv /tmp/toolrl-phase1-venv
/tmp/toolrl-phase1-venv/bin/python -m pip install -r scripts/phase1/requirements.txt
/tmp/toolrl-phase1-venv/bin/python scripts/phase1/audit.py --output /tmp/toolrl-phase1-audit.json
```

报告会记录执行时HEAD，所以提交后的复跑与本次基线证据在该元数据字段上允许不同；数据统计、文件hash与探针结果应一致。脚本不解析真实凭据、不访问模型/训练服务。

## Git交付与遗留问题

初始工作区干净，origin/upstream正确，main保留基线。Phase1按有意义的commit提交并仅推送origin功能分支；最终commit SHA、推送和Draft PR结果在本轮交付消息中给出，避免把提交自身SHA写进自身产生循环。不会合并PR或修改远程main。

遗留：源数据provenance、完整近重复/污染检查、SFT adapter、独立评测器、BFCL固定版本与数据、GPU环境锁、训练/保存/resume修复及真实性能测量。均已分配到后续阶段，本阶段不扩展为训练实现。
