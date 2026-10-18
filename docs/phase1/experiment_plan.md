# 实验路线与 Phase 2 执行计划

目标是检验各训练阶段解决什么问题。预算总额500元，[环境与成本约束](environment_plan.md) 对全部实验共同生效。下表是未来计划，未执行训练。

## 冻结比较条件

主模型先选 `Qwen2.5-1.5B-Instruct`，3B作为吞吐/效果都支持时的替代；E0 的 Base 指“本项目训练前的同一 Instruct checkpoint”。所有实验固定 model/tokenizer revision、数据split、protocol prompt、独立评测器、推理预算。E2/E3统一 RL prompt manifest、n、最大长度、有效update数/采样量，另记录总GPU小时和SFT额外成本。

论文无KL与代码默认KL不能混称：先把两者差异冻结在配置与报告，推荐正式主线采用论文式reward KL=0，并保留默认0.001为必要时单列对照；没有实测前不改源码。E3记录reference起点为SFT checkpoint，若统一改为E0 reference则是另一项设计。

## E0–E4

| 实验 | 假设与训练方式 | 数据 / 前置条件 | 评测与保存 | 显存与成本风险 |
| --- | --- | --- | --- | --- |
| E0 Base | 识别已有能力、协议适应成本和能力短板，无训练 | 冻结test、内部validation、API-Bank；BFCL接入后补齐 | 全部逐样本预测、独立六项指标、切片、token/耗时；模型revision | 推理较低，长prompt和KV仍需实测；所有训练依赖E0 |
| E1 Base→SFT | 少量蒸馏能否教会格式且保留泛化；不预设提升 | SFT400经分组去重后的train/val；先修复SFT adapter | E0同一评测；loss、格式/EM变化、输出长度、HF checkpoint | 全参中高；LoRA可降但需固定方案；蒸馏长回答增加成本 |
| E2 Base→GRPO | 无SFT时结构reward能否提高工具准确率 | RL train/val；先确认reward可靠且候选有非零组内方差 | 同一评测、reward分量/KL/entropy/adv、有效更新、checkpoint | 高：actor+vLLM+reference+KV；是cold-start必要对照，独立于E1 |
| E3 Base→SFT→GRPO | SFT是帮助探索/格式，还是限制泛化 | E1固定checkpoint + 与E2同一RL数据/预算 | 与E1/E2配对比较；同时报告额外SFT成本与样本 | 高；不能只给E3更多RL步数或更长生成预算 |
| E4 Reward Optimization | 对已证实坏例的一个reward修正能否提高独立指标 | 依据validation选择与E2或E3对应的初始化；默认用E0初始化重跑、复用E2为控制；仅改reward且相同步数 | 独立指标+对抗probe、额外调用率、无工具切片；保存reward版本/flags | 高；主预算只容一个主要改动，不做全开关组合搜索 |

六项独立指标定义见 [benchmark方案](benchmark_audit.md)。原始Benchmark官方分数另表；Bamboogle只作为服务费获批后的可选泛化评测，不作为低成本主榜前提。

E4若选择在某个已训练checkpoint上继续，必须同时从该checkpoint做默认reward的等步续训控制；两组采用相同optimizer重置策略，并把控制费用计入55元帽。预算不足时采用上述从E0初始化、复用E2控制的路线，不能把额外训练步数误当reward改进。

## 为什么出现错误不一定需要RL

先对E0/E1按固定标签分析：格式/解析、工具选择、缺/多参数、值/类型、无工具判断、信息缺失需澄清、过长/截断、数据标签错误。

- 若主要是模板/截断/标签错误，先修数据和推理流程，不用RL掩盖。
- 若主要是格式不足，小量SFT或协议适配更直接；若SFT过长/过拟合，先调整目标和训练量。
- 若任务有可自动验证答案、模型采样能覆盖部分正确候选、reward与独立指标排序一致且算力可控，才值得做GRPO。
- 若一题n=4全部同分，则该组advantage接近0；大面积全错/全对说明当前难度/模型/采样设置信息不足，应先诊断而非盲加epoch。
- 若自由回答没有可靠verifier，或错误来自不可获得的环境信息，不强行把训练reward包装成任务成功。

## 小试验规模与停止条件

候选范围：RL先256–512个训练prompt，独立64–128 validation；group n=4、每batch16–32 prompt，actor小mini batch、dynamic token budget从2048–4096试起。长度由tokenizer分布决定，先2048 prompt/512 response的覆盖测试，超长样本隔离而非默默截断；覆盖不足则调整预算并记录数据变化。

这些是性能试探范围，**不是已可执行配置**：当前worker会把actor mini/micro和log_prob micro乘rollout.n，并按DP数整除；设置前必须检查最终有效batch和gradient accumulation，防止误以为micro=1就只有一个completion。1卡优先，不默认2卡。

先10–20有效update验证数值/保存/速度，再决定30–100步探索（依实测与上限）；SFT先10–20步、再1–3epoch的小验证曲线。E2/E3必须覆盖相同唯一prompt与采样数，若用更多epoch弥补小batch应报告。每5–10步用validation检查，按validation的独立EM选择checkpoint；test只在冻结候选后统一跑，不用test挑reward/epoch。

达到阶段费用上限、持续NaN/崩溃、数据/评测不可靠、或reward提高但独立指标明显恶化即停止并记录。停止是决策结果，不隐藏失败run。先保留完整小规模E0–E4，剩余预算才加关键对照的第二seed或3B；单seed/小样本结果仅探索性，不能以微小差异下结论。

## Phase 2 具体顺序及交付

1. **本地**运行并复核Phase1审计证据，追溯源数据元信息；做工具schema类型检查、近重复/历史前缀聚类、SFT/RL/benchmark交叉污染检查。
2. **本地**先去重再按对话/工具族分组，冻结RL train/validation/final-test及SFT train/val的manifest与SHA256；旧test80仅作legacy sanity，不当最终未见测试。
3. **本地**实现独立parser/verifier与逐样本预测schema，使用合成fixture跑通全指标。修API-Bank路径、固定分母/重试策略；BFCL固定版本和注册集成另一个小commit，不执行搜索服务。
4. **本地/只读联网**仅获取模型config/tokenizer以统计长度（届时再做），选择A/B候选环境、列出完整安装锁与镜像；不在Mac装CUDA。
5. **提交具体费用方案供确认**：GPU、整机小时价、最长开机时间、预算帽、模型大小和下载位置。未确认前停在这里，不租机。
6. **获批后**执行10–20步性能/兼容冒烟（不等于正式训练授权），从测量估算正式成本；E0先50–100条，再跑固定完整评测。若开机/服务授权不覆盖后续动作，另提具体方案。
7. 交付 `split_manifest`、数据质量/泄漏报告、verifier与测试、E0 predictions/metrics、环境和吞吐日志、更新预算、Git commit及中文Phase2报告；停止，等待Phase3方案。

阶段之间不自动大规模训练。实验记录统一使用 [experiment_record_template.yaml](experiment_record_template.yaml)，每次运行保存最终解析的Hydra配置、全部reward环境变量、日志与预测hash；大文件放外部持久存储，Git只保存小manifest与摘要。预算记录已消费/本run预计/剩余三列，以AutoDL实际账单校正。
