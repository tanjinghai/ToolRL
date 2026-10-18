# 数据审计：现状、污染风险与清洗方案

审计基线 `8cee13e`。机器可复核证据：[`evidence/audit.json`](evidence/audit.json)，由 `scripts/phase1/audit.py` 只读生成，含文件SHA256、版本、零基索引和重叠prompt hash。未重新生成/改写任何训练数据，未下载模型/tokenizer。

## 来源、格式和划分

README 声称 RL 混合2K ToolACE、1K Hammer(masked)、1K xLAM；现有原始文件没有source_id、原始split、conversation_id或生成记录，**只能确认总条数，无法逐条验证该比例或跨来源重叠**。后续应恢复源数据ID/许可证和清洗变换，不以工具名形态猜测来源后当事实。

| 文件 | 实测条数 | 结构 |
| --- | ---: | --- |
| `dataset/rlla_4k_raw/rlla_rl.json` | 4000 | instruction/input/output 三字段，think只有2种模板 |
| `dataset/rlla_4k_raw/rlla_sft.json` | 400 | 同字段，400种think；README称R1蒸馏，无逐样本来源证据 |
| `dataset/rlla_4k/train.parquet` | 3920 | data_source、prompt、ability、reward_model、extra_info |
| `dataset/rlla_4k/test.parquet` | 80 | 同上 |

`rlla_4k_raw/rlla.py` 对RL4000按 numpy seed31415打乱后取末2%为test。已在内存重算索引，现有两个Parquet的**顺序与全部记录都一致**；prompt中的system/user、reward ground_truth和extra_info都与raw逐条一致。所有data_source为rlla、ability为math，后者是继承模板标签，不代表数据是数学任务；原文件docstring也误写GSM8k。

## 工具分布与复杂度

以下对output先隔离think再解析调用，避免think里的标签字面量干扰；“无工具”指回答主体无tool_call，不包括已知的None工具名规范化策略。

| 指标 | RL4000 | SFT400 |
| --- | ---: | ---: |
| 无工具/response样本 | 482（12.05%） | 54（13.5%） |
| 单调用样本 | 1823 | 187 |
| 多调用样本 | 1695 | 159 |
| 调用总数 | 6104 | 576 |
| 最大单样本调用数 | 9 | 5 |
| 最大单调用参数数 | 11 | 7 |
| input含静态obs | 391 | 37 |
| think空白词数中位/最大 | 15 / 15 | 222 / 397 |
| prompt字符中位 / P95 / 最大 | 2934 / 5457 / 13358 | 2803 / 5720 / 9638 |
| output字符中位 / P95 / 最大 | 263 / 647 / 3177 | 1562 / 2576 / 4931 |

输入包含多轮上下文不等于在线rollout执行工具。参数包括字符串、整数、浮点、布尔、list、dict和null；schema与value嵌套/超长字符串增加解析和显存风险。字符和空格词数不能换算成精确Qwen token长度，后续必须用固定tokenizer复测。

## 已验证质量问题

1. **RL重复**：完全相同record/prompt有8个重复组，冗余9条（3991个唯一）；归一化input冗余30条，末user query冗余70条。这两种较弱键只是候选近重复，工具清单/上下文不同可能合理。相同调用标签尤其无工具标签不算重复数据。
2. **标签不在工具清单**：RL13条、SFT1条；包括工具名括号后缀被截断、尾空格、空名称、疑似SQL片段被误拆成工具。例RL零基25的label名称缺少instruction中完整括号后缀；572含空工具名；SFT130含SQL片段名称。现统计是严格名称匹配，需人工核验，不能自动模糊修正后直接认为标签正确。
3. **参数未出现在声明中**：RL169个调用、112条样本；SFT11个调用、8条样本。部分key含长片段/逗号/引号，疑似上游结构化转换破坏了嵌套参数，而非普通拼写错误。需回溯原始来源核验。该检查只比较声明key，不声称已完成所有required/type/enum检查。
4. **JSON主体**：隔离think后的所有tool_call行可JSON解析，调用基本schema name:str/parameters:dict通过；“JSON可解析”不等于语义和参数schema正确。
5. **SFT协议碰撞**：17条think提及字面量tool_call/response标签；原reward的split从第一个开标签截取，17条会解析失败。实际回答主体JSON合法，因此应区分解析器兼容性问题与数据语法错误。训练蒸馏模型还可能学会在think复述协议；修复/隔离需验证，不能仅改parser就忽略风险。

## SFT/RL交叉重叠：不能直接相信互不相交

| 两份数据 | 完全相同instruction+input的唯一prompt交集 | 归一化input交集 | 末user query交集 |
| --- | ---: | ---: | ---: |
| RL train / test | 0 | 0 | 2 |
| SFT / 全部RL | **37** | 40 | 44 |
| SFT / RL train | 36 | 39 | 43 |
| SFT / RL test | **1** | 1 | 1 |

这是prompt层面的重叠，SFT/RL的output think通常不同，不能靠整条JSON去重发现。论文附录称SFT400与RL4k内容不同，但本地公开文件存在上述精确prompt交集；公开数据与论文描述有差异，不能推断作者内部实验具体split。

因此 **SFT→GRPO若拿当前test80直接做最终测试，至少有一题已在SFT训练候选中出现**（SFT零基索引393、test零基索引42，prompt hash见证据）。此外test太小、按行随机拆分、不按对话/近重复分组，且在训练期间反复用于checkpoint选择；原划分适合legacy sanity，不适合可信最终报告。

本地API-Bank三个level与RL+SFT的归一化完整input/末query交集均为0；Bamboogle问题交集为0。**不等于证明无污染**：没有BFCL数据，没有embedding/改写/对话前缀/跨来源原始ID审计，也无法排除基座预训练污染。

## 后续清洗与版本化方案

1. 保留所有原文件与hash；构建独立manifest，sample_id基于canonical原始样本hash，并补source原ID、conversation_id、teacher/data构造版本、tool_schema_hash、清洗理由。
2. 先在RL/SFT/benchmark全局检查精确prompt、归一化query、历史前缀和近重复；同对话/改写簇放同一split，benchmark污染候选优先移出训练，并保存排除清单。只改变空白/工具后缀的规则须人工抽查，避免合并不同任务。
3. schema检查覆盖工具名、required/optional、未知key、类型/enum、嵌套字段、同名工具、空调用/No-tool。对疑似结构转换损坏样本先隔离；来源可追溯才能修复，否则不自动“猜正确答案”。
4. SFT解析先分think与回答，禁止标签字面量混淆；保留完整蒸馏与简化目标两个版本的变换记录，不能悄悄改成新的主训练目标。
5. 在去重/分组后冻结RL train/validation/final-test（候选约80/10/10，按工具数/无工具/来源分层）；若近重复组使比例变化，以隔离可靠性优先。SFT另划train/val，清除与所有测试的交集，具体条数在清洗后确定。
6. tokenizer长度审计后决定max_prompt/response；显式记录超长隔离率、截断数和类别覆盖，禁止只截掉工具清单。保留legacy3920/80作为对照文件，不覆盖它们。

Phase2输出数据质量摘要、逐项排除理由、split manifest与数据版本hash，再开始Baseline。来源恢复/模糊去重/完整类型校验和GPU上的真实token长度均未在Phase1完成。
