# Reward 机制审计

依据：`verl/utils/reward_score/rlla.py`、`main_ppo.py::RewardManager`、训练配置及 [论文 v1 §3.3–3.4、§5](https://arxiv.org/html/2504.13958v1)。本轮不修改 Reward；探针数据见 `evidence/audit.json::reward_probes`，不是 benchmark 成绩。

## 默认实际公式

`compute_score` 返回 `(total, format, correctness, length)`。默认 format∈{0,1}，工具 correctness∈[-3,3]，length=0，total∈[-3,4]。奖励写在回答末 token，随后可能减 reference KL。

1. **Format**（`customize_format_reward_func`）：根据标准答案是否含 tool/response 标签选择四类正则，要求 think、换行和单个 tool/response 标签块。它检查文本外壳，**不验证内部 JSON/schema**，不检查推理正确性。
2. **工具名**（`match_score`）：比较工具名多重集，`intersection/(|G|+|P|-intersection)`，保留出现次数、不计顺序；完全相等列表有快速路径。
3. **参数名**：对于同名候选工具，比较参数 key 多重集的同类相似度。
4. **参数值**：每个 GT key 在 prediction 存在且 Python `==` 相等得 1 分；字符串大小写/单位无语义容错，嵌套值整体比较；`True==1`、`1==1.0` 不区分类型。
5. **多工具**（`compute_tool_call_reward`）：按 GT 顺序找同名且尚未使用、参数局部分最高的预测调用，贪心一对一匹配。总分 `s=name_similarity+Σ(param_similarity+correct_value_count)`；分母 `S=1+Σ(1+|GT parameters|)`；correctness=`6*s/S-3`。完整 Python 列表相等直接给 3。
6. **无工具标签**：correctness 恒为 0，response 内容正确与否不检查。错误文本只要符合外壳也能得 total=1，因此不同样本类别的 reward 上限不同。
7. **Length**：仅 `WITHLENGTH=1` 加入，按 think 的空白分词数量线性加分并截到 1；不是 tokenizer token 数。`SCHEDULELENGTH` 单独开并不会启用 length，且调度没有全程上下界裁剪。

## 变体实际行为

| 开关 | 实现和注意点 |
| --- | --- |
| 默认 | 已经对工具名/参数名/参数值分别给部分分，最细分的主设计 |
| `REFINEDREWARD=1` | 名字/keys **列表**严格相等才给该分量 1；名称与原默认相比更严格，并不表示奖励拆得更细；顺序变化会丢分 |
| `INTERMEDIATEREWARD=1` | 同名配对的完整调用 dict 相等才得该调用 1；工具名部分分仍在；分母为 1+GT 调用数 |
| `COARSEREWARD=1` | 整个 GT/prediction **列表**相等才 +3，否则 -3；参数 dict key 顺序不影响 Python dict 相等，但调用列表顺序影响 |
| `CORRECTMAX1=1` | correctness 范围改 [-1,1]，format 仍 [0,1] |
| `MAX1STEP30MAX3=1` | step<30 correctness 除以3；step≥30 format 除以2；不只是改 correctness |
| `SCHEDULEREWARD=1` | format 从 [−2,2] 向 [−1,1]，correctness 从 [−2,2] 向 [−3,3] 调度；与默认 format=[0,1] 的差异需记录 |
| `STRICTMATCH` | 在 wrapper 中设置但 rlla.py 无读取逻辑，当前无效 |

开关可以叠加但没有互斥校验，默认 wrapper 没重置 `INTERMEDIATEREWARD`；必须显式记录/清空所有相关环境变量。不要把运行名称当配置：当前 `compute_score` 用 `EXPERIMENT_NAME` 是否包含小写 qwen/llama 来切 assistant 文本，否则抛异常，甚至原脚本占位名称也不能直接用。

## 已复现的边界行为

默认探针，单调用 GT=`f(x=1)`：

| 预测 | total / format / correctness | 含义 |
| --- | --- | --- |
| 完全正确 | 4 / 1 / 3 | 正常上限 |
| `f(x=True)` | 4 / 1 / 3 | Python 类型宽松相等被当完全正确 |
| 正确调用 + 同名错误调用 | 3 / 1 / 2 | 有惩罚但仍高分，额外调用未充分隔离 |
| 正确值 + 额外 junk 参数 | 3 / 1 / 2 | 正确值可抵消一部分无效参数成本 |
| `f(x=2)` | 2 / 1 / 1 | 名字/keys 正确产生部分分，是设计意图，不是自动视为 bug |
| 标签块内部非 JSON | −2 / 1 / −3 | format=1 不能作为 JSON 有效率 |
| 无工具 GT，回答内容错误但外壳正确 | 1 / 1 / 0 | 无工具 correctness 缺失 |

另一个含两个同名工具的例子，仅交换 GT 顺序，correctness 从 **0.428571** 变为 **2.142857**；原因为局部贪心匹配，不能保证全局最优。coarse 对相同调用集合重排给 −3，refined 对名字列表重排给 0。

## 论文与代码边界

- 论文 §3.3 写“optimal matching”，源码是贪心，存在顺序依赖。
- 论文 §3.4 声称省略 reference KL，当前默认脚本 reward 侧仍有 KL=0.001；显式 KL loss 关闭并未关闭它。
- 论文使用工具集合的表述；代码部分路径按列表严格相等，parallel call 的集合等价性不成立。
- 这些是公开版本差异，不能据此推断作者全部内部实验使用了哪一份代码。

## 可优化方向及优先级（后续实验，不在本轮落地）

1. **先修可验证性**：统一严格解析与 schema/type 校验，明确 bool/int/float、None/no-tool 的等价规则；错误 JSON 与协议外壳分开记录。先建有预期答案的 adversarial cases，避免只看训练 reward 上升。
2. **匹配与额外调用**：parallel 采用按工具名约束的全局最大权匹配；sequential 保留顺序，不能一概排序。显式控制多余调用/参数、重复调用，防止枚举候选获得高分。
3. **无工具**：区分不需要工具、工具不适用、缺少必填参数；增加可验证的 no-tool decision 奖励。一般自由回答语义无可靠规则时不强行加 correctness，不用付费 judge 随意补齐。
4. **长度**：先默认关闭；若坏例说明推理不足，再做长度/效率单因素实验。无条件奖励长 think 有灌水风险，不能把长输出等同于更强推理。
5. **协议解耦**：只传 completion 给 scorer，不依赖实验名称猜模型格式；错误 GT 应预审而非训练时裸 `json.loads` 崩溃。减少每条完整输出日志，降低 I/O 和费用。

E4 每次只变一个主要因素，共用独立 verifier；先以反例和离线预测检查新 reward 排序，再决定是否值得额外 GRPO。
