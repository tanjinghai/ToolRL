# 统一评测、Verifier 与 Bad Case 使用说明

主入口 `python -m toolrl_pipeline`。CPU路径不导入verl/torch/vLLM；`--backend vllm`是显式的未来GPU适配器，Phase2未执行真实推理。预测生成与评分分离，已保存预测可在Mac离线重复分析。

## 从冻结数据跑CPU Mock

以下从仓库根目录执行。目录名必须新建；有意恢复同一次生成用`--resume`，不会只重做失败题。

```bash
python -m toolrl_pipeline generate \
  --dataset dataset/generated/phase2-v1/eval/validation.jsonl \
  --output-dir runs/my-mock --run-id my-mock \
  --model-id mock --model-revision synthetic-v1 \
  --backend mock --mock-mode mixed

python -m toolrl_pipeline score \
  --dataset dataset/generated/phase2-v1/eval/validation.jsonl \
  --predictions runs/my-mock/predictions.jsonl \
  --output-dir runs/my-mock-rescored
```

`oracle` mock把预制GT作为模拟输出，目的是检查计分器可达100%，不是模型能力。真实backend只接收sample_id/prompt/tools，单元测试断言GT不进入请求。Mock标识必须与backend一致，报告显著显示is_mock；不会把CPU mock解释成Baseline。

未来仅在已批准GPU阶段显式使用`--backend vllm --model-path /已有本地模型 --model-revision /固定revision或checkpoint标识`，不自动下载权重。`--max-model-len`（当前候选6144）、`--model-dtype`、TP与显存占用率单独记录为backend_config，必须经过显存/长prompt冒烟。不能因为默认值存在就认为GPU已支持。采样参数由 [evaluation.json](../../configs/evaluation.json)统一控制：temperature0、top_p1、max_new_tokens1024、seed31415、n1。

## 输入输出契约

每条prediction JSONL包含：schema version、run_id、sample_id、数据版本/文件hash、model id/revision、generation_config、backend_config、backend/is_mock、evaluator Git SHA及代码hash、raw_output、status、finish_reason、elapsed；实际backend还记录token数。

相同run按sample_id只允许一个终态。重复ID、未知ID、数据hash/version不一致、混合模型/解码/后端配置会明确失败；缺失预测补为missing，保留固定分母。生成异常写error；length/max_tokens终止写truncated，默认结构指标也判失败。中断留下半行JSON时解析报错，不能擅自丢行并重选成功答案；Phase3恢复器应保留原件并记录修复trial。

输出包括：

| 文件 | 内容 |
| --- | --- |
| run.json / predictions.jsonl | 生成配置、完整原始输出、失败记录；无自动“只重试失败题” |
| scored.jsonl | 每题GT、prediction、解析对象、各指标、错误和定位详情 |
| metrics.json | 分子/分母/比例、缺失/异常、复杂度/来源level切片；显式valid-GT子集 |
| bad_cases.md | 每种错误数量/比例、复杂度表、每类最多2个典型失败；不是训练建议自动授权 |
| manual_review.jsonl | GT/schema不可验证或不支持的异常，供人工核验 |

## 指标口径

独立实现位于`toolrl_pipeline/verifier.py`，**不导入或调用原训练Reward**。

| 指标 | 分母与判定 |
| --- | --- |
| Format Validity | 所有预定样本；protocol + 全部JSON + shape/schema有效，且无生成错误/截断；另列protocol/json/schema三项 |
| Tool Name Accuracy | 有工具GT；名字多重集全相等（保留调用次数）；名字指标本身不要求调用顺序 |
| Parameter Name Accuracy | 有工具GT；名称一致的一对一匹配中各组key集合全相等；不因值类型错误抹掉独立key指标 |
| Parameter Value Accuracy | 有工具GT；key、类型及全部值匹配；parallel允许全局一对一重排，sequential按位置匹配 |
| Exact Match Accuracy | 所有预定样本；完整调用结构精确匹配，no-tool要求合法response且无调用；不评自由文本语义 |
| No-tool Accuracy | 无工具GT子集；合法无调用；子集为空输出null/N/A，不伪造0或100% |

bool/int严格区分；integer/字符串数值不混同。仅明确number schema允许1与1.0等价。dict键顺序无关、array顺序有关；未知工具、多余调用/参数、重复调用不能靠“包含一个正确答案”通过。同名多调用使用全局匹配，不受贪心GT顺序影响；未知多调用顺序标invalid_ground_truth，不能默认忽略。

思考中的标签字面量可存在；多个外层tool/response块、嵌套外层协议、坏行、重复JSON key、NaN等失败，不丢坏行“修好”预测。格式检查采用明确协议与schema子集，不是任意JSON Schema全集或工具执行沙箱。

JSON/schema合法不等于任务完成。`free_text_semantic_accuracy`及`task_execution_success_rate`固定null；GT有问题标invalid_ground_truth，主表仍保留该题失败分母，并**另外**给valid_ground_truth_subset，不能悄悄删题提高主分数。

## 错误分类规则

错误是多标签，比例分母统一为预定题数，比例之和可以超过100%。

- format_error：协议/JSON/shape/schema错误，或生成失败/截断。
- wrong_tool：未知名称或同时缺少GT工具且出现其他工具；missing_tool / extra_tool按名称多重集差计算。
- unnecessary_tool：无工具GT却产生调用，同时也可标extra_tool。
- missing_argument / extra_argument：同名最佳一对一配对中相对GT的key差；声明required缺失/声明外key亦报告。
- wrong_argument_type：违反声明类型；wrong_argument_value：类型允许但相对GT值错误。多个参数可同时触发两类。
- truncated_output：finish_reason为length/max_tokens；invalid_ground_truth：标签/schema错误、unsupported schema或多调用顺序未知。
- wrong_call_order：同一调用集合在顺序任务中重排；避免把单纯次序错误笼统解释为工具选择错。

切片包括no_tool/single/multi、调用数0/1/2–4/5+、参数数0/1–2/3–5/6+、API-Bank level。配对匹配和诊断配对均处理调用重复，最多64条预测调用，超限判格式失败。

## 比较上一次实验

```bash
python -m toolrl_pipeline score \
  --dataset dataset/generated/phase2-v1/eval/validation.jsonl \
  --predictions runs/E1/predictions.jsonl --output-dir runs/E1-analysis \
  --previous-results runs/E0/scored.jsonl
```

要求相同完整ID、dataset版本/hash、采样和backend配置；输出修复/退化ID、错误标签变化、各类错误数量差与指标差。它只能描述记录的差异，不能自动证明因果或统计显著性。最终Test生成与评分需要显式`--allow-final-test`；开发期用validation，不解锁final Test做reward/checkpoint选择。
