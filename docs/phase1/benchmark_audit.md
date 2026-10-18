# Benchmark 与统一评测方案

基线 `8cee13e`；静态审计，未加载模型、未调用付费服务。本阶段只设计统一接口。

## 仓库实际提供什么

| Benchmark | 本地资源 / 入口 | 依赖与可运行性 |
| --- | --- | --- |
| BFCL | `benchmarks/BFCL/rlla_qwen.py::RLLAHandler` | 只有 handler，无数据、bfcl package、注册配置和完整启动脚本；必须另取官方固定版本并接入注册表 |
| API-Bank | `generate.py`、`evaluate.py`，399/67/131 条 level1/2/3 processed JSON | vLLM 离线生成；无需实时 API 服务，但有 `PATH_TO_YOUR_SCORE_ROOT` 硬编码，依赖当前工作目录；不是原 API-Bank 的完整执行型评测 |
| Bamboogle | `generate.py`、`utils.py`、125 条 QA | vLLM + Serper 实时搜索 + OpenAI judge；需要实际密钥/网络/付费额度；最多4轮，真正追加 observation |

BFCL handler 手工生成 Qwen ChatML，非通用模型模板；`decode_ast` 返回 name→arguments 列表，`decode_execute` 返回调用字符串，由外部 BFCL 框架决定执行和计分。本仓库不能单独算 BFCL 官方准确率。解析时忽略坏行可能把部分损坏输出“修好”，必须另外统计 parse error。

API-Bank 可在后续修正路径后按以下形态运行（**当前还不能直接用占位路径运行**）：

```bash
cd benchmarks/API-Bank
python generate.py --model_paths /absolute/path/to/hf_model
python evaluate.py --model_paths /absolute/path/to/hf_model
```

`generate.py` 保存逐样本 `data/raw_output/thought/tool_calls`；`evaluate.py` 保存 score.json、分 level 和 overall leaderboard。实际评分：answer 若为列表仅取第一个；prediction 中**任意一个**调用与之相等即可通过，额外调用不扣分；无 name/parameters 时还会补成标准答案工具名。故这个分数只能叫“仓库 soft-match”，不能冒称严格工具准确率或官方 API-Bank 执行成功率。

`generate.py` 会跳过已有 score=1 的样本而重做失败样本，Bamboogle 也只跳过已答对记录；重复运行可能产生累积重试偏差，不是固定一次评测。异常分支可能引用尚未赋值的 `assistant_output`，并且原统计基于已有结果，需补齐失败样本分母。

Bamboogle 的调用形态为在该目录执行 `python generate.py --model_paths /absolute/path/to/model`，先 `cp secret.example.json secret.json` 并在本地填写。`utils.py` 在 import 时就读取 serper_key；`base_url` 虽有示例字段但 client 没有使用，不能据此认定代理已配置。脚本只执行输出中第一条搜索，judge 为 gpt-4o，自由文本 Yes/No；结果保存 history/预测/score。搜索结果随时间变化，judge 与 API 费用必须单列。**未经确认不运行此付费流程。**

`verl/trainer/main_eval.py` 只分派 MATH，不能直接评 ToolRL；`RayPPOTrainer._validate` 是 reward 均值，也不能取代独立指标。

## 后续统一设计

统一 `model_artifact → render_prompt → generate_once → predictions.jsonl → independent_verifier → metrics.json`。Base/SFT/GRPO 只替换模型产物，保持 tokenizer、protocol prompt、test manifest、seed、temperature、max_new_tokens 一致；LoRA 记录合并方式。主对照 E0 是未经本项目训练的 **Instruct** 模型，“Base”不是另换 Qwen pretrained base。

至少分开两种评测：①共享 ToolRL 协议，隔离训练阶段的影响；②各模型原生 tool-use 协议作为额外诊断，不能拿两种协议下结果直接排主榜。官方 BFCL 原始指标和本项目诊断指标分表，不混合权重制造一个总分。

每个预测保存稳定 sample_id、dataset hash/split、模型 revision/checkpoint、原始完整 completion、解析对象、parse/schema error、终止原因、token 数/时长、generation config、各项正确性及 evaluator Git SHA。失败/超长/空输出都保留在固定分母；按 `(run_id,sample_id)` 缓存所有终态，重试另建 trial，绝不只保留成功答案。

## 指标定义（新增功能，当前仓库未统一提供）

| 指标 | 统一定义与分母 |
| --- | --- |
| Tool Name Accuracy | 有工具 GT 的样本中，预测名字多重集完全相等比例；重复/额外工具算错 |
| Parameter Name Accuracy | 有工具 GT 的样本中，存在名称一致的一对一匹配且每对 key 集全相等的比例；工具名错误直接失败 |
| Parameter Value Accuracy | 同类样本中名称与 keys 正确，且类型明确、规范化后的所有值全正确的比例；另报条件于名称/keys 正确的诊断值 |
| Exact Match Accuracy | 完整调用结构匹配；parallel 无序、sequential 有序，由任务元数据指定；无工具样本要求预测无工具；不评价自由回答语义 |
| Format Validity | 分别报协议外壳合法率、JSON可解析率、schema合法率；多行调用必须全部合法，不能忽略坏行 |
| No-tool Accuracy | GT 标为无工具的样本里，预测不产生可执行调用的比例；补报需要工具却不调用的漏调用率 |

名称/参数主指标是严格样本级 accuracy，另可报 micro precision/recall/F1，但须写清分母。类型规范化遵循 benchmark 允许规则；不静默把所有值转字符串。对不存在 no-tool 子集的 benchmark 报 N/A，不报 0 或凭空生成标签。

先用独立 CPU verifier 在合成 fixture 上验证：乱序并行、重复工具、额外工具、同名多次、嵌套参数、bool/int、空工具、坏 JSON、截断、缺键。该 verifier 不复用训练 scorer，训练 reward 仅作为诊断列。

## Phase 2 最小评测闭环

先锁定内部 held-out + 全部597条 API-Bank 的数据版本，建立 manifest/无模型 verifier；再做50–100条 E0 冒烟和完整 E0。BFCL 固定官方版本、下载数据、集成 handler 并验证官方命令后再跑，记录版本/题目数量/类别，禁止直接用移动的 latest 数据对照论文。

BFCL 官方资源入口：[Gorilla BFCL](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard)（本阶段未下载、未验证其当前 API）。Bamboogle 延后到主链路稳定且明确批准 API 预算后；可缓存固定 observation 做可复现诊断，但必须标记为离线变体，不冒充实时 benchmark。

统计上给样本数与95%区间、按来源/调用数/参数复杂度切片；后续训练模型使用相同题目的 paired bootstrap。小样本单 seed 微小差距不声称确定提升。
