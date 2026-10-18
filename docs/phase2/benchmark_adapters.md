# Benchmark 适配与固定版本

## API-Bank：已完成CPU统一入口

原`generate.py/evaluate.py`替换为薄入口，不再依赖当前目录或`PATH_TO_YOUR_SCORE_ROOT`；实际实现共用toolrl_pipeline。历史soft-match逻辑保留为独立命名`legacy_api_bank_soft_match`对照，不能当严格工具匹配，更不是原API-Bank完整工具执行成功率。

```bash
# 在任意目录均可用绝对脚本路径；示例从仓库根目录执行
python benchmarks/API-Bank/generate.py \
  --output-dir runs/api-mock --run-id api-mock \
  --model-id mock --model-revision synthetic-v1 --backend mock

python benchmarks/API-Bank/evaluate.py \
  --predictions runs/api-mock/predictions.jsonl --output-dir runs/api-rescored
```

省略dataset时从本地固定三个processed JSON构建确定性输入；共597题（399/67/131），数据版本同时绑定原始hash、adapter和schema代码hash。可显式传`--dataset`复用已冻结的`dataset/generated/phase2-v1/benchmarks/api_bank.jsonl`。不再按TinyZero路径切模型名拼目录，run_id/model revision必须明确。

生成一题只写一次，异常也保留，resume跳过所有已记录终态；失败/缺失始终计入597分母。严格EM要求完整调用结构；soft-match仍模拟原“任意一条正确即可、额外调用不扣分、缺名字可借GT名称、Python宽松相等”的行为，并有边界fixture，两个分数同时给出。

**新发现**：118题原GT与自身工具schema存在冲突或无法验证（典型：schema为list而GT是字符串形式list、数值类型不符、声明外参数/工具）。没有猜修或删题；主strict分母仍为597、这些题标invalid_ground_truth，另给479题valid-GT子集。该子集也不是官方API-Bank分数。人工复核文件包含逐题原因，不能把GT异常导致的损失当作模型退化。

API-Bank CPU mock与异目录CLI已实测；vLLM真模型路径仅实现lazy adapter，GPU尚未验证。

## BFCL：固定快照、问题污染检查与严格桥接

固定 [Gorilla提交a84eb805fe8d66635a8e402a78783f13044decba](https://github.com/ShishirPatil/gorilla/tree/a84eb805fe8d66635a8e402a78783f13044decba/berkeley-function-call-leaderboard)，不用latest。选该2025-04-24快照是接近ToolRL发布时期且可验证旧OSS handler接口，不声称等于论文作者未发布的评测环境。

`configs/bfcl_snapshot.json`列出17个顶层BFCL_v3问题文件（9,504,006 bytes）的Git blob、大小和commit。`fetch_assets.py --bfcl`只读下载并逐文件校验Git blob SHA1与SHA256，缓存到data_local，不重新clone ToolRL，也不取模型/工具执行数据库。此次提取6842条user消息做交叉检查，无训练关联组命中；这只覆盖所列快照。

已静态验证固定OSSHandler的`_format_prompt(self,messages,function)`、`decode_ast(self,result,language)`、`decode_execute(self,result)`签名，并记录源文件hash。`bfcl_decode_ast / bfcl_decode_execute`通过CPU fixtures：严格ToolRL输出转换成BFCL调用对象/字符串，任何坏行都会报错，不沿用原handler静默跳行的行为。

Phase3集成路线：在单独固定BFCL环境复制原RLLAHandler作为prompt适配，继承覆盖两个decode方法调用上述strict bridge；按该快照README注册`bfcl/model_handler/handler_map.py`与`bfcl/constants/model_metadata.py`。先在其官方命令`bfcl generate --model <注册名> --test-category <固定类别>`和`bfcl evaluate ...`验证小样本，再保留官方AST/执行/multi-turn分数。没有安装/导入完整BFCL包，没有执行官方计分或声称当前桥接已是官方评测通过。

## Bamboogle：本轮仅保留计划

本地125题纳入文字污染检查；未运行generate.py、搜索工具或OpenAI judge，未使用任何密钥。未来应在独立批准的服务预算下固定搜索缓存/时间、judge版本、轮数和重试规则；实时检索分数和离线缓存诊断分开。它不能混进当前六项结构指标主榜冒充任务成功率。
