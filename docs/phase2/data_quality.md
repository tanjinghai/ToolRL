# 数据质量、隔离与冻结策略

版本：`phase2-v1-13c266c7a084`；seed：31415。机器结果见 [data_quality.json](evidence/data_quality.json)，逐条决策见 [RL](evidence/decisions/rl.jsonl) / [SFT](evidence/decisions/sft.jsonl)。完整原文、lineage、quarantine、group_edges 位于本地 `dataset/generated/phase2-v1/`，Git保存小型manifest和证据，不重复上传42MB派生数据。

## 清洗结果

| 来源 | 原始 | 保留 | 确定修复 | 精确重复删除 | 隔离 |
| --- | ---: | ---: | ---: | ---: | ---: |
| RL | 4000 | 3125 | 0 | 9 | 866 |
| SFT | 400 | 319 | 0 | 0 | 81 |
| 合计 | 4400 | 3444 | 0 | 9 | 947 |

原始JSON和旧Parquet未修改。sample_id=`source + SHA256(canonical raw record)`，跨构建稳定；精确重复的不同出现位置共用内容ID，但用source/raw_index单独记录每次决定。version.json记录完整原始文件SHA256、转换代码hash、benchmark快照hash、随机种子、全部产物和manifest hash。

保留只表示协议/声明字段的结构检查通过，不声称外部工具执行成功或原始标签语义已获独立认证。release缺少逐样本ToolACE/Hammer/xLAM来源ID，继续记录unknown，不能根据名称猜出来源。

## 修复与隔离规则

`prepare_record` 不进行字符串数字→数值、缺省补参、模糊改工具名或猜测嵌套参数。实现的确定修复仅包括：唯一能精确对应声明名称的前后空白修正，以及完全相同工具定义的metadata去重；真实数据没有符合该规则而被修复的样本，不能把quarantine伪装为清洗成功。

隔离947条中，809条归为需人工核验，138条为缺少原始来源无法可靠确认。主要reason如下，允许一条样本有多个reason：

| reason | 涉及出现次数 | 处理 |
| --- | ---: | --- |
| wrong_argument_type | 809 | 保留原值隔离；不把字符串化list/数字自动转换成新GT |
| ambiguous_multi_call_order | 203 | 当前问题有then/after等次序线索，不能确认同轮parallel标签语义，等待人工复核 |
| extra_argument | 120 | 声明之外的key，部分为损坏的嵌套片段；不能猜拆 |
| wrong_tool | 13 | 工具不在清单、截断名称等，等待源数据恢复 |
| unsupported_schema | 6 | 例如Callable/set无法在当前JSON接口可靠验证 |
| conflicting_duplicate_tool_name | 4 | 同名不同声明，不能擅自选一个 |
| unparseable_target | 1 | 含空工具名称等损坏调用结构 |

类型转换器支持JSON类型和明确Python标注别名，如str/int/float、List、Tuple、Union、Optional、list(str)。JSON number允许int/float数值等价；integer拒绝float/bool；所有类型拒绝字符串数值的隐式转换。源schema常未提供required，系统不臆造必填清单；明确required会校验，GT/预测之间少参数也会单列错误。

## 先分组，再划分

在所有原始RL/SFT记录及外部benchmark问题上构建连通组，使用：精确instruction+input、归一化完整input、≥24字符的共同user历史、完整input的≥64字符前缀、含obs的历史前缀、当前问题的3词shingle Jaccard≥0.90（至少8词）。保留数字/实体，不把不同城市/参数全部模板化。近重复候选中过于常见（>200行）的shingle不单独生成二次方配对，限制见机器报告；这不是全语义去重证明。

跨来源保留组统一锚定train，共48组；其余组按来源/任务类型和组大小、固定seed分配，目标80/10/10。组不能拆开，故实际比例允许偏离。**工具定义相同本身不构成重复请求**：统计schema重复，但不将所有使用同一工具的样本误合并；本轮也不声称tool-disjoint泛化。

原37个SFT/RL精确重叠prompt中，27个双侧通过质量检查并同时保留在train；另外10个SFT因独立质量问题隔离，不是因为发生重叠而删除。近重复/历史关联也一起锚定，避免保留重叠训练样本时污染评测。

| 来源 | Train | Validation | Test |
| --- | ---: | ---: | ---: |
| RL | 2511 | 307 | 307 |
| SFT | 245 | 37 | 37 |
| 公共评测（held-out合并后按prompt去重） | 不作为训练入口 | 344 | 344 |

公共Validation为no-tool52、single160、multi132；公共Test仅做静态分布/完整性记录，不生成预测或用于选配置。所有E0–E4共用这个Test。SFT test37永不喂给SFT；SFT validation37也不进入RL train。

清洗影响分布：RL no-tool占比由12.05%升至15.42%，multi由42.38%降至38.59%；SFT no-tool由13.5%升至16.93%。结构复杂/顺序含糊样本更易隔离，因此不能把新数据上的分数直接和论文旧分数比较。未来恢复样本必须新建版本并保留旧final Test。

## 外部交叉检查与边界

已对本地API-Bank597题、Bamboogle125题，以及固定BFCL提交 `a84eb805...` 的17个问题文件所含6842条user消息检查精确/历史/上述词法近重复；未发现需隔离的训练关联组。保留组在train/validation/test之间交集为0，精确prompt交集也为0。

这解决了**本轮检测口径内及Phase1已知的泄漏**，不能证明不存在改写语义重叠、未发布源数据交叉、BFCL其他版本或基座预训练污染。BFCL此次只获取问题快照，没有运行官方评分/执行环境。依赖标记parallel来自原系统协议的simultaneous说明；次序线索会隔离，但隐含工具依赖仍需源数据/人工核验，不能将其当作完整Agent trajectory监督。

## 格式与长度

- SFT JSONL：`prompt=[system,user]`、`target_output=完整原assistant输出`，含sample_id/group_id/version/split/source/tools/provenance。Parquet为`prompt`消息列表、`response`文本及版本字段。think未删除，包括思考中标签字面量；解析器先隔离think，不把其中标签当动作。
- GRPO Parquet：`data_source=rlla_phase2`、`ability=tool_use`、`prompt`、`reward_model={style:rule,ground_truth:target_output}`、available_tools JSON字符串和ID/version/source/split/extra_info。原RLHFDataset只把prompt渲染给模型，GT字段与输入分离；逐条测试与原始instruction/input相等。
- SFT的`encode_sft`按模型原生模板仅渲染一次，prompt labels=-100，原assistant目标和一个EOS参与监督；超长抛错，不默默截断。完整Trainer接入仍属于Phase3，不能直接把新Parquet交给有已知问题的旧SFTDataset。
- 真实Qwen tokenizer验证覆盖282条SFT train/validation，未读取final Test；详见 [tokenizer证据](evidence/tokenizer_validation.json)。RL train有14条prompt>2048、1条>4096（最长4293）；SFT train有10条总长>2048（最长2706），validation另2条。**Phase3必须做长度前置检查/分桶或有记录的train子集，禁止用旧默认左截断丢工具定义。** 新推理入口显式记录context上限，GPU可承受配置仍待实测。

无think SFT值得作为“推理长度/成本”的独立消融，但本轮不构造替代主目标、更不声称已经完成Bad Case驱动的数据优化。未来`select_training_subset`只接受train ID，传入held-out或未知ID即拒绝。
