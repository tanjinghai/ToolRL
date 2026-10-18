# 连续 GPU 实验执行计划（Phase 2 设计，Phase 3 实现）

本文件仅为设计。Phase 2/3 均验收完成前不租 GPU；本轮没有申请实例、启动训练或调用付费 API。原先“分阶段多次租卡”的建议由本计划替代：**CPU 上准备完整工程，获批后尽量在同一实例的一次连续租期执行已批准实验图**。

## 租赁之前必须交付

- Phase 2：冻结数据/manifest/hash、独立 verifier、固定分母评测、Bad Case、真实 tokenizer 长度检查、CPU mock 报告。最终 Test 不用于选择任何训练策略。
- Phase 3：SFT Trainer/messages adapter、GRPO 小显存配置、Reward 独立版本及反例测试、checkpoint 保存/恢复、部署脚本、预算账本和实验调度器；本文件不提前实现调度器。
- CPU dry-run 应完整遍历所有 DAG 分支，包括失败、预算不足、需要人工复核、备份失败、关机失败。未通过验收不能靠开 GPU 现场写主流程。
- 用户提供本项目**历史实际支出**，批准实例型号、租赁时的真实整实例报价、最大本次费用、最大墙钟时长、允许执行的 E0–E4 范围和候选配置。历史支出未知时不能把余额假定为500元。

授权保存为一次运行的不可变 approval manifest：型号、GPU数、实例价格口径、有效期、Git SHA、数据版本、模型revision、允许的实验/候选分支、各阶段费用帽、允许的重试次数、关机方式。连续租期不构成无限追加实验的授权；超出 manifest 的训练一律暂停。

## 一次实例的目录与缓存

```text
/root/autodl-tmp/toolrl/
  repo/                          # 只checkout已审核的固定commit
  env/                           # 一个验证过的训练环境或容器镜像digest
  cache/huggingface/              # E0/E2共用Base；tokenizer revision固定
  data/<dataset_version>/        # 已冻结生成物，首次校验后不重复构建
  runs/<run_id>/
    approval.json budget.json resolved_config.yaml
    logs/ predictions/ metrics/ bad_cases/
    checkpoints/{sft,grpo}/step_*/
    state.json artifacts.json
  backups/                       # 本地临时区，不等于异地持久备份
```

cache 模型按 immutable revision 索引，不同训练模型用独立目录，禁止写回 Base。checkpoint 以 run_id + step 标识；不以“latest”裸路径替代可追溯版本。忽略目录不代表持久存储，AutoDL 实例释放/磁盘故障可能丢失本地盘；重要产物须备份到用户指定的独立存储并核验hash。

## 执行图与每一步验收

| 顺序 | 工作 | 开始条件 / 结果 / 失败动作 |
| --- | --- | --- |
| 1 | 初始化实例 | 必须已有有效批准；记录实际GPU/driver/RAM/磁盘/小时价，型号或价格超授权立即停止 |
| 2 | 同步Git | 仅从origin拉取固定commit；验证工作区/源码hash，禁止临时合并upstream |
| 3 | 环境配置与检查 | 复用预制镜像/锁文件；pip check、CUDA/BF16/FlashAttention/vLLM/Ray/FSDP分项检查；不兼容即在环境费用帽内诊断，超时关机 |
| 4 | 模型与数据缓存 | Base权重只下载一次，按revision/hash完成标记；数据用Phase2冻结版本，禁止GPU开着重新清洗；断点下载不留下可误用的半成品 |
| 5 | GPU冒烟/性能测量 | 2–8条生成、10–20有效updates、最长prompt、导出再加载；记录峰值显存、组内方差、s/update、初始/稳态/验证/保存用时 |
| 6 | E0 Baseline | 按冻结validation/外部benchmark配置评测；保存全部预测，失败计入分母；先确认模板/长度无系统故障 |
| 7 | E1 SFT | 仅获批训练配置；验证assistant-only labels、EOS、有限loss和权重变化；保存完整与最终checkpoint |
| 8 | SFT统一评测 | 同一validation、同一解码配置；与E0配对比较，不能只比训练loss |
| 9 | 自动Bad Case与决策 | 读取已保存预测，无需驻留模型；产出简短decision.json/md，按预登记分支选择后续已获批任务 |
| 10 | E2 Cold Start GRPO | 从同一Base初始化；数据/reward/verifier闸门通过；不依赖SFT成功，不预设SFT优于cold start |
| 11 | E3 SFT→GRPO | 使用固定E1 checkpoint；与E2相同RL数据/有效更新/采样预算；SFT明显退化时按批准规则跳过或仅短诊断 |
| 12 | E4 Reward消融 | 只执行批准的一个主要变体；与E2或E3从相同初始化、同样步数比较，避免把额外训练当reward改进 |
| 13 | 最终评测 | 候选冻结后才解锁final Test；所有模型同一test manifest、decoder预算、独立verifier；官方分数另表 |
| 14 | 备份与产物核验 | 模型/config/tokenizer、日志、预测、metrics、bad cases、环境、费用、RNG/optimizer状态等清单逐项hash；备份回读可恢复 |
| 15 | 费用核对与实例关闭 | 更新实际账单，确认批准的AutoDL停机动作成功；“训练进程结束”或容器内shutdown不等于平台停止计费；保留控制台关机兜底 |

不要因一条分支失败就重复安装环境或重下载Base。按依赖释放GPU模型内存并清理vLLM/Ray子进程，保存共享数据和缓存；后续stage加载自己的模型产物。

## 决策节点：预登记，而非 GPU 开机时临时猜测

以下是 Phase 3 需用 CPU 测试和 validation 校准的候选工程门槛，不是统计显著性的通用定义。正式运行前把阈值写入批准的配置，不能看 final Test 后改。

| 信号 | 建议动作 |
| --- | --- |
| artifact hash变化、split交集非零、GT/schema系统异常 | 停止相关训练；生成异常报告；不能在GPU上重构数据后继续原run |
| E1格式改善明显，工具/参数EM仍有空间 | 用Bad Case定位E2/E3的验证切片；仅在reward可验证且采样有差异时进行GRPO |
| E1相对E0验证EM下降，例如≥3个百分点且配对区间支持下降 | 保留E1产物与结果；优先保留已批准E2；E3只在预授权短诊断范围内运行，否则跳过 |
| 原reward满分但独立EM失败、bool/int等关键对抗probe不通过 | 暂停正式RL，先在CPU核验reward实现；部分分与EM不同是设计意图，不能一概视作失效；满分假阳性是重点阻塞项 |
| 连续观察窗口内>90% prompt组全部同分 | 输出采样/正确率/零adv分析；只能尝试已批准的温度/难度小诊断，不能自动扩到大模型/更多epoch |
| 任一NaN/Inf、loss爆炸、权重同步错误 | 立即停止该训练；保留最早异常batch和日志；不得自动用坏checkpoint覆盖最后可用版本 |
| OOM / 速度不达预算 | 允许一次已批准的小micro/token-budget诊断；影响生成长度/有效batch时必须保持E2/E3对照一致并创建新配置版本 |
| 需要人工选择新实验 | 先写简短决策包和可恢复checkpoint，备份并按批准策略关机；不得为了等回复长期占卡 |

原训练Reward在Phase1发现类型/匹配问题；Phase3必须保留原实现，同时为安全校验或新变体单独版本化。若原版不能通过关键可验证性闸门，不能仅为了完整E0–E4流程强行启动正式RL。研究原版缺陷的对照需明确记录风险与授权范围。

## 预算与最大占用时间

总项目帽约500元，包含历史消费。每次stage开始前计算：

`剩余 = 总帽 - 历史实际支出 - 本租期已发生GPU/存储/API费用 - 已承诺待扣费用`

`预计stage成本 = 初始化/加载/训练/验证/保存/备份的预计墙钟小时 × 实际整实例小时价 + 其他费用`

只有 `预计stage成本×1.3 + 停机/备份储备 ≤ 剩余` 且不超过stage帽/批准范围时才开始；报价按卡时乘卡数，整实例价不能再重复乘。用实测P50/P95步时更新估计；价格、GPU数、模型大小和实验范围变化需新批准。保存余额/费用账本，禁止阶段结束后才发现已经超限。

调度器须同时有费用帽、墙钟超时、外部watchdog和失败次数限制。最后一个正式阶段结束前保留备份/最终评测/停机余量；余额不足优先缩小获批规模或跳过可选消融，不牺牲数据和指标可靠性。API-Bank本地推理没有搜索服务依赖；Bamboogle搜索/judge没有隐含授权，继续保持关闭。

## 状态、恢复与备份

每个stage为 `planned → running → completed/failed/stopped/skipped`，state.json原子更新并记录前置artifact hash。完成标记依赖产物校验；不能看目录存在就算完成。预测按(run_id,sample_id)保存所有终态，续跑不只重做失败题；有意重试另建trial，固定分母。

训练checkpoint需要模型/tokenizer、optimizer、scheduler、RNG、步数/epoch和数据游标；Phase1旧导出只有HF权重，不能声称精确resume。Phase3先实现并测试恢复语义，再做连续GPU运行；如只能重启优化器，明确创建新run并保持控制实验同样重启。

最终备份清单包含Base引用及revision（无需重复上传Base）、所有选定SFT/GRPO模型、最终解析配置、数据/manifest引用、代码SHA、环境锁、日志、完整预测、指标/Bad Case、授权与费用账本。备份失败须在预留窗口内处理；窗口到期先保存可恢复状态并按批准方式停机，不能无上限占卡。

Phase 2 结束后停在CPU准备阶段；Phase 3完成并验收后，再向用户提交一个可审阅的连续租期执行方案，取得实际GPU与费用批准。
