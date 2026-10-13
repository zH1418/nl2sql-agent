# NL2SQL v2 量测报告 · 2026-10-08T11:40:58

- **链路**：v2-create-agent（create_agent 探针闭环）
- **模型**：`ark-code-latest`
- **配置**：初始预算 `5` · 行数上限 `1000` · 候选过滤 `False`
- **数据规模**：meta 4 表 / dw 5 表（246 行）

## 汇总

| 指标 | 值 |
|---|---|
| 问题数 | 10 |
| ★ **命中**（出数据且**值 + 行数**均与参考一致） | **10**（命中率 **100.0%**） |
| （参考）行数一致 / 值一致 | 10 / 10 |
| 语义漂移（出数据但与参考不一致） | 0 |
| 主动澄清 / 消歧（`ask_user`，**合法结果**） | 0 |
| 失败（无数据且未问用户） | 0 |
| 出数率（含漂移） | 100.0% |
| 工具调用总数 | 21（`build_and_execute` 11 / `ask_user` 0） |
| 模型调用总数（AI 消息） | 31 |
| 平均每问工具调用 | 2.1 |
| 平均每问模型调用 | 3.1 |
| 预算消耗（min/avg/max，初始 5） | 1 / 1.2 / 3 |
| ★ 未被惩罚的查询数（消耗 ≤1 ⇒ 正常路径净 0） | 9 / 10 |
| 时延（min/avg/max，秒） | 4.94 / 6.09 / 8.62 |
| 召回规模（平均 表/列/指标） | 2.2 / 9.8 / 1.9 |

> ⚠️ **结果具有随机性**：LLM 的分支决策（是否追问、选哪个指标、重试几次）每次可能不同。
> **稳定可复现的是确定性计数**（工具调用次数、预算消耗、命中行数）；
> 单次采样的成功率会波动，若要报告成功率应多次采样取均值。

## 逐题明细

| # | 问题 | 结果 | 行数 | 参考 | ★值一致 | 工具序列 | 模型调用 | 预算 | 时延(s) |
|---|---|---|---|---|---|---|---|---|---|
| q01 | 统计华北地区各品类的销售额 | ✅ 命中 | 6 | 6 | ✅ | recall_candidates→build_and_execute | 3 | 5→4 | 6.57 |
| q02 | 各大区的销售总额是多少 | ✅ 命中 | 5 | 5 | ✅ | recall_candidates→build_and_execute | 3 | 5→4 | 4.94 |
| q03 | 各品类的销售总额 | ✅ 命中 | 6 | 6 | ✅ | recall_candidates→build_and_execute | 3 | 5→4 | 4.98 |
| q04 | 销售额最高的前3个品类 | ✅ 命中 | 3 | 3 | ✅ | recall_candidates→build_and_execute | 3 | 5→4 | 5.11 |
| q05 | 各品类的平均订单金额 | ✅ 命中 | 6 | 6 | ✅ | recall_candidates→build_and_execute | 3 | 5→4 | 5.45 |
| q06 | 华东和华南两个大区的销售额对比 | ✅ 命中 | 2 | 2 | ✅ | recall_candidates→build_and_execute | 3 | 5→4 | 6.91 |
| q07 | 各会员等级的销售额 | ✅ 命中 | 4 | 4 | ✅ | recall_candidates→build_and_execute | 3 | 5→4 | 6.17 |
| q08 | 各商品品牌的销售额排名 | ✅ 命中 | 15 | 15 | ✅ | recall_candidates→build_and_execute | 3 | 5→4 | 6.03 |
| q09 | 各省份的商品销量 | ✅ 命中 | 6 | 6 | ✅ | recall_candidates→build_and_execute→build_and_execute | 4 | 5→2 | 8.62 |
| q10 | 全国销售总额是多少 | ✅ 命中 | 1 | 1 | ✅ | recall_candidates→build_and_execute | 3 | 5→4 | 6.08 |

## 失败归因分布

| 归因 | 次数 |
|---|---|
| `select` | 1 |

## 预算锁（失败率）

- 总运行 **0** 次 · 失败 **0** 次 · 失败率 **0.00%**
- 告警阈值：5% / 10%（最小样本 20）
- 已触发告警：无

## 产物用途

- **RAGAS 评测**：`records[].question` + `generated_sql` + `rows` 与 `reference_sql` + `reference_rows` 做**执行结果比对**（不是比 SQL 文本）。
- **回归基线**：改动后用同一脚本重跑，对比 `aggregate` 的确定性计数。

## 量测期间的发现（★ 首轮量测暴露的数据缺陷，已修）

**AOV 指标的 `relevant_columns` 指错了列**：

| 项 | 值 |
|---|---|
| 缺陷 | `conf/meta_config.yaml` 里 `AOV.relevant_columns = [fact_order.order_quantity]`，但 `description` 写的是「所有订单的**成交金额**平均值」、alias 是「平均订单金额」 |
| 后果 | 问「各品类的平均订单金额」时，模型按元数据取了 **`order_quantity`** ⇒ 算出「平均件数」（手机数码 1.0）而非「平均金额」（6716.24），且**反复重试至预算耗尽** |
| 修复 | `conf/meta_config.yaml` + `meta.metric_info` + `meta.column_metric` 三处同步改为 `fact_order.order_amount` |
| 验证 | 修复后该题命中（6 行，与参考 SQL 一致） |

> ★ **这条发现本身就是「离线评测有价值」的证据**：在线链路**全程绿灯**（qualify 通过、有数据返回），错的是**元数据**，只有「与参考 SQL 做结果比对」才暴露得出来。
