"""核心纯函数与记账逻辑的 pytest 回归（**全部离线，不需要 LLM / VM / 网络**）。

验收判据中「可离线判定」的部分都在这里：

| 分组 | 覆盖判据 |
|---|---|
| `TestAstBuilderReasonClass` | 归因细分（`select` / `structure` / `backbone`）+ 正常路径 |
| `TestJudgeVerdict` | L1 探针判定口径（阈值只作用于向量路；歧义只看指标） |
| `TestControlMiddleware` | ★ 三层记账（净 0 / 净 −2）+ **budget=5 ⇒ 3 次机会** + 三终止条件 |
| `TestFailureLedger` | ★ 预算锁（失败率记账 + 5%/10% 告警 + 最小样本 + 告警去重） |
| `TestContentToText` | 内容块列表归一化（新端点的兼容点） |
| `TestClarifyHelpers` | 澄清拼装（跳空值、无残留标点）+ `ask_user` 表单归一化 |
| `TestSqlflowHelpers` | LIMIT 改写（不重复拼接）+ 选词摘要 |

跑法：
    PYTHONPATH=. .venv/Scripts/python.exe -m pytest tests/ -q
"""
from __future__ import annotations

import pytest
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from app.agent.agent.budget_lock import MIN_SAMPLE, FailureLedger
from app.agent.agent.middleware.control import (
    REASON_BUDGET_EXHAUSTED,
    ControlMiddleware,
)
from app.agent.agent.sqlflow import with_limit
from app.agent.agent.state import INITIAL_BUDGET
from app.agent.agent.tools.ask_user import _normalize
from app.agent.agent.tools.build_and_execute import _summarize
from app.agent.ast_builder import ASTBuildError, build_sql
from app.agent.recall import judge_verdict
from app.agent.utils.llm_utils import content_to_text
from app.services.query_service import _assemble_clarified_query

# ============================================================ 共用夹具

TABLES = [
    {
        "name": "fact_order", "role": "fact", "description": "订单事实表",
        "columns": [
            {"name": "order_id", "type": "varchar", "role": "primary_key", "examples": [], "description": "", "alias": []},
            {"name": "region_id", "type": "varchar", "role": "foreign_key", "examples": [], "description": "", "alias": []},
            {"name": "order_quantity", "type": "int", "role": "measure", "examples": [1], "description": "", "alias": ["销量"]},
            {"name": "order_amount", "type": "float", "role": "measure", "examples": [899.0], "description": "", "alias": ["销售额"]},
        ],
    },
    {
        "name": "dim_region", "role": "dim", "description": "地区维度表",
        "columns": [
            {"name": "region_id", "type": "varchar", "role": "primary_key", "examples": [], "description": "", "alias": []},
            {"name": "region_name", "type": "varchar", "role": "dimension", "examples": ["华北"], "description": "", "alias": ["大区"]},
        ],
    },
]
METRICS = [
    {"name": "GMV", "description": "成交总额", "relevant_columns": ["fact_order.order_amount"], "alias": ["成交总额"]},
]

GOOD_CHOICE = {
    "stores": ["fact_order"],
    "select": ["dim_region.region_name"],
    "metrics": [{"name": "GMV", "agg": "SUM"}],
    "group_by": ["dim_region.region_name"],
}


# ============================================================ ① AST 归因

class TestAstBuilderReasonClass:
    """`ASTBuildError.reason_class` 三类归因必须准确（决定给模型什么反馈）。"""

    def test_正常路径出_sql(self):
        sql = build_sql(dict(GOOD_CHOICE), TABLES, METRICS)
        assert "SELECT" in sql.upper() and "GMV" in sql.upper() or "order_amount" in sql

    def test_编造列_归因_structure(self):
        bad = dict(GOOD_CHOICE, select=["dim_region.NOT_EXIST"], group_by=["dim_region.NOT_EXIST"])
        with pytest.raises(ASTBuildError) as ei:
            build_sql(bad, TABLES, METRICS)
        assert ei.value.reason_class == "structure"

    def test_未给事实表_归因_select(self):
        with pytest.raises(ASTBuildError) as ei:
            build_sql({"select": ["dim_region.region_name"]}, TABLES, METRICS)
        assert ei.value.reason_class == "select"

    def test_未知指标_归因_select(self):
        with pytest.raises(ASTBuildError) as ei:
            build_sql(dict(GOOD_CHOICE, metrics=[{"name": "NOT_A_METRIC", "agg": "SUM"}]), TABLES, METRICS)
        assert ei.value.reason_class == "select"

    def test_白名单外聚合_归因_select(self):
        with pytest.raises(ASTBuildError) as ei:
            build_sql(dict(GOOD_CHOICE, metrics=[{"name": "GMV", "agg": "DROP"}]), TABLES, METRICS)
        assert ei.value.reason_class == "select"

    def test_多事实表_归因_backbone(self):
        with pytest.raises(ASTBuildError) as ei:
            build_sql({"stores": ["fact_order", "dim_region"]}, TABLES, METRICS)
        assert ei.value.reason_class == "backbone"

    def test_引用未join的表_归因_backbone(self):
        tables = [dict(t) for t in TABLES]
        # 抽掉 dim_region 的主键 ⇒ join 链推不出来
        tables[1]["columns"] = [c for c in tables[1]["columns"] if c["name"] != "region_id"]
        with pytest.raises(ASTBuildError) as ei:
            build_sql(dict(GOOD_CHOICE), tables, METRICS)
        assert ei.value.reason_class == "backbone"

    def test_向后兼容_仍是_ValueError(self):
        with pytest.raises(ValueError):
            build_sql({}, TABLES, METRICS)

    def test_默认归因保守为_structure(self):
        err = ASTBuildError("裸抛")
        assert err.reason_class == "structure"

    # ---- ★ `IN` 的多值支持（首轮量测暴露的能力缺口）----

    def test_where_in_列表正确展开(self):
        choice = dict(GOOD_CHOICE, where=[
            {"column": "dim_region.region_name", "op": "in", "value": ["华东", "华南"]}
        ])
        sql = build_sql(choice, TABLES, METRICS)
        assert "IN ('华东', '华南')" in sql

    def test_where_in_标量自动包成单元素(self):
        choice = dict(GOOD_CHOICE, where=[
            {"column": "dim_region.region_name", "op": "IN", "value": "华北"}
        ])
        assert "IN ('华北')" in build_sql(choice, TABLES, METRICS)

    def test_where_in_空列表归因_structure(self):
        choice = dict(GOOD_CHOICE, where=[
            {"column": "dim_region.region_name", "op": "in", "value": []}
        ])
        with pytest.raises(ASTBuildError) as ei:
            build_sql(choice, TABLES, METRICS)
        assert ei.value.reason_class == "structure"

    def test_where_等值不受影响(self):
        choice = dict(GOOD_CHOICE, where=[
            {"column": "dim_region.region_name", "op": "=", "value": "华北"}
        ])
        sql = build_sql(choice, TABLES, METRICS)
        assert "= '华北'" in sql and " IN (" not in sql.upper()

    # ---- ★ `_referenced_tables` 不能把聚合表达式当表名（量测暴露的缺陷）----

    def test_order_by_用聚合表达式_不误报_backbone(self):
        """★ `order_by.column = "SUM(fact_order.order_quantity)"` 曾把 `SUM(fact_order`
        当成表名 ⇒ 误报 `backbone`（且错误信息误导为「主键列缺失」）。"""
        choice = {
            "stores": ["fact_order"],
            "metrics": [{"name": "SUM(fact_order.order_quantity)"}],
            "group_by": ["dim_region.region_name"],
            "order_by": {"column": "SUM(fact_order.order_quantity)", "asc": False},
        }
        sql = build_sql(choice, TABLES, METRICS)
        assert "SUM(fact_order.order_quantity)" in sql

    def test_聚合表达式不产生伪表名(self):
        from app.agent.ast_builder import _referenced_tables

        choice = {"stores": ["fact_order"],
                  "order_by": {"column": "SUM(fact_order.order_quantity)"}}
        refs = _referenced_tables(choice, TABLES)
        assert refs == {"fact_order"}, refs

    def test_裸聚合表达式按列归属反查(self):
        from app.agent.ast_builder import _referenced_tables

        choice = {"stores": ["fact_order"], "order_by": {"column": "SUM(order_amount)"}}
        assert _referenced_tables(choice, TABLES) == {"fact_order"}

    def test_未知表前缀被忽略而非误报(self):
        from app.agent.ast_builder import _referenced_tables

        choice = {"select": ["not_a_table.col"]}
        assert _referenced_tables(choice, TABLES) == set()

    def test_strip_agg(self):
        from app.agent.ast_builder import _strip_agg

        assert _strip_agg("SUM(fact_order.order_quantity)") == "fact_order.order_quantity"
        assert _strip_agg("avg( order_amount )") == "order_amount"
        assert _strip_agg("COUNT(*)") == "*"
        assert _strip_agg("dim_region.province") == "dim_region.province"


# ============================================================ ② L1 判定

class TestJudgeVerdict:
    """`verdict` 口径：阈值只作用于向量路；**歧义只看指标**（取值多命中是常态）。"""

    def test_三路全空_low(self):
        assert judge_verdict([], [], [], None) == "low"

    def test_向量分低于阈值且无取值_low(self):
        assert judge_verdict([0.5], [], [], None) == "low"

    def test_单一指标_unique(self):
        assert judge_verdict([1.0], [{"name": "GMV"}], [], [1.0]) == "unique"

    def test_两个高相似指标_ambiguous(self):
        assert judge_verdict([1.0, 1.0], [{"name": "GMV"}, {"name": "AOV"}], [], [1.0, 1.0]) == "ambiguous"

    def test_多个取值_不算歧义(self):
        # ★ 真实数据修正过的口径：华北/华东 同属地区名都会被召回，不应误报
        assert judge_verdict([0.9], [{"name": "GMV"}], [{"value": "华北"}, {"value": "华东"}], [0.9]) == "unique"

    def test_指标低于阈值但有取值_unique(self):
        assert judge_verdict([0.7], [{"name": "GMV"}], [{"value": "华北"}], [0.7]) == "unique"


# ============================================================ ③ 动态轮次

class _Req:
    """最小 `ToolCallRequest` 替身（`_settle` 只读 `.state`）。"""

    def __init__(self, state: dict):
        self.state = state


def _tool_msg(ok: bool) -> Command:
    """造一个工具返回：成功 ⇒ reason_class 空；失败 ⇒ 非空。"""
    return Command(update={
        "reason_class": "" if ok else "empty",
        "messages": [ToolMessage(content="x", tool_call_id="c1")],
    })


class TestControlMiddleware:
    """三层记账 + 三终止条件。"""

    def test_思考扣1(self):
        mw = ControlMiddleware()
        out = mw._before({"sql_budget": 5}, None)
        assert out == {"sql_budget": 4}

    def test_工具成功_返还_净0(self):
        mw = ControlMiddleware()
        out = mw._settle(_Req({"sql_budget": 4}), _tool_msg(ok=True))
        assert out.update["sql_budget"] == 5        # 4 → 5（返还）

    def test_工具失败_额外扣_净负2(self):
        mw = ControlMiddleware()
        out = mw._settle(_Req({"sql_budget": 4}), _tool_msg(ok=False))
        assert out.update["sql_budget"] == 3        # 4 → 3（惩罚）

    def test_budget5_恰好3次机会(self):
        """★ 用户算术：异常路径净 −2 ⇒ 5 → 3 → 1 → −1，共 **3 次**机会。

        注意循环条件：中间件是 **`budget > 0` 才允许再想一轮**（`before_model` 里判的），
        所以 `budget=1` 时**仍会**尝试一次（思考 −1 → 0，工具失败 −1 → −1）⇒ 第 3 次机会成立。
        """
        budget = INITIAL_BUDGET
        chances = 0
        while budget > 0:               # ← 与 `before_model` 的判据一致
            budget -= 1                 # ① 思考
            budget -= 1                 # ③ 工具失败（不返还）
            chances += 1
        assert chances == 3

    def test_预算为1时仍允许一次尝试(self):
        """★ 语义锁定：终止判据是 `budget <= 0` ⇒ `budget=1` **仍会**再想一轮。"""
        mw = ControlMiddleware()
        assert mw._before({"sql_budget": 1}, None) == {"sql_budget": 0}
        assert mw._before({"sql_budget": 0}, None)["jump_to"] == "end"
        assert mw._before({"sql_budget": -1}, None)["jump_to"] == "end"

    def test_正常路径不消耗预算(self):
        """★ 不变式：思考 −1 + 工具成功 +1 ⇒ 净 0，不出错就不会被预算终止。"""
        mw = ControlMiddleware()
        budget = INITIAL_BUDGET
        for _ in range(10):
            budget = mw._before({"sql_budget": budget}, None)["sql_budget"]   # −1
            budget = mw._settle(_Req({"sql_budget": budget}), _tool_msg(True)).update["sql_budget"]  # +1
        assert budget == INITIAL_BUDGET

    def test_澄清触发_终止且不扣预算(self):
        mw = ControlMiddleware()
        out = mw._before({"sql_budget": 5, "pending_ask": True}, None)
        assert out == {"jump_to": "end"}
        assert "sql_budget" not in out

    def test_needs_next_终止并给话术(self):
        mw = ControlMiddleware()
        out = mw._before({"sql_budget": 5, "needs_next": "L1", "reason_class": "backbone"}, None)
        assert out["jump_to"] == "end"
        assert out["messages"][0].content            # 有解释，不是静默终止

    def test_预算耗尽_快速失败(self):
        mw = ControlMiddleware()
        out = mw._before({"sql_budget": 0}, None)
        assert out["jump_to"] == "end"
        assert out["reason_class"] == REASON_BUDGET_EXHAUSTED
        assert out["needs_next"] == "L1"

    def test_未给预算时兜底初始值(self):
        mw = ControlMiddleware()
        assert mw._before({}, None) == {"sql_budget": INITIAL_BUDGET - 1}


# ============================================================ ④ 预算锁

class TestFailureLedger:
    """失败率记账 + 阈值告警。"""

    def test_记账与快照(self):
        led = FailureLedger()
        for _ in range(8):
            led.record(True)
        for _ in range(2):
            led.record(False, reason="empty")
        snap = led.snapshot()
        assert snap["total"] == 10 and snap["failures"] == 2
        assert snap["failure_rate"] == 0.2
        assert snap["by_reason"] == {"empty": 2}

    def test_最小样本内不告警(self):
        led = FailureLedger()
        led.record(False)               # 1/1 = 100% 但样本不足
        assert led.snapshot()["alerted"] == []

    def test_越过告警阈值且只报一次(self):
        led = FailureLedger()
        for _ in range(MIN_SAMPLE - 1):
            led.record(True)
        led.record(False)               # 20 次中 1 次失败 = 5% ⇒ 触发
        assert led.snapshot()["alerted"] == ["warn"]
        led.record(False)               # 再失败不重复报
        assert led.snapshot()["alerted"] == ["warn"]

    def test_直接越过严重阈值(self):
        led = FailureLedger()
        for _ in range(MIN_SAMPLE - 3):
            led.record(True)
        for _ in range(3):
            led.record(False, reason="dberror")   # 20 次中 3 次 = 15%
        assert led.snapshot()["alerted"] == ["critical"]

    def test_reset(self):
        led = FailureLedger()
        led.record(False)
        led.reset()
        snap = led.snapshot()
        assert snap["total"] == 0 and snap["alerted"] == []

    def test_record_outcome_判定口径(self):
        from app.agent.agent.budget_lock import record_outcome

        led = FailureLedger()
        import app.agent.agent.budget_lock as bl

        old = bl.ledger
        bl.ledger = led
        try:
            assert record_outcome({"reason_class": "", "needs_next": None}) is True
            assert record_outcome({"reason_class": "empty"}) is False
            assert record_outcome({"needs_next": "L1", "reason_class": "backbone"}) is False
        finally:
            bl.ledger = old
        assert led.snapshot()["total"] == 3 and led.snapshot()["failures"] == 2


# ============================================================ ⑤ content 归一化

class TestContentToText:
    """新端点（`ark-code-latest`）返回内容块列表 ⇒ 必须归一化。"""

    @pytest.mark.parametrize("value,expected", [
        ("abc", "abc"),
        ([{"type": "text", "text": "你好", "index": 0}], "你好"),
        ([{"type": "text", "text": "A"}, {"type": "text", "text": "B"}], "AB"),
        ([{"type": "reasoning", "text": "x"}, {"type": "text", "text": "Y"}], "Y"),
        (None, ""),
    ])
    def test_归一化(self, value, expected):
        assert content_to_text(value) == expected


# ============================================================ ⑥ 澄清辅助

class TestClarifyHelpers:
    """澄清拼装 + `ask_user` 表单归一化。"""

    def test_拼装跳过空值(self):
        got = _assemble_clarified_query("华北的销售额", {"q1": "GMV", "q1__other": "", "q2": [], "q3": None})
        assert "q1=GMV" in got
        assert "q1__other" not in got and "q2=" not in got and "q3=" not in got

    def test_无残留标点(self):
        got = _assemble_clarified_query("问题", {"q1": "A"})
        assert "：。" not in got
        assert got.startswith("问题。补充信息：")

    def test_全空则原样返回(self):
        assert _assemble_clarified_query("  原问题  ", {}) == "原问题"

    def test_列表答案拼接(self):
        got = _assemble_clarified_query("问题", {"q1": ["A", "B"]})
        assert "q1=A, B" in got

    def test_ask_user_归一化_question改text(self):
        out = _normalize([{"question": "口径？", "type": "radio", "options": ["含税", "不含税"]}])
        assert out[0]["text"] == "口径？" and out[0]["type"] == "radio"
        assert out[0]["id"] == "q1"

    def test_ask_user_单选无选项退化为text(self):
        out = _normalize([{"text": "补充说明", "type": "radio"}])
        assert out[0]["type"] == "text" and "options" not in out[0]

    def test_ask_user_有选项但类型非法_默认radio(self):
        out = _normalize([{"text": "选一个", "type": "weird", "options": ["A", "B"]}])
        assert out[0]["type"] == "radio"

    def test_ask_user_丢弃空文本(self):
        assert _normalize([{"text": "  "}, "not-a-dict", {"text": "有效"}]) == [
            {"id": "q1", "text": "有效", "type": "text"}
        ]


# ============================================================ ⑦ sqlflow 辅助

class TestSqlflowHelpers:
    """LIMIT 改写与选词摘要。"""

    def test_with_limit_替换已有_LIMIT(self):
        got = with_limit("SELECT a FROM t LIMIT 5", 1000)
        assert got.upper().count("LIMIT") == 1 and "1000" in got

    def test_with_limit_新增_LIMIT(self):
        got = with_limit("SELECT a FROM t", 1000)
        assert got.upper().count("LIMIT") == 1

    def test_选词摘要含指标维度过滤(self):
        got = _summarize(
            {"metrics": [{"name": "GMV"}], "group_by": ["dim_region.region_name"],
             "where": [{"column": "dim_region.region_name", "op": "=", "value": "华北"}]},
            METRICS,
        )
        assert "指标=GMV" in got and "维度=region_name" in got and "过滤=region_name=华北" in got

    def test_选词摘要无聚合条件(self):
        assert _summarize({}, METRICS) == "无聚合条件"
