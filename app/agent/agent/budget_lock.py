"""预算锁：**失败率记账 + 阈值告警**。

★ 为什么需要它：

「动态轮次」让**单次查询**不会失控（预算耗尽即快速失败），但它**管不了整体**——
如果某天失败率悄悄升高（prompt 漂移、模型换版、数据源变化、候选退化），
单次预算永远正常，**问题会以「用户投诉」的形式暴露**。

⇒ 预算锁就是那个**整体安全阀**：持续记账，越过阈值就告警。

★ 设计要点：

| 点 | 做法 | 理由 |
|---|---|---|
| 阈值 | **5% 告警 / 10% 严重告警** | 最终口径 |
| 最小样本 | **≥ 20 次才判定** | 否则「1 次失败 / 1 次运行 = 100%」立刻误报 |
| 告警去重 | 每个阈值**只报一次**（`_alerted`） | 避免日志刷屏 |
| 线程安全 | `threading.Lock` | ★ 参照实现的 `_AST_STATS_LOCK` 教训：`dict[k] += 1` 非原子 |
| 归因分布 | 按 `reason_class` 分桶 | 便于定位「哪一类失败变多了」 |

⚠️ **进程内状态**：本实现是模块级单例（够 MVP 用）。生产上应把 `snapshot()`
导出到指标系统（Prometheus / 日志聚合），否则多进程部署时各算各的。
"""

from __future__ import annotations

import threading

from app.core.log import logger

# 告警阈值（失败率）
WARN_RATE = 0.05        # 5%  ⇒ warning
CRITICAL_RATE = 0.10    # 10% ⇒ error

# ★ 最小样本：少于这个次数不做阈值判定（否则小样本必然误报）
MIN_SAMPLE = 20


class FailureLedger:
    """失败率记账器（线程安全）。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._total = 0
        self._failures = 0
        self._by_reason: dict[str, int] = {}
        self._alerted: set[str] = set()

    # ------------------------------------------------------------ 记账
    def record(self, ok: bool, reason: str = "") -> None:
        """记一次查询结果。`ok=False` 时按 `reason` 分桶。"""
        with self._lock:
            self._total += 1
            if not ok:
                self._failures += 1
                key = reason or "unknown"
                self._by_reason[key] = self._by_reason.get(key, 0) + 1

            rate = self._failures / self._total if self._total else 0.0
            total, failures = self._total, self._failures

            # ---- 阈值判定（在锁内做，避免并发下重复告警）----
            triggered: str | None = None
            if total >= MIN_SAMPLE:
                if rate >= CRITICAL_RATE and "critical" not in self._alerted:
                    self._alerted.add("critical")
                    triggered = "critical"
                elif rate >= WARN_RATE and "warn" not in self._alerted:
                    self._alerted.add("warn")
                    triggered = "warn"

        # ---- 日志放在锁外（日志可能慢，不占锁）----
        if triggered == "critical":
            logger.error(
                "【预算锁】失败率 {} 越过严重阈值 {}（{}/{}）⇒ 请检查 prompt / 模型 / 数据源",
                f"{rate:.1%}", f"{CRITICAL_RATE:.0%}", failures, total,
            )
        elif triggered == "warn":
            logger.warning(
                "【预算锁】失败率 {} 越过告警阈值 {}（{}/{}）",
                f"{rate:.1%}", f"{WARN_RATE:.0%}", failures, total,
            )

    # ------------------------------------------------------------ 读取
    def snapshot(self) -> dict:
        """当前快照（供上报 / 量测产物）。"""
        with self._lock:
            total, failures = self._total, self._failures
            by_reason = dict(self._by_reason)
            alerted = sorted(self._alerted)
        rate = failures / total if total else 0.0
        return {
            "total": total,
            "failures": failures,
            "successes": total - failures,
            "failure_rate": round(rate, 4),
            "by_reason": by_reason,
            "alerted": alerted,
            "thresholds": {"warn": WARN_RATE, "critical": CRITICAL_RATE, "min_sample": MIN_SAMPLE},
        }

    def reset(self) -> None:
        """清空（测试用）。"""
        with self._lock:
            self._total = 0
            self._failures = 0
            self._by_reason.clear()
            self._alerted.clear()


# 模块级单例
ledger = FailureLedger()


def record_outcome(final_state: dict) -> bool:
    """从 agent 的最终 state 判定「本次查询是否成功」，并记账。

    判据（与归因口径一致）：
    - `needs_next` 有值（如 `"L1"`）⇒ **需扩圈**，本次未解决 ⇒ 失败
    - `reason_class` 非空 ⇒ 最后一次工具失败未挽回 ⇒ 失败
    - 其余 ⇒ 成功

    返回是否成功（便于调用方复用）。
    """
    state = final_state or {}
    reason = str(state.get("reason_class") or "")
    ok = not state.get("needs_next") and not reason
    ledger.record(ok=ok, reason="" if ok else (reason or "unknown"))
    return ok
