from __future__ import annotations

import threading

from app.core.log import logger

WARN_RATE = 0.05        # 5%  ⇒ warning
CRITICAL_RATE = 0.10    # 10% ⇒ error

MIN_SAMPLE = 20


class FailureLedger:
    """失败率记账器（线程安全）。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._total = 0
        self._failures = 0
        self._by_reason: dict[str, int] = {}
        self._alerted: set[str] = set()

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

            triggered: str | None = None
            if total >= MIN_SAMPLE:
                if rate >= CRITICAL_RATE and "critical" not in self._alerted:
                    self._alerted.add("critical")
                    triggered = "critical"
                elif rate >= WARN_RATE and "warn" not in self._alerted:
                    self._alerted.add("warn")
                    triggered = "warn"

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


ledger = FailureLedger()


def record_outcome(final_state: dict) -> bool:
    state = final_state or {}
    reason = str(state.get("reason_class") or "")
    ok = not state.get("needs_next") and not reason
    ledger.record(ok=ok, reason="" if ok else (reason or "unknown"))
    return ok
