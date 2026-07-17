"""Tests for the extracted GatewayKanbanWatchersMixin (god-file Phase 3).

The kanban watcher loops were lifted out of gateway/run.py into a mixin that
GatewayRunner inherits. These tests confirm the mixin exposes the methods and
that GatewayRunner picks them up via the MRO (behavior-neutral relocation).
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

from gateway.kanban_watchers import (
    GatewayKanbanWatchersMixin,
    _bounded_deferral_signal,
    _next_dispatch_bad_ticks,
    _probe_dispatch_health,
)

KANBAN_METHODS = [
    "_kanban_notifier_watcher",
    "_kanban_dispatcher_watcher",
    "_kanban_advance",
    "_kanban_unsub",
    "_kanban_rewind",
    "_deliver_kanban_artifacts",
]


def test_mixin_defines_kanban_methods():
    for m in KANBAN_METHODS:
        assert hasattr(GatewayKanbanWatchersMixin, m), f"mixin missing {m}"


def test_gateway_runner_inherits_mixin():
    # Import here so a heavy gateway import only happens if the first test passed.
    from gateway.run import GatewayRunner

    assert issubclass(GatewayRunner, GatewayKanbanWatchersMixin)
    # Each kanban method resolves to the mixin's implementation via the MRO.
    for m in KANBAN_METHODS:
        owner = next(c for c in GatewayRunner.__mro__ if m in c.__dict__)
        assert owner is GatewayKanbanWatchersMixin, (
            f"{m} resolved to {owner.__name__}, expected the mixin"
        )


def test_watcher_loops_are_coroutines():
    # The two long-running watchers are async loops.
    assert inspect.iscoroutinefunction(GatewayKanbanWatchersMixin._kanban_notifier_watcher)
    assert inspect.iscoroutinefunction(GatewayKanbanWatchersMixin._kanban_dispatcher_watcher)


def test_singleton_dispatcher_lock_is_exclusive(tmp_path):
    """Only one holder of the dispatcher lock at a time — the backstop that
    stops concurrent dispatchers double reclaiming and corrupting shared
    kanban SQLite index pages under wal_autocheckpoint=0."""
    import os

    from gateway.kanban_watchers import _acquire_singleton_lock, _release_singleton_lock

    lock = tmp_path / "kanban" / ".dispatcher.lock"

    h1, st1 = _acquire_singleton_lock(lock)
    assert st1 == "held" and h1 is not None

    # A second acquire while the first is held must be refused, not granted.
    h2, st2 = _acquire_singleton_lock(lock)
    assert st2 == "contended" and h2 is None

    # Releasing the first lets a fresh acquire succeed (lock is reusable).
    _release_singleton_lock(h1)
    h3, st3 = _acquire_singleton_lock(lock)
    assert st3 == "held" and h3 is not None
    _release_singleton_lock(h3)


class _HealthConn:
    def __init__(self, rows):
        self.rows = rows

    def execute(self, _query):
        return self

    def fetchall(self):
        return self.rows

    def close(self):
        pass


class _HealthKb:
    DEFAULT_BOARD = "default"

    def __init__(self, rows):
        self.rows = rows

    def list_boards(self, include_archived=False):
        return [{"slug": "default"}]

    def connect(self, board=None):
        return _HealthConn(self.rows)


def _dispatch_result(**overrides):
    values = {
        "skipped_unassigned": [],
        "resource_deferred": [],
        "skipped_per_profile_capped": [],
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_resource_admission_deferral_resets_stuck_ticks():
    result = _dispatch_result(
        resource_deferred=["available_memory_mb=512<minimum=3072"]
    )
    actionable, reasons = _probe_dispatch_health(
        _HealthKb([{"id": "t1", "assignee": "default"}]),
        [("default", result)],
    )

    assert actionable is False
    assert reasons == ["[default] available_memory_mb=512<minimum=3072"]
    assert _next_dispatch_bad_ticks(
        5, actionable_ready=actionable, any_spawned=False
    ) == 0


def test_profile_capacity_deferral_does_not_hide_other_failed_spawn(monkeypatch):
    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda _name: True)
    result = _dispatch_result(
        skipped_per_profile_capped=[("t_capped", "writer", 2)]
    )
    kb = _HealthKb([
        {"id": "t_capped", "assignee": "writer"},
        {"id": "t_failed", "assignee": "reviewer"},
    ])

    actionable, reasons = _probe_dispatch_health(kb, [("default", result)])

    assert actionable is True
    assert "profile writer at capacity" in reasons[0]
    assert _next_dispatch_bad_ticks(
        2, actionable_ready=actionable, any_spawned=False
    ) == 3


def test_profile_capacity_only_resets_stuck_ticks(monkeypatch):
    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda _name: True)
    result = _dispatch_result(
        skipped_per_profile_capped=[("t_capped", "writer", 2)]
    )
    actionable, reasons = _probe_dispatch_health(
        _HealthKb([{"id": "t_capped", "assignee": "writer"}]),
        [("default", result)],
    )

    assert actionable is False
    assert reasons
    assert _next_dispatch_bad_ticks(
        5, actionable_ready=actionable, any_spawned=False
    ) == 0


def test_unassigned_ready_work_remains_stuck_eligible():
    result = _dispatch_result(skipped_unassigned=["t_unassigned"])
    actionable, _reasons = _probe_dispatch_health(
        _HealthKb([{"id": "t_unassigned", "assignee": None}]),
        [("default", result)],
    )
    assert actionable is True


def test_deferral_signal_is_bounded():
    signal = _bounded_deferral_signal(["x" * 500] * 20)
    assert len(signal) <= 2000
    assert "and 12 more" in signal
