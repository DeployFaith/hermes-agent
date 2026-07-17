from __future__ import annotations


def _task(kb):
    return kb.Task(
        id="t_scope",
        title="scope",
        body=None,
        assignee="worker",
        status="running",
        priority=0,
        created_by="test",
        created_at=1,
        started_at=None,
        completed_at=None,
        workspace_kind="scratch",
        workspace_path=None,
        claim_lock="lock",
        claim_expires=None,
        tenant=None,
        current_run_id=7,
    )


def test_scope_disabled_preserves_command(monkeypatch):
    from hermes_cli import kanban_db as kb

    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"kanban": {"worker_systemd_scope": False}},
    )
    cmd = ["hermes", "chat"]
    assert kb._systemd_scope_worker_cmd(_task(kb), cmd) is cmd


def test_scope_enabled_wraps_worker_with_limits(monkeypatch):
    from hermes_cli import kanban_db as kb

    monkeypatch.setenv("XDG_RUNTIME_DIR", "/run/user/1001")
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {
            "kanban": {
                "worker_systemd_scope": True,
                "worker_memory_high_mb": 1200,
                "worker_memory_max_mb": 1800,
            }
        },
    )
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/systemd-run")

    wrapped = kb._systemd_scope_worker_cmd(_task(kb), ["hermes", "chat"])

    assert wrapped[:5] == [
        "/usr/bin/systemd-run",
        "--user",
        "--scope",
        "--quiet",
        "--collect",
    ]
    assert "--unit=hermes-kanban-t_scope-7" in wrapped
    assert "--property=OOMPolicy=continue" in wrapped
    assert "--property=MemoryHigh=1200M" in wrapped
    assert "--property=MemoryMax=1800M" in wrapped
    assert wrapped[-3:] == ["--", "hermes", "chat"]


def test_scope_enabled_fails_closed_without_systemd_run(monkeypatch):
    import pytest
    from hermes_cli import kanban_db as kb

    monkeypatch.setenv("XDG_RUNTIME_DIR", "/run/user/1001")
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"kanban": {"worker_systemd_scope": True}},
    )
    monkeypatch.setattr("shutil.which", lambda name: None)

    with pytest.raises(RuntimeError, match="systemd-run is unavailable"):
        kb._systemd_scope_worker_cmd(_task(kb), ["hermes", "chat"])


def test_resource_admission_blocks_low_memory_without_claiming(monkeypatch, tmp_path):
    from hermes_cli import kanban_db as kb

    root = tmp_path / ".hermes"
    (root / "profiles" / "worker").mkdir(parents=True)
    (root / "profiles" / "worker" / "config.yaml").write_text("{}\n")
    monkeypatch.setenv("HERMES_HOME", str(root))
    kb.init_db()

    with kb.connect_closing() as conn:
        task_id = kb.create_task(conn, title="admission", assignee="worker")
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (task_id,))

        monkeypatch.setattr(
            kb,
            "_worker_resource_admission_reason",
            lambda _conn: "available_memory_mb=100<minimum=3072",
        )
        spawned = []
        result = kb.dispatch_once(
            conn,
            spawn_fn=lambda *args, **kwargs: spawned.append(args) or 999,
        )

        task = kb.get_task(conn, task_id)
        assert result.resource_deferred == ["available_memory_mb=100<minimum=3072"]
        assert spawned == []
        assert task is not None
        assert task.status == "ready"
        assert task.consecutive_failures == 0


def test_live_resource_admission_accepts_healthy_host(tmp_path):
    import sqlite3
    from hermes_cli import kanban_db as kb

    conn = sqlite3.connect(tmp_path / "board.db")
    try:
        reason = kb._worker_resource_admission_reason(
            conn,
            {
                "worker_resource_admission": True,
                "worker_min_available_mem_mb": 1,
                "worker_max_swap_used_percent": 100,
                "worker_min_root_free_mb": 1,
                "worker_min_tmp_free_mb": 1,
                "worker_max_memory_pressure_avg10": 100,
            },
        )
        assert reason is None
    finally:
        conn.close()
