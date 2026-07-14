from __future__ import annotations

import argparse
import json
import os
import socket
import sys
from pathlib import Path

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_preflight as kp


@pytest.fixture
def board(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _profile(home: Path, name: str = "worker") -> Path:
    path = home / "profiles" / name
    (path / "skills").mkdir(parents=True)
    (path / "config.yaml").write_text("skills:\n  disabled: []\n", encoding="utf-8")
    return path


def _skill(root: Path, name: str, *, platforms: str = "") -> Path:
    path = root / "skills" / name
    path.mkdir(parents=True)
    platform_line = f"platforms: [{platforms}]\n" if platforms else ""
    (path / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Test.\n{platform_line}---\n\nBody.\n",
        encoding="utf-8",
    )
    return path


def _archived_open_run(
    conn, *, title: str = "legacy", pid: int = 123456, expiry: int = 100
):
    task_id = kb.create_task(conn, title=title, assignee="worker")
    claimed = kb.claim_task(
        conn, task_id, claimer=f"{socket.gethostname()}:{pid}", ttl_seconds=60
    )
    assert claimed and claimed.current_run_id
    run_id = claimed.current_run_id
    conn.execute(
        "UPDATE tasks SET status='archived', claim_expires=?, worker_pid=? WHERE id=?",
        (expiry, pid, task_id),
    )
    conn.execute(
        "UPDATE task_runs SET claim_expires=?, worker_pid=? WHERE id=?",
        (expiry, pid, run_id),
    )
    conn.commit()
    return task_id, run_id


def test_archived_reconcile_applies_preserving_parent_and_history(board):
    with kb.connect_closing() as conn:
        task_id, run_id = _archived_open_run(conn)
        conn.execute(
            "UPDATE tasks SET completed_at=77, result='keep-result' WHERE id=?",
            (task_id,),
        )
        child = kb.create_task(
            conn, title="child", assignee="worker", parents=[task_id]
        )
        conn.execute("UPDATE tasks SET status='todo' WHERE id=?", (child,))
        before_events = len(kb.list_events(conn, task_id))
        result = kb.reconcile_archived_runs(
            conn, [task_id], apply=True, now=200, pid_alive=lambda pid: False
        )
        assert result.applied == 1 and result.refused == 0
        task = kb.get_task(conn, task_id)
        assert (task.status, task.completed_at, task.result) == (
            "archived",
            77,
            "keep-result",
        )
        assert task.current_run_id is None
        run = kb.latest_run(conn, task_id)
        assert (
            run.id == run_id
            and run.status == "reclaimed"
            and run.outcome == "reclaimed"
        )
        assert len(kb.list_events(conn, task_id)) == before_events + 1
        assert kb.get_task(conn, child).status == "todo"
        again = kb.reconcile_archived_runs(
            conn, [task_id], apply=True, now=201, pid_alive=lambda pid: False
        )
        assert again.applied == 0 and again.refused == 1
        assert len(kb.list_events(conn, task_id)) == before_events + 1


def test_archived_reconcile_accepts_distinct_local_claimer_and_worker_pids(board):
    with kb.connect_closing() as conn:
        task_id, run_id = _archived_open_run(conn, pid=123456)
        lock = f"{socket.gethostname()}:654321"
        conn.execute("UPDATE tasks SET claim_lock=? WHERE id=?", (lock, task_id))
        conn.execute("UPDATE task_runs SET claim_lock=? WHERE id=?", (lock, run_id))
        conn.commit()
        result = kb.reconcile_archived_runs(
            conn, [task_id], apply=True, now=200, pid_alive=lambda pid: False
        )
        assert result.applied == 1


def test_archived_reconcile_dry_run_and_atomic_batch(board):
    with kb.connect_closing() as conn:
        good, good_run = _archived_open_run(conn, title="good")
        bad = kb.create_task(conn, title="not archived", assignee="worker")
        dry = kb.reconcile_archived_runs(
            conn, [good], now=200, pid_alive=lambda pid: False
        )
        assert dry.items[0].eligible and dry.applied == 0
        assert kb.latest_run(conn, good).ended_at is None
        atomic = kb.reconcile_archived_runs(
            conn, [good, bad], apply=True, now=200, pid_alive=lambda pid: False
        )
        assert atomic.applied == 0 and atomic.refused == 1
        assert kb.latest_run(conn, good).id == good_run
        assert kb.latest_run(conn, good).ended_at is None
        partial = kb.reconcile_archived_runs(
            conn,
            [good, bad],
            apply=True,
            allow_partial=True,
            now=200,
            pid_alive=lambda pid: False,
        )
        assert partial.applied == 1 and partial.refused == 1


@pytest.mark.parametrize(
    "mutator,reason",
    [
        (
            lambda c, t, r: c.execute(
                "UPDATE tasks SET claim_lock=NULL WHERE id=?", (t,)
            ),
            "ownership_mismatch",
        ),
        (
            lambda c, t, r: c.execute(
                "UPDATE task_runs SET claim_lock='remote:123456' WHERE id=?", (r,)
            ),
            "ownership_mismatch",
        ),
        (
            lambda c, t, r: c.execute(
                "UPDATE tasks SET claim_expires=NULL WHERE id=?", (t,)
            ),
            "expiry_missing_or_malformed",
        ),
        (
            lambda c, t, r: c.execute(
                "UPDATE tasks SET current_run_id='999999999999999999999999' WHERE id=?",
                (t,),
            ),
            "run_id_malformed",
        ),
        (
            lambda c, t, r: c.execute(
                "UPDATE tasks SET worker_pid=NULL WHERE id=?", (t,)
            ),
            "pid_missing_or_malformed",
        ),
        (
            lambda c, t, r: c.execute(
                "UPDATE task_runs SET worker_pid='999999999999999999999' WHERE id=?",
                (r,),
            ),
            "pid_missing_or_malformed",
        ),
        (
            lambda c, t, r: c.execute(
                "UPDATE task_runs SET status='done' WHERE id=?", (r,)
            ),
            "run_terminal",
        ),
    ],
)
def test_archived_reconcile_malformed_rows_refuse_without_liveness(
    board, mutator, reason
):
    calls = []
    with kb.connect_closing() as conn:
        task_id, run_id = _archived_open_run(conn)
        mutator(conn, task_id, run_id)
        conn.commit()
        result = kb.reconcile_archived_runs(
            conn,
            [task_id],
            apply=True,
            now=200,
            pid_alive=lambda pid: calls.append(pid) or False,
        )
        assert result.items[0].reason == reason
        assert result.applied == 0
        assert calls == []


def test_archived_reconcile_remote_live_and_multiple_open_refuse_safely(board):
    with kb.connect_closing() as conn:
        task_id, run_id = _archived_open_run(conn)
        lock = "remote-host:123456"
        conn.execute("UPDATE tasks SET claim_lock=? WHERE id=?", (lock, task_id))
        conn.execute("UPDATE task_runs SET claim_lock=? WHERE id=?", (lock, run_id))
        conn.commit()
        calls = []
        remote = kb.reconcile_archived_runs(
            conn,
            [task_id],
            apply=True,
            now=200,
            pid_alive=lambda pid: calls.append(pid) or False,
        )
        assert remote.items[0].reason == "ownership_nonlocal" and calls == []

        local = f"{socket.gethostname()}:123456"
        conn.execute("UPDATE tasks SET claim_lock=? WHERE id=?", (local, task_id))
        conn.execute("UPDATE task_runs SET claim_lock=? WHERE id=?", (local, run_id))
        conn.commit()
        live = kb.reconcile_archived_runs(
            conn, [task_id], apply=True, now=200, pid_alive=lambda pid: True
        )
        assert live.items[0].reason == "worker_alive"

        conn.execute(
            "INSERT INTO task_runs(task_id,status,claim_lock,claim_expires,worker_pid,started_at) "
            "VALUES (?, 'running', ?, 100, 123456, 1)",
            (task_id, local),
        )
        conn.commit()
        multiple = kb.reconcile_archived_runs(
            conn, [task_id], apply=True, now=200, pid_alive=lambda pid: False
        )
        assert multiple.items[0].reason == "multiple_open_runs"


def test_archived_reconcile_unexpired_refuses_before_liveness(board):
    calls = []
    with kb.connect_closing() as conn:
        task_id, _ = _archived_open_run(conn, expiry=200)
        result = kb.reconcile_archived_runs(
            conn,
            [task_id],
            apply=True,
            now=200,
            pid_alive=lambda pid: calls.append(pid) or False,
        )
        assert result.items[0].reason == "ownership_not_expired"
        assert calls == []


def test_cli_reconcile_json_exit_contract(board, monkeypatch, capsys):
    with kb.connect_closing() as conn:
        good, _ = _archived_open_run(conn)
    monkeypatch.setattr(kb, "_pid_alive", lambda pid: False)
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    kc.build_parser(sub)
    args = parser.parse_args([
        "kanban",
        "reconcile-archived-runs",
        good,
        "missing",
        "--apply",
        "--json",
    ])
    assert kc.kanban_command(args) == 1
    payload = json.loads(capsys.readouterr().out)
    assert (
        payload["atomic"] is True
        and payload["applied"] == 0
        and payload["refused"] == 1
    )


def test_raw_peer_directory_rejected_before_claim_even_without_skills(board):
    raw_peer = "12D3Koo" + "a" * 40
    (board / "profiles" / raw_peer.lower()).mkdir(parents=True)
    with kb.connect_closing() as conn:
        task_id = kb.create_task(conn, title="peer", assignee=raw_peer)
        spawned = []
        result = kb.dispatch_once(conn, spawn_fn=lambda *a, **k: spawned.append(a) or 1)
        assert task_id in result.skipped_nonspawnable
        assert kb.get_task(conn, task_id).status == "ready"
        assert kb.list_runs(conn, task_id) == []
        assert spawned == []


def test_missing_skill_blocks_before_respawn_guard(board, monkeypatch):
    _profile(board)
    with kb.connect_closing() as conn:
        task_id = kb.create_task(
            conn, title="need skill", assignee="worker", skills=["absent"]
        )
        monkeypatch.setattr(
            kb, "check_respawn_guard", lambda *a: pytest.fail("guard ran")
        )
        result = kb.dispatch_once(conn, spawn_fn=lambda *a: pytest.fail("spawn ran"))
        assert (task_id, "skill_missing") in result.preflight_blocked
        task = kb.get_task(conn, task_id)
        assert task.status == "blocked" and task.block_kind == "capability"
        assert kb.list_runs(conn, task_id) == []


def test_preflight_capability_block_cas_miss_emits_no_event(board):
    with kb.connect_closing() as conn:
        task_id = kb.create_task(conn, title="raced", assignee="worker")
        conn.execute("UPDATE tasks SET status='todo' WHERE id=?", (task_id,))
        conn.commit()
        before = list(kb.list_events(conn, task_id))
        assert not kb._capability_block_preflight(
            conn, task_id, expected_status="ready", reason="skill_missing"
        )
        assert list(kb.list_events(conn, task_id)) == before
        assert kb.get_task(conn, task_id).status == "todo"


def test_preflight_is_declarative_bounded_and_platform_aware(
    board, tmp_path, monkeypatch
):
    profile = _profile(board)
    skill = _skill(profile, "safe")
    (skill / "SKILL.md").write_text(
        "---\nname: safe\ndescription: Test.\n---\n\n"
        "inline: !`touch SHOULD_NOT_EXIST`\n{{ shell('touch ALSO_NOT') }}\n",
        encoding="utf-8",
    )
    import agent.skill_utils as skill_utils

    before_loader = skill_utils._yaml_load_fn
    popen = []
    monkeypatch.setattr("subprocess.Popen", lambda *a, **k: popen.append(a))
    assert kp.preflight_spawn("worker", ["safe"], shared_home=board).ok
    assert skill_utils._yaml_load_fn is before_loader
    assert popen == []
    assert (
        kp.preflight_spawn("worker", ["plugin:safe"], shared_home=board).code
        == "skill_qualified_unsupported"
    )

    _skill(profile, "wrong-platform", platforms="definitely-not-this-platform")
    assert (
        kp.preflight_spawn("worker", ["wrong-platform"], shared_home=board).code
        == "skill_platform_incompatible"
    )

    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "SKILL.md").write_text("---\nname: escaped\n---\n", encoding="utf-8")
    (profile / "skills" / "escaped").symlink_to(outside, target_is_directory=True)
    assert (
        kp.preflight_spawn("worker", ["escaped"], shared_home=board).code
        == "skill_path_escape"
    )


def test_preflight_local_shared_external_disabled_and_review_dedupe(board, tmp_path):
    profile = _profile(board)
    _skill(profile, "local")
    _skill(board, "shared")
    external = tmp_path / "external"
    _skill(external.parent, "external")
    # _skill created tmp_path/skills/external; use that skills root as external dir.
    external_root = tmp_path / "skills"
    (profile / "config.yaml").write_text(
        f"skills:\n  external_dirs:\n    - {external_root}\n  disabled:\n    - disabled\n",
        encoding="utf-8",
    )
    _skill(profile, "disabled")
    _skill(profile, "sdlc-review")
    assert kp.preflight_spawn("worker", ["local"], shared_home=board).ok
    assert kp.preflight_spawn("worker", ["shared"], shared_home=board).ok
    assert kp.preflight_spawn("worker", ["external"], shared_home=board).ok
    assert (
        kp.preflight_spawn("worker", ["disabled"], shared_home=board).code
        == "skill_disabled"
    )
    reviewed = kp.preflight_spawn(
        "worker", ["local", "sdlc-review", "local"], review=True, shared_home=board
    )
    assert reviewed.ok and reviewed.skills == ("local", "sdlc-review")
