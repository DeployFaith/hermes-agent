"""Fixtures shared across hermes_cli kanban tests."""

from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture
def all_assignees_spawnable(monkeypatch):
    """Pretend every assignee maps to a real Hermes profile.

    Most dispatcher tests use synthetic assignees ("alice", "bob") that
    don't correspond to actual profile directories on disk. Without this
    patch, the dispatcher's profile-exists guard (PR #20105) routes
    those tasks into ``skipped_nonspawnable`` instead of spawning, which
    would break tests that assert spawn behavior.
    """
    from hermes_cli import profiles

    monkeypatch.setattr(profiles, "profile_exists", lambda name: True)
    # Dispatcher preflight resolves the profile home after the existence guard.
    # Synthetic assignees in legacy unit tests share the fixture's disposable
    # HERMES_HOME; patch both halves of that contract deterministically.
    monkeypatch.setattr(
        profiles,
        "resolve_profile_env",
        lambda name: os.environ["HERMES_HOME"],
    )


@pytest.fixture
def required_review_skill(kanban_home):
    """Install the review capability required by fail-closed dispatch tests."""
    skill_dir = Path(kanban_home) / "skills" / "sdlc-review"
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_dir.joinpath("SKILL.md").write_text(
        "---\nname: sdlc-review\ndescription: Review changes.\n---\n",
        encoding="utf-8",
    )
    return skill_dir


@pytest.fixture(autouse=True)
def _suppress_concurrent_hermes_gate(request, monkeypatch):
    """Default ``_detect_concurrent_hermes_instances`` to ``[]`` for every test.

    The Windows update path now refuses to proceed when another
    ``hermes.exe`` is detected (issue #26670). On a developer's Windows
    machine running the test suite via ``hermes`` itself, this would
    flag the running agent as a concurrent instance and abort every
    ``cmd_update`` test. Tests that want to exercise the gate explicitly
    re-patch ``_detect_concurrent_hermes_instances`` with their own
    return value — autouse here gives a clean default without touching
    the rest of the suite.

    Tests that need to call the REAL function (e.g. unit tests for the
    helper itself) opt out with ``@pytest.mark.real_concurrent_gate``.
    """
    if request.node.get_closest_marker("real_concurrent_gate"):
        return
    try:
        from hermes_cli import main as _cli_main
    except Exception:
        return
    # raising=False: under pytest's per-test spawn isolation, a concurrent
    # xdist worker importing a module that transitively touches hermes_cli.main
    # can briefly expose a partially-initialized module object here — one where
    # _detect_concurrent_hermes_instances isn't defined yet. A bare setattr
    # would raise AttributeError and error the (unrelated) test. The attribute
    # always exists once main.py finishes importing, so a no-op when it's
    # transiently absent is the correct, race-free default.
    monkeypatch.setattr(
        _cli_main,
        "_detect_concurrent_hermes_instances",
        lambda *_a, **_k: [],
        raising=False,
    )
