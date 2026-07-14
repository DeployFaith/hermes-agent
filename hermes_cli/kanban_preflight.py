"""Side-effect-free dispatcher preflight for profile identities and skills.

This module deliberately reads only bounded declarative files.  It does not use
Hermes' runtime skill loader, plugin registry, template renderer, usage tracker,
or config caches: dispatcher availability checks must never execute capability.
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

_MAX_CONFIG_BYTES = 256 * 1024
_MAX_SKILL_METADATA_BYTES = 64 * 1024
_MAX_SKILL_FILES = 4096
_MAX_SKILL_DEPTH = 4
_SKILL_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_RAW_PEER_RE = re.compile(r"^(?:12d3koo|qm)[a-z0-9]{20,}$", re.IGNORECASE)
_PLATFORM_MAP = {"macos": "darwin", "windows": "win32", "linux": "linux"}
_EXCLUDED_DIRS = frozenset({
    ".git",
    ".github",
    ".hub",
    ".archive",
    ".venv",
    "venv",
    "node_modules",
    "site-packages",
    "__pycache__",
    ".tox",
    ".nox",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "references",
    "templates",
    "assets",
    "scripts",
})


@dataclass(frozen=True)
class PreflightResult:
    ok: bool
    code: str
    profile: Optional[str] = None
    skills: tuple[str, ...] = ()


def stable_effective_skills(
    skills: Optional[Iterable[Any]], *, review: bool = False
) -> tuple[str, ...]:
    """Return the exact stable-deduped force-load list used by preflight/spawn."""
    result: list[str] = []
    seen: set[str] = set()
    for raw in [*(skills or ()), *(["sdlc-review"] if review else [])]:
        if not isinstance(raw, str):
            continue
        name = raw.strip()
        if name and name not in seen:
            seen.add(name)
            result.append(name)
    return tuple(result)


def normalize_local_profile(raw: Any) -> tuple[Optional[str], str]:
    """Validate a spawn-bound identity before any filesystem existence lookup."""
    if not isinstance(raw, str) or not raw.strip():
        return None, "assignee_missing"
    text = raw.strip()
    if _RAW_PEER_RE.fullmatch(text) or text.startswith(("peer:", "/p2p/")):
        return None, "assignee_peer_id"
    try:
        from hermes_cli.profiles import normalize_profile_name, validate_profile_name

        canonical = normalize_profile_name(text)
        validate_profile_name(canonical)
    except Exception:
        return None, "assignee_malformed"
    return canonical, "ok"


def _bounded_read(path: Path, limit: int) -> bytes:
    with path.open("rb") as handle:
        data = handle.read(limit + 1)
    if len(data) > limit:
        raise ValueError("oversized")
    return data


def _load_yaml_mapping(path: Path, limit: int) -> dict[str, Any]:
    try:
        raw = _bounded_read(path, limit)
        text = raw.decode("utf-8")
        import yaml

        parsed = yaml.safe_load(text)
        return parsed if isinstance(parsed, dict) else {}
    except FileNotFoundError:
        return {}
    except Exception as exc:
        raise ValueError("metadata_unreadable") from exc


def _profile_config(profile_home: Path) -> dict[str, Any]:
    return _load_yaml_mapping(profile_home / "config.yaml", _MAX_CONFIG_BYTES)


def _contained(candidate: Path, root: Path) -> Optional[Path]:
    try:
        resolved_root = root.expanduser().resolve(strict=True)
        resolved = candidate.expanduser().resolve(strict=True)
        resolved.relative_to(resolved_root)
        return resolved
    except (OSError, RuntimeError, ValueError):
        return None


def _skill_roots(
    profile_home: Path, shared_home: Path, config: dict[str, Any]
) -> tuple[Path, ...]:
    roots: list[Path] = [profile_home / "skills"]
    shared = shared_home / "skills"
    if shared != roots[0]:
        roots.append(shared)
    skills_cfg = config.get("skills") if isinstance(config.get("skills"), dict) else {}
    external = skills_cfg.get("external_dirs", [])
    if isinstance(external, str):
        external = [external]
    if isinstance(external, list):
        for raw in external[:128]:
            if isinstance(raw, str) and raw.strip():
                roots.append(Path(os.path.expandvars(raw)).expanduser())
    deduped: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        try:
            key = os.path.normcase(str(root.resolve(strict=True)))
        except (OSError, RuntimeError):
            continue
        if key not in seen and root.is_dir():
            seen.add(key)
            deduped.append(root)
    return tuple(deduped)


def _disabled_skills(config: dict[str, Any]) -> set[str]:
    skills_cfg = config.get("skills") if isinstance(config.get("skills"), dict) else {}
    disabled = skills_cfg.get("disabled", [])
    result = {str(value) for value in disabled} if isinstance(disabled, list) else set()
    platform_disabled = skills_cfg.get("platform_disabled")
    if isinstance(platform_disabled, dict):
        cli_disabled = platform_disabled.get("cli", [])
        if isinstance(cli_disabled, list):
            result.update(str(value) for value in cli_disabled)
    return result


def _frontmatter(path: Path) -> dict[str, Any]:
    raw = _bounded_read(path, _MAX_SKILL_METADATA_BYTES).decode("utf-8")
    if not raw.startswith("---"):
        return {}
    marker = raw.find("\n---", 3)
    if marker < 0:
        raise ValueError("metadata_malformed")
    import yaml

    parsed = yaml.safe_load(raw[3:marker])
    if parsed is None:
        return {}
    if not isinstance(parsed, dict):
        raise ValueError("metadata_malformed")
    return parsed


def _platform_allowed(metadata: dict[str, Any]) -> bool:
    platforms = metadata.get("platforms")
    if not platforms:
        return True
    if not isinstance(platforms, list):
        platforms = [platforms]
    current = sys.platform
    for raw in platforms:
        value = _PLATFORM_MAP.get(str(raw).strip().lower(), str(raw).strip().lower())
        if current.startswith(value):
            return True
    return False


def _index_skills(roots: tuple[Path, ...]) -> tuple[dict[str, Path], Optional[str]]:
    """Build one bounded declarative index; local roots retain precedence."""
    index: dict[str, Path] = {}
    visited = 0
    for root in roots:
        resolved_root = _contained(root, root)
        if resolved_root is None:
            continue
        stack: list[tuple[Path, int]] = [(resolved_root, 0)]
        while stack:
            directory, depth = stack.pop()
            if depth > _MAX_SKILL_DEPTH:
                return {}, "skill_scan_depth"
            try:
                entries = sorted(directory.iterdir(), key=lambda p: p.name)
            except OSError:
                return {}, "skill_scan_unreadable"
            for entry in entries:
                visited += 1
                if visited > _MAX_SKILL_FILES:
                    return {}, "skill_scan_limit"
                if entry.name in _EXCLUDED_DIRS:
                    continue
                resolved = _contained(entry, resolved_root)
                if resolved is None:
                    return {}, "skill_path_escape"
                if resolved.is_dir():
                    skill_md = resolved / "SKILL.md"
                    if skill_md.is_file():
                        safe_md = _contained(skill_md, resolved_root)
                        if safe_md is None:
                            return {}, "skill_path_escape"
                        try:
                            metadata = _frontmatter(safe_md)
                        except ValueError as exc:
                            return {}, str(exc)
                        name = metadata.get("name", resolved.name)
                        if isinstance(name, str) and _SKILL_NAME_RE.fullmatch(name):
                            index.setdefault(name, safe_md)
                        continue
                    stack.append((resolved, depth + 1))
    return index, None


def preflight_spawn(
    assignee: Any,
    skills: Optional[Iterable[Any]],
    *,
    review: bool = False,
    shared_home: Optional[Path] = None,
    cache: Optional[dict[str, Any]] = None,
) -> PreflightResult:
    """Prove a local profile and every force-loaded skill are spawnable."""
    profile, code = normalize_local_profile(assignee)
    effective = stable_effective_skills(skills, review=review)
    if profile is None:
        return PreflightResult(False, code, skills=effective)
    try:
        from hermes_cli.profiles import profile_exists, resolve_profile_env

        if not profile_exists(profile):
            return PreflightResult(False, "assignee_nonlocal", profile, effective)
        profile_home = Path(resolve_profile_env(profile))
    except Exception:
        return PreflightResult(False, "assignee_nonlocal", profile, effective)
    if not effective:
        return PreflightResult(True, "ok", profile, effective)
    for name in effective:
        if ":" in name:
            return PreflightResult(
                False, "skill_qualified_unsupported", profile, effective
            )
        if not _SKILL_NAME_RE.fullmatch(name):
            return PreflightResult(False, "skill_name_invalid", profile, effective)
    cached = cache.get(profile) if cache is not None else None
    if cached is None:
        try:
            config = _profile_config(profile_home)
        except ValueError:
            return PreflightResult(False, "skill_config_unreadable", profile, effective)
        disabled = _disabled_skills(config)
        roots = _skill_roots(profile_home, shared_home or profile_home, config)
        index, error = _index_skills(roots)
        cached = (disabled, index, error)
        if cache is not None:
            cache[profile] = cached
    disabled, index, error = cached
    if any(name in disabled for name in effective):
        return PreflightResult(False, "skill_disabled", profile, effective)
    if error:
        return PreflightResult(False, error, profile, effective)
    for name in effective:
        path = index.get(name)
        if path is None:
            return PreflightResult(False, "skill_missing", profile, effective)
        try:
            metadata = _frontmatter(path)
        except ValueError as exc:
            return PreflightResult(False, str(exc), profile, effective)
        if not _platform_allowed(metadata):
            return PreflightResult(
                False, "skill_platform_incompatible", profile, effective
            )
    return PreflightResult(True, "ok", profile, effective)
