"""Utility helpers for the AI × Banking newsletter pipeline."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional
from urllib import request, error

try:  # pragma: no cover - optional dependency
    import yaml  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - handled at runtime
    yaml = None

CACHE_ROOT = Path(".cache")
RAW_CACHE = CACHE_ROOT / "raw"
NORMALIZED_CACHE = CACHE_ROOT / "normalized"


class ToolExecutionError(RuntimeError):
    """Raised when a remote MCP tool invocation fails."""


class ToolUnavailableError(RuntimeError):
    """Raised when a tool is misconfigured or intentionally disabled."""


@dataclass
class RetryConfig:
    retries: int = 3
    backoff_factor: float = 0.5
    max_delay: float = 8.0


def setup_logging(debug: bool = False) -> None:
    """Configure logging for the CLI entry point."""

    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_yaml(path: Path) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required to load YAML configuration files.")
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def hash_dict(data: Mapping[str, Any]) -> str:
    normalized = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def parse_date(value: Any) -> Optional[dt.datetime]:
    if value is None:
        return None
    if isinstance(value, dt.datetime):
        return value
    if isinstance(value, dt.date):
        return dt.datetime.combine(value, dt.time.min)
    if isinstance(value, (int, float)):
        return dt.datetime.fromtimestamp(value, tz=dt.timezone.utc)
    text = str(value).strip()
    if not text:
        return None
    # Try ISO 8601 first.
    try:
        return dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        pass
    # Fallback to RFC 2822 via email.utils
    from email.utils import parsedate_to_datetime

    try:
        return parsedate_to_datetime(text)
    except (TypeError, ValueError):
        return None


def within_days(timestamp: dt.datetime, days: int, now: Optional[dt.datetime] = None) -> bool:
    if timestamp is None:
        return False
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=dt.timezone.utc)
    now = now or dt.datetime.now(dt.timezone.utc)
    delta = now - timestamp
    return delta <= dt.timedelta(days=days)


@dataclass
class ToolConfig:
    name: str
    url: str
    token: Optional[str] = None

    @classmethod
    def from_mapping(cls, name: str, mapping: Mapping[str, Any]) -> "ToolConfig":
        raw_url = str(mapping.get("url", "")).strip()
        url = os.path.expandvars(raw_url)
        if not url or url == "-":
            raise ToolUnavailableError(f"Tool '{name}' is not configured (url={raw_url!r}).")
        token_raw = mapping.get("token")
        token = os.path.expandvars(token_raw) if isinstance(token_raw, str) else None
        if token:
            token = token.strip() or None
        return cls(name=name, url=url.rstrip("/"), token=token)


class MCPToolkit:
    """Minimal HTTP client for MCP tools with retry support."""

    def __init__(self, configs: Mapping[str, ToolConfig], timeout: float = 30.0, retry: Optional[RetryConfig] = None) -> None:
        self._configs = configs
        self._timeout = timeout
        self._retry = retry or RetryConfig()
        self._log = logging.getLogger(self.__class__.__name__)

    @classmethod
    def from_file(cls, path: Path) -> "MCPToolkit":
        raw = load_yaml(path) if path.suffix in {".yaml", ".yml"} else json.loads(path.read_text(encoding="utf-8"))
        configs: Dict[str, ToolConfig] = {}
        clients = raw.get("clients", {}) if isinstance(raw, MutableMapping) else {}
        for name, mapping in clients.items():
            try:
                configs[name] = ToolConfig.from_mapping(name, mapping)
            except ToolUnavailableError as exc:  # log and skip optional tools
                logging.getLogger(cls.__name__).debug(str(exc))
        return cls(configs)

    def call(self, tool: str, method: str, payload: Mapping[str, Any]) -> Any:
        config = self._configs.get(tool)
        if not config:
            raise ToolUnavailableError(f"Tool '{tool}' is unavailable.")
        url = f"{config.url}/{method}"
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if config.token:
            headers["Authorization"] = f"Bearer {config.token}"
        attempt = 0
        delay = self._retry.backoff_factor
        while True:
            attempt += 1
            req = request.Request(url, data=data, headers=headers, method="POST")
            try:
                with request.urlopen(req, timeout=self._timeout) as resp:
                    body = resp.read().decode("utf-8")
                    if not body:
                        return None
                    return json.loads(body)
            except error.HTTPError as exc:  # Non-retryable for 4xx
                if 400 <= exc.code < 500 and exc.code != 429:
                    raise ToolExecutionError(f"Tool {tool}.{method} failed with status {exc.code}: {exc.read()}" ) from exc
                if attempt > self._retry.retries:
                    raise ToolExecutionError(f"Tool {tool}.{method} failed after {attempt} attempts: {exc}") from exc
            except error.URLError as exc:
                if attempt > self._retry.retries:
                    raise ToolExecutionError(f"Tool {tool}.{method} unreachable: {exc}") from exc
            except json.JSONDecodeError as exc:
                raise ToolExecutionError(f"Invalid JSON from tool {tool}.{method}: {exc}") from exc
            # retry path
            self._log.warning("Retrying %s.%s attempt %s", tool, method, attempt)
            delay = min(delay * 2, self._retry.max_delay)
            time_sleep(delay)


def time_sleep(seconds: float) -> None:
    """Wrapper that can be monkeypatched in tests."""

    import time

    time.sleep(seconds)


def cache_path(base: Path, key: str) -> Path:
    ensure_dir(base)
    return base / f"{key}.json"


def load_cached_json(path: Path) -> Optional[Any]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except json.JSONDecodeError:
        path.unlink(missing_ok=True)
        return None


def save_cached_json(path: Path, data: Any) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)


def isoformat(timestamp: dt.datetime) -> str:
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=dt.timezone.utc)
    return timestamp.isoformat()


def sentence_split(text: str) -> List[str]:
    cleaned = text.replace("\n", " ")
    parts = [p.strip() for p in cleaned.replace("?", ".").replace("!", ".").split(".")]
    return [p for p in parts if p]


def summarize_text(text: str, sentences: int = 3) -> str:
    chunks = sentence_split(text)
    return " ".join(chunks[:sentences])


def bullets_from_text(text: str, count: int = 3) -> List[str]:
    sentences = sentence_split(text)
    bullets: List[str] = []
    for idx in range(count):
        try:
            bullets.append(sentences[idx] if idx < len(sentences) else "Gelişme hakkında ayrıntı bekleniyor.")
        except IndexError:
            bullets.append("Gelişme hakkında ayrıntı bekleniyor.")
    return bullets[:count]


def compute_score(
    source_trust: float,
    novelty: float,
    sector_impact: float,
    tr_relevance: float,
    diversity: float,
    weights: Mapping[str, float],
) -> float:
    return (
        source_trust * weights.get("source_trust", 0.0)
        + novelty * weights.get("novelty", 0.0)
        + sector_impact * weights.get("sector_impact", 0.0)
        + tr_relevance * weights.get("tr_relevance", 0.0)
        + diversity * weights.get("diversity", 0.0)
    )


__all__ = [
    "CACHE_ROOT",
    "RAW_CACHE",
    "NORMALIZED_CACHE",
    "RetryConfig",
    "MCPToolkit",
    "ToolExecutionError",
    "ToolUnavailableError",
    "setup_logging",
    "ensure_dir",
    "load_yaml",
    "hash_dict",
    "parse_date",
    "within_days",
    "time_sleep",
    "cache_path",
    "load_cached_json",
    "save_cached_json",
    "isoformat",
    "sentence_split",
    "summarize_text",
    "bullets_from_text",
    "compute_score",
]
