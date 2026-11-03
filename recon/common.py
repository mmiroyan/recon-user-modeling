"""Shared helpers: data I/O, domain configs, turn formatting, and LLM calls."""

from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import litellm
from tqdm import tqdm

litellm.drop_params = True
litellm.suppress_debug_info = True


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_json(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str | Path, data: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def load_split_ids(split_path: str | Path, split_names: list[str]) -> list[str]:
    """Session IDs belonging to the named splits, in split order."""
    split = load_json(split_path)
    ids: list[str] = []
    for name in split_names:
        if name not in split:
            raise ValueError(f"Split '{name}' not found in {split_path}. Available: {sorted(split)}")
        ids.extend(split[name])
    return ids


def target_author_of(sessions: dict) -> str:
    return next(iter(sessions.values()))["metadata"]["target_author_name"]


# ---------------------------------------------------------------------------
# Domain configs
# ---------------------------------------------------------------------------

DOMAIN_CONFIG_KEYS = (
    "reasoning_preamble",
    "action_preamble",
    "reconstruction_preamble",
    "judge_preamble",
    "response_format",
)


def load_domain_config(path: str | Path) -> dict[str, str]:
    cfg = load_json(path)
    missing = [k for k in DOMAIN_CONFIG_KEYS if k not in cfg]
    if missing:
        raise ValueError(f"{path}: missing keys {missing}")
    return cfg


def preamble_block(text: str) -> str:
    text = (text or "").strip()
    return f"{text}\n\n" if text else ""


# ---------------------------------------------------------------------------
# Turns
# ---------------------------------------------------------------------------

# Optional regex filters that drop target turns (e.g. transcription artifacts).
TURN_FILTERS: dict[str, str] = {
    "inaudible": r"(?i)inaudible",
    "interrupted": r"--",
}


def compile_turn_filters(names: list[str] | None) -> dict[str, re.Pattern]:
    unknown = [n for n in (names or []) if n not in TURN_FILTERS]
    if unknown:
        raise ValueError(f"Unknown turn filters {unknown}. Available: {sorted(TURN_FILTERS)}")
    return {n: re.compile(TURN_FILTERS[n]) for n in (names or [])}


def passes_filters(action: str, filters: dict[str, re.Pattern]) -> bool:
    return not any(p.search(action) for p in filters.values())


def is_target_author(author: str, target_author: str) -> bool:
    return target_author.lower() in author.lower()


def resolve_author(author: str, target_author: str, target_alias: str | None) -> str:
    if target_alias is not None and is_target_author(author, target_author):
        return target_alias
    return author


def format_turn(author: str, action: str, context_format: str = "xml") -> str:
    if context_format == "xml":
        return f'<turn author="{author}">\n{action}\n</turn>'
    return f"{author}: {action}"


def format_chunk_context(
    chunk: dict,
    target_author: str,
    target_alias: str | None,
    context_format: str = "xml",
) -> str:
    """Render the context turns of a retrieved chunk (all turns but the last).

    The last turn of a chunk is always the target author's action.
    """
    return "\n".join(
        format_turn(resolve_author(t["author"], target_author, target_alias), t["action"], context_format)
        for t in chunk["turns"][:-1]
    )


def chunk_action(chunk: dict) -> str:
    return chunk["turns"][-1]["action"]


# ---------------------------------------------------------------------------
# LLM calls
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LLM:
    """A litellm model plus sampling settings."""

    model: str
    api_base: str | None = None
    temperature: float = 0.7
    timeout: int = 120
    reasoning_effort: str | None = None
    enable_thinking: bool | None = None

    def _extra(self) -> dict:
        extra: dict = {}
        if self.api_base:
            extra["api_base"] = self.api_base
        if self.reasoning_effort:
            extra["reasoning_effort"] = self.reasoning_effort
        if self.enable_thinking is not None:
            extra["extra_body"] = {"chat_template_kwargs": {"enable_thinking": self.enable_thinking}}
        return extra

    def complete(self, prompt: str, **kwargs) -> str:
        resp = litellm.completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            timeout=self.timeout,
            temperature=self.temperature,
            **self._extra(),
            **kwargs,
        )
        return (resp.choices[0].message.content or "").strip()

    def complete_json(self, prompt: str, schema: dict, schema_name: str) -> dict:
        text = self.complete(
            prompt,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": schema_name, "schema": schema, "strict": True},
            },
        )
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            return json.loads(match.group()) if match else {}


RETRY_BACKOFF = [1, 4]  # seconds before the 2nd and 3rd attempt


def with_retries(fn, label: str = ""):
    """Call ``fn()`` up to 3 times; retry on exceptions and empty strings."""
    result = ""
    for attempt, delay in enumerate([0] + RETRY_BACKOFF):
        if delay:
            time.sleep(delay)
        try:
            result = fn()
        except Exception as exc:
            tqdm.write(f"  [warn] {label} failed (attempt {attempt + 1}): {exc}")
            continue
        if result:
            return result
        tqdm.write(f"  [warn] {label} returned empty output (attempt {attempt + 1})")
    return result


def run_parallel(tasks: list[tuple], max_workers: int, desc: str) -> dict:
    """Run ``(key, fn)`` tasks in a thread pool and return ``{key: fn()}``."""
    results: dict = {}
    errors = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(fn): key for key, fn in tasks}
        for future in tqdm(as_completed(futures), total=len(futures), desc=desc, unit="call"):
            key = futures[future]
            try:
                results[key] = future.result()
            except Exception as exc:
                errors += 1
                tqdm.write(f"  [error] {key}: {exc}")
    if errors:
        tqdm.write(f"  [warn] {errors}/{len(tasks)} call(s) failed")
    return results
