"""Map token budget parsing and heuristic resolution."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable, Optional


DEFAULT_MAP_TOKEN_BUDGET = 8192
CONTEXT_WINDOW_PADDING = 1024
AUTO_CONTEXT_EXPANSION_CAP = 16384
_BUDGET_PRESET_ORDER = ("small", "medium", "large", "xlarge")
_BUDGET_PRESET_TOKENS = {
    "small": 2048,
    "medium": 4096,
    "large": 8192,
    "xlarge": 12288,
}


@dataclass(frozen=True)
class MapBudgetRequest:
    mode: str
    raw_value: str
    requested_tokens: Optional[int] = None
    hint: Optional[str] = None
    diagnostic: Optional[str] = None


@dataclass(frozen=True)
class MapBudgetDecision:
    mode: str
    request: str
    requested_tokens: Optional[int]
    effective_tokens: int
    reason: str


def _coerce_positive_int(value: Any) -> Optional[int]:
    """Parse a positive integer budget when possible."""
    if value is None or value == "":
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def parse_map_budget_request(value: Any, default_tokens: int = DEFAULT_MAP_TOKEN_BUDGET) -> MapBudgetRequest:
    """Parse CLI/MCP token budget inputs into a normalized request."""
    if isinstance(value, MapBudgetRequest):
        return value

    if value is None or value == "":
        return MapBudgetRequest(mode="fixed", raw_value=str(default_tokens), requested_tokens=default_tokens)

    if isinstance(value, (int, float)):
        requested_tokens = _coerce_positive_int(value)
        if requested_tokens is not None:
            return MapBudgetRequest(mode="fixed", raw_value=str(int(value)), requested_tokens=requested_tokens)
        return MapBudgetRequest(
            mode="fixed",
            raw_value=str(default_tokens),
            requested_tokens=default_tokens,
            diagnostic=f"Ignoring non-positive map token budget {value!r}; falling back to {default_tokens}.",
        )

    if isinstance(value, str):
        normalized = value.strip().lower()
        if not normalized:
            return MapBudgetRequest(mode="fixed", raw_value=str(default_tokens), requested_tokens=default_tokens)
        requested_tokens = _coerce_positive_int(normalized)
        if requested_tokens is not None:
            return MapBudgetRequest(mode="fixed", raw_value=normalized, requested_tokens=requested_tokens)
        if normalized in {"auto", "dynamic"}:
            return MapBudgetRequest(mode="auto", raw_value=normalized)
        if normalized in _BUDGET_PRESET_TOKENS:
            return MapBudgetRequest(mode="ai_guided", raw_value=normalized, hint=normalized)
        return MapBudgetRequest(
            mode="fixed",
            raw_value=normalized,
            requested_tokens=default_tokens,
            diagnostic=(
                f"Ignoring unsupported map token budget {value!r}; "
                f"use an integer or one of auto|small|medium|large. Falling back to {default_tokens}."
            ),
        )

    if isinstance(value, dict):
        raw_value = json.dumps(value, sort_keys=True)
        normalized_mode = str(value.get("mode") or "").strip().lower()
        hint = str(value.get("hint") or value.get("size") or value.get("budget") or "").strip().lower() or None
        requested_tokens = _coerce_positive_int(
            value.get("requested_tokens", value.get("tokens", value.get("token_limit")))
        )

        if not normalized_mode:
            if requested_tokens is not None:
                normalized_mode = "fixed"
            elif hint in _BUDGET_PRESET_TOKENS:
                normalized_mode = "ai_guided"
            else:
                normalized_mode = "auto"

        if normalized_mode in {"auto", "dynamic"}:
            return MapBudgetRequest(mode="auto", raw_value=raw_value, hint=hint)

        if normalized_mode in {"ai", "ai_guided", "guided"}:
            if hint not in _BUDGET_PRESET_TOKENS:
                hint = "medium"
            return MapBudgetRequest(mode="ai_guided", raw_value=raw_value, hint=hint, requested_tokens=requested_tokens)

        if normalized_mode == "fixed":
            if requested_tokens is None:
                requested_tokens = default_tokens
            return MapBudgetRequest(mode="fixed", raw_value=raw_value, requested_tokens=requested_tokens)

        return MapBudgetRequest(
            mode="fixed",
            raw_value=raw_value,
            requested_tokens=default_tokens,
            diagnostic=(
                f"Ignoring unsupported map budget mode {normalized_mode!r}; "
                f"falling back to fixed {default_tokens} tokens."
            ),
        )

    return MapBudgetRequest(
        mode="fixed",
        raw_value=str(default_tokens),
        requested_tokens=default_tokens,
        diagnostic=f"Ignoring unsupported map token budget type {type(value).__name__}; falling back to {default_tokens}.",
    )


def _choose_auto_preset(
    total_files: int,
    chat_file_count: int,
    mentioned_file_count: int,
    query_terms: Iterable[str],
    changed_file_count: int,
    changed_neighbor_depth: int,
) -> tuple[str, str]:
    """Pick a preset tier for dynamic map sizing."""
    query_terms = [term for term in query_terms if term]
    score = 0
    reasons = []

    if total_files >= 5000:
        score += 3
        reasons.append("very large repository scope")
    elif total_files >= 1000:
        score += 2
        reasons.append("large repository scope")
    elif total_files >= 250:
        score += 1
        reasons.append("medium repository scope")
    elif total_files <= 40:
        score -= 1
        reasons.append("small repository scope")

    if len(query_terms) >= 3:
        score += 1
        reasons.append("task query adds targeting needs")
    if len(query_terms) >= 7:
        score += 1
        reasons.append("task query is broad or detailed")
    if mentioned_file_count >= 3:
        score += 1
        reasons.append("multiple mentioned files widen the likely surface")

    if changed_file_count:
        score -= 1
        reasons.append("changed-file focus narrows the scope")
        if changed_neighbor_depth > 0:
            score += 1
            reasons.append("changed neighbors widen the changed-only slice")

    if chat_file_count:
        score -= 2
        reasons.append("chat files already pin the immediate focus")

    if score <= -1:
        preset = "small"
    elif score <= 1:
        preset = "medium"
    elif score <= 3:
        preset = "large"
    else:
        preset = "xlarge"

    return preset, ", ".join(reasons[:4]) if reasons else "balanced repository scope"


def resolve_map_budget(
    request: MapBudgetRequest,
    *,
    total_files: int,
    chat_file_count: int = 0,
    mentioned_file_count: int = 0,
    query_terms: Optional[Iterable[str]] = None,
    changed_file_count: int = 0,
    changed_neighbor_depth: int = 0,
    max_context_window: Optional[int] = None,
    map_mul_no_files: int = 8,
) -> MapBudgetDecision:
    """Resolve a normalized request into an effective map token budget."""
    query_terms = list(query_terms or [])

    if request.mode == "fixed":
        effective_tokens = request.requested_tokens or DEFAULT_MAP_TOKEN_BUDGET
        reason = f"Fixed map budget requested at {effective_tokens} tokens."
    else:
        if request.mode == "auto":
            preset, preset_reason = _choose_auto_preset(
                total_files,
                chat_file_count,
                mentioned_file_count,
                query_terms,
                changed_file_count,
                changed_neighbor_depth,
            )
            effective_tokens = _BUDGET_PRESET_TOKENS[preset]
            reason = (
                f"Auto budget chose {effective_tokens} tokens ({preset}) for {total_files} file(s): "
                f"{preset_reason}."
            )
        else:
            hint = request.hint if request.hint in _BUDGET_PRESET_TOKENS else "medium"
            tier_index = _BUDGET_PRESET_ORDER.index(hint)
            if total_files >= 5000 and not chat_file_count and tier_index < len(_BUDGET_PRESET_ORDER) - 1:
                tier_index += 1
            elif chat_file_count and tier_index > 0:
                tier_index -= 1
            preset = _BUDGET_PRESET_ORDER[tier_index]
            effective_tokens = _BUDGET_PRESET_TOKENS[preset]
            reason = (
                f"AI-guided budget request '{hint}' resolved to {effective_tokens} tokens ({preset}) "
                f"for {total_files} file(s)."
            )

        if not chat_file_count and max_context_window:
            available = max(0, max_context_window - CONTEXT_WINDOW_PADDING)
            expanded = min(max(1, effective_tokens) * max(1, map_mul_no_files), available)
            expanded = min(expanded, AUTO_CONTEXT_EXPANSION_CAP)
            if expanded > effective_tokens:
                effective_tokens = expanded
                reason += (
                    f" Expanded to {effective_tokens} because no chat files were pinned and the "
                    f"context window allowed a broader map."
                )

    return MapBudgetDecision(
        mode=request.mode,
        request=request.raw_value,
        requested_tokens=request.requested_tokens,
        effective_tokens=effective_tokens,
        reason=reason,
    )
