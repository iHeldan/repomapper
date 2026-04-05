"""Shared RepoMap report and payload dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class RankingReason:
    code: str
    message: str


@dataclass
class RankedFile:
    path: str
    rank: float
    base_rank: float
    included_in_map: bool = False
    is_changed_file: bool = False
    changed_neighbor_distance: Optional[int] = None
    is_test_file: bool = False
    is_entrypoint_file: bool = False
    is_public_api_file: bool = False
    is_chat_file: bool = False
    is_mentioned_file: bool = False
    is_important_file: bool = False
    definitions: int = 0
    references: int = 0
    referenced_by_files: int = 0
    references_to_files: int = 0
    matched_query_terms: List[str] = field(default_factory=list)
    matched_query_path_terms: List[str] = field(default_factory=list)
    matched_query_symbol_terms: List[str] = field(default_factory=list)
    related_changed_files: List[str] = field(default_factory=list)
    related_tests: List[str] = field(default_factory=list)
    related_sources: List[str] = field(default_factory=list)
    entrypoint_signals: List[str] = field(default_factory=list)
    public_api_signals: List[str] = field(default_factory=list)
    summary_kind: Optional[str] = None
    summary_items: List[str] = field(default_factory=list)
    mentioned_identifiers: List[str] = field(default_factory=list)
    sample_symbols: List[str] = field(default_factory=list)
    inbound_neighbors: List[str] = field(default_factory=list)
    outbound_neighbors: List[str] = field(default_factory=list)
    lines_of_interest: List[int] = field(default_factory=list)
    reasons: List[RankingReason] = field(default_factory=list)


@dataclass
class FileReport:
    excluded: Dict[str, str]
    definition_matches: int
    reference_matches: int
    total_files_considered: int
    diagnostics: List[str] = field(default_factory=list)
    query: Optional[str] = None
    query_terms: List[str] = field(default_factory=list)
    changed_files: List[str] = field(default_factory=list)
    changed_neighbor_depth: int = 0
    ranked_files: List[RankedFile] = field(default_factory=list)
    selected_files: List[str] = field(default_factory=list)
    map_tokens: int = 0
    map_token_budget: int = 0
    map_token_budget_mode: str = "fixed"
    map_token_budget_request: Optional[str] = None
    map_token_budget_reason: Optional[str] = None


@dataclass
class ConnectionStep:
    source: str
    target: str
    relation: str
    symbols: List[str] = field(default_factory=list)
    symbol_hops: List["SymbolTraceHop"] = field(default_factory=list)


@dataclass
class SymbolTraceHop:
    source_file: str
    target_file: str
    relation: str
    source_symbol: Optional[str] = None
    target_symbol: Optional[str] = None
    source_line: Optional[int] = None
    target_line: Optional[int] = None
    evidence_kind: Optional[str] = None
    detail: Optional[str] = None


@dataclass
class ConnectionReport:
    start_file: str
    end_file: str
    path: List[str] = field(default_factory=list)
    steps: List[ConnectionStep] = field(default_factory=list)
    symbol_path: List[SymbolTraceHop] = field(default_factory=list)
    diagnostics: List[str] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class ImpactTarget:
    path: str
    seed_file: str
    distance: int
    path_from_seed: List[str] = field(default_factory=list)
    steps: List[ConnectionStep] = field(default_factory=list)
    symbol_path: List[SymbolTraceHop] = field(default_factory=list)
    seed_focus_lines: List[int] = field(default_factory=list)
    seed_hunks: List["ImpactHunk"] = field(default_factory=list)
    changed_boundary_symbols: List[str] = field(default_factory=list)
    changed_boundary_distances: Dict[str, int] = field(default_factory=dict)
    closest_changed_hunk_distance: Optional[int] = None
    boundary_symbols: List[str] = field(default_factory=list)
    boundary_relations: List[str] = field(default_factory=list)
    boundary_locations: List["ImpactLocation"] = field(default_factory=list)
    boundary_snippets: List["ImpactSnippet"] = field(default_factory=list)
    focus_lines: List[int] = field(default_factory=list)
    is_test_file: bool = False
    is_entrypoint_file: bool = False
    is_public_api_file: bool = False
    is_important_file: bool = False
    summary_kind: Optional[str] = None
    summary_items: List[str] = field(default_factory=list)
    reasons: List[RankingReason] = field(default_factory=list)


@dataclass
class ImpactSuggestion:
    priority: int
    kind: str
    target: str
    message: str
    seed_file: Optional[str] = None
    path_from_seed: List[str] = field(default_factory=list)
    anchor_file: Optional[str] = None
    anchor_line: Optional[int] = None
    anchor_symbol: Optional[str] = None
    anchor_kind: Optional[str] = None
    anchor_excerpt: Optional[str] = None


@dataclass
class ImpactQuickAction:
    priority: int
    kind: str
    target: str
    message: str
    effort: str = "small"
    target_role: str = "boundary"
    risk_level: str = "low"
    confidence: float = 0.5
    focus_symbols: List[str] = field(default_factory=list)
    focus_reason: Optional[str] = None
    why_now: Optional[str] = None
    expected_outcome: Optional[str] = None
    follow_if_true: Optional[str] = None
    follow_if_false: Optional[str] = None
    location_hint: Optional[str] = None
    command_hint: Optional[str] = None
    seed_file: Optional[str] = None
    path_from_seed: List[str] = field(default_factory=list)
    anchor_file: Optional[str] = None
    anchor_line: Optional[int] = None
    anchor_symbol: Optional[str] = None
    anchor_kind: Optional[str] = None
    anchor_excerpt: Optional[str] = None


@dataclass
class ImpactEditCandidate:
    path: str
    target_role: str
    reason: str
    priority: int = 0
    confidence: float = 0.5
    line: Optional[int] = None
    symbol: Optional[str] = None
    symbol_kind: Optional[str] = None
    location_hint: Optional[str] = None
    seed_file: Optional[str] = None
    path_from_seed: List[str] = field(default_factory=list)
    source_action_kind: Optional[str] = None
    source_action_target: Optional[str] = None


@dataclass
class ImpactEditPlanStep:
    step: int
    priority: int
    title: str
    instruction: str
    target: str
    target_role: str
    confidence: float = 0.5
    action_kind: Optional[str] = None
    location_hint: Optional[str] = None
    command_hint: Optional[str] = None
    focus_symbols: List[str] = field(default_factory=list)
    why_now: Optional[str] = None
    expected_outcome: Optional[str] = None
    follow_if_true: Optional[str] = None
    follow_if_false: Optional[str] = None
    seed_file: Optional[str] = None
    path_from_seed: List[str] = field(default_factory=list)
    edit_candidates: List[ImpactEditCandidate] = field(default_factory=list)


@dataclass
class ImpactTestCluster:
    kind: str
    seed_file: str
    paths: List[str] = field(default_factory=list)
    covers: List[str] = field(default_factory=list)
    closest_distance: Optional[int] = None
    focus_symbols: List[str] = field(default_factory=list)
    command_hint: Optional[str] = None
    reason: Optional[str] = None


@dataclass
class ImpactLocation:
    file: str
    line: int
    kind: str
    symbol: str


@dataclass
class ImpactSnippet:
    file: str
    start_line: int
    end_line: int
    highlight_line: int
    kind: str
    symbol: str
    excerpt: str


@dataclass
class ImpactHunk:
    start_line: int
    end_line: int


@dataclass
class ImpactSymbol:
    name: str
    target_files: List[str] = field(default_factory=list)
    seed_files: List[str] = field(default_factory=list)
    target_count: int = 0
    closest_distance: Optional[int] = None
    is_changed_seed_symbol: bool = False
    closest_changed_hunk_distance: Optional[int] = None
    locations: List[ImpactLocation] = field(default_factory=list)


@dataclass
class ImpactReport:
    seed_files: List[str]
    max_depth: int
    max_results: int
    impacted_files: List[ImpactTarget] = field(default_factory=list)
    changed_lines_by_file: Dict[str, List[int]] = field(default_factory=dict)
    changed_hunks_by_file: Dict[str, List[ImpactHunk]] = field(default_factory=dict)
    changed_seed_symbols: Dict[str, List[str]] = field(default_factory=dict)
    shared_symbols: List[ImpactSymbol] = field(default_factory=list)
    quick_actions: List[ImpactQuickAction] = field(default_factory=list)
    edit_candidates: List[ImpactEditCandidate] = field(default_factory=list)
    edit_plan: List[ImpactEditPlanStep] = field(default_factory=list)
    test_clusters: List[ImpactTestCluster] = field(default_factory=list)
    suggested_checks: List[ImpactSuggestion] = field(default_factory=list)
    diagnostics: List[str] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class ReviewChangedFile:
    path: str
    target_role: str
    changed_lines: List[int] = field(default_factory=list)
    changed_hunks: List["ImpactHunk"] = field(default_factory=list)
    changed_symbols: List[str] = field(default_factory=list)
    related_tests: List[str] = field(default_factory=list)
    entrypoint_signals: List[str] = field(default_factory=list)
    public_api_signals: List[str] = field(default_factory=list)
    summary_kind: Optional[str] = None
    summary_items: List[str] = field(default_factory=list)
    is_test_file: bool = False
    is_entrypoint_file: bool = False
    is_public_api_file: bool = False
    is_important_file: bool = False


@dataclass
class ReviewFocusItem:
    priority: int
    kind: str
    title: str
    target: str
    target_role: str
    message: str
    risk_level: str = "low"
    confidence: float = 0.5
    location_hint: Optional[str] = None
    command_hint: Optional[str] = None
    focus_symbols: List[str] = field(default_factory=list)
    why_now: Optional[str] = None
    expected_outcome: Optional[str] = None
    follow_if_true: Optional[str] = None
    follow_if_false: Optional[str] = None
    seed_file: Optional[str] = None
    path_from_seed: List[str] = field(default_factory=list)
    anchor_file: Optional[str] = None
    anchor_line: Optional[int] = None
    anchor_symbol: Optional[str] = None
    anchor_kind: Optional[str] = None


@dataclass
class ReviewReport:
    current_branch: Optional[str]
    base_ref: Optional[str]
    max_depth: int
    max_results: int
    changed_files: List[ReviewChangedFile] = field(default_factory=list)
    changed_lines_by_file: Dict[str, List[int]] = field(default_factory=dict)
    changed_hunks_by_file: Dict[str, List["ImpactHunk"]] = field(default_factory=dict)
    changed_seed_symbols: Dict[str, List[str]] = field(default_factory=dict)
    impacted_files: List[ImpactTarget] = field(default_factory=list)
    shared_symbols: List[ImpactSymbol] = field(default_factory=list)
    quick_actions: List[ImpactQuickAction] = field(default_factory=list)
    edit_candidates: List[ImpactEditCandidate] = field(default_factory=list)
    edit_plan: List[ImpactEditPlanStep] = field(default_factory=list)
    test_clusters: List[ImpactTestCluster] = field(default_factory=list)
    suggested_checks: List[ImpactSuggestion] = field(default_factory=list)
    review_focus: List[ReviewFocusItem] = field(default_factory=list)
    changed_public_api_files: List[str] = field(default_factory=list)
    changed_entrypoint_files: List[str] = field(default_factory=list)
    changed_test_files: List[str] = field(default_factory=list)
    changed_config_files: List[str] = field(default_factory=list)
    diagnostics: List[str] = field(default_factory=list)
    error: Optional[str] = None


__all__ = [
    "ConnectionReport",
    "ConnectionStep",
    "FileReport",
    "ImpactEditCandidate",
    "ImpactEditPlanStep",
    "ImpactHunk",
    "ImpactLocation",
    "ImpactQuickAction",
    "ImpactReport",
    "ImpactSnippet",
    "ImpactSuggestion",
    "ImpactSymbol",
    "ImpactTarget",
    "ImpactTestCluster",
    "RankedFile",
    "RankingReason",
    "ReviewChangedFile",
    "ReviewFocusItem",
    "ReviewReport",
    "SymbolTraceHop",
]
