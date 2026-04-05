"""Fixture-backed eval runner for RepoMapper trace and impact quality."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from repomap_class import RepoMap, Tag


@dataclass
class EvalCase:
    name: str
    mode: str
    fixture: str
    golden: str
    files: List[str]
    tags: Dict[str, List[Dict[str, Any]]]
    start_file: Optional[str] = None
    end_file: Optional[str] = None
    seed_files: Optional[List[str]] = None
    max_hops: int = 6
    max_depth: int = 2
    max_results: int = 10


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _load_cases(repo_root: Path) -> List[EvalCase]:
    manifest_path = repo_root / "evals" / "cases.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    return [EvalCase(**item) for item in data]


def _build_repo_map(case: EvalCase, fixture_root: Path) -> RepoMap:
    repo_map = RepoMap(root=str(fixture_root), token_counter_func=lambda text: len(text.split()))
    tags_by_rel = {}
    for rel_path, tag_specs in case.tags.items():
        abs_path = fixture_root / rel_path
        tags_by_rel[rel_path] = [
            Tag(rel_path, str(abs_path), int(spec["line"]), str(spec["name"]), str(spec["kind"]))
            for spec in tag_specs
        ]

    repo_map.get_tags = lambda fname, rel_fname: tags_by_rel.get(Path(rel_fname).as_posix(), [])
    return repo_map


def _normalize_trace(report) -> Dict[str, Any]:
    return {
        "path": report.path,
        "relations": [step.relation for step in report.steps],
        "symbols": [step.symbols for step in report.steps],
        "symbol_path": [
            {
                "relation": hop.relation,
                "source_file": hop.source_file,
                "target_file": hop.target_file,
                "source_symbol": hop.source_symbol,
                "target_symbol": hop.target_symbol,
                "evidence_kind": hop.evidence_kind,
            }
            for hop in report.symbol_path
        ],
    }


def _normalize_impact(report) -> Dict[str, Any]:
    return {
        "impacted_files": [
            {
                "path": target.path,
                "distance": target.distance,
                "relations": [step.relation for step in target.steps],
                "boundary_symbols": target.boundary_symbols,
                "symbol_path": [
                    {
                        "relation": hop.relation,
                        "source_file": hop.source_file,
                        "target_file": hop.target_file,
                        "source_symbol": hop.source_symbol,
                        "target_symbol": hop.target_symbol,
                        "evidence_kind": hop.evidence_kind,
                    }
                    for hop in target.symbol_path
                ],
            }
            for target in report.impacted_files
        ],
        "quick_actions": [
            {
                "kind": action.kind,
                "target": action.target,
                "target_role": action.target_role,
            }
            for action in report.quick_actions
        ],
        "edit_plan": [
            {
                "title": step.title,
                "target": step.target,
                "target_role": step.target_role,
            }
            for step in report.edit_plan
        ],
        "test_clusters": [
            {
                "kind": cluster.kind,
                "paths": cluster.paths,
            }
            for cluster in report.test_clusters
        ],
    }


def run_eval_case(case: EvalCase, *, repo_root: Optional[Path] = None, update: bool = False) -> Dict[str, Any]:
    repo_root = repo_root or _repo_root()
    fixture_root = repo_root / case.fixture
    golden_path = repo_root / case.golden
    repo_map = _build_repo_map(case, fixture_root)
    file_scope = [str(fixture_root / rel_path) for rel_path in case.files]

    started = time.perf_counter()
    if case.mode == "trace":
        report = repo_map.trace_file_path(
            str(fixture_root / (case.start_file or "")),
            str(fixture_root / (case.end_file or "")),
            files=file_scope,
            max_hops=case.max_hops,
        )
        actual = _normalize_trace(report)
    elif case.mode == "impact":
        report = repo_map.analyze_file_impact(
            [str(fixture_root / rel_path) for rel_path in (case.seed_files or [])],
            files=file_scope,
            max_depth=case.max_depth,
            max_results=case.max_results,
        )
        actual = _normalize_impact(report)
    else:
        raise ValueError(f"Unsupported eval mode: {case.mode}")
    duration_ms = round((time.perf_counter() - started) * 1000, 2)

    if update:
        golden_path.write_text(json.dumps(actual, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    expected = json.loads(golden_path.read_text(encoding="utf-8"))
    passed = actual == expected
    return {
        "name": case.name,
        "mode": case.mode,
        "passed": passed,
        "duration_ms": duration_ms,
        "expected": expected,
        "actual": actual,
    }


def run_evals(
    *,
    case_names: Optional[List[str]] = None,
    update: bool = False,
    repo_root: Optional[Path] = None,
) -> Dict[str, Any]:
    repo_root = repo_root or _repo_root()
    cases = _load_cases(repo_root)
    if case_names:
        requested = set(case_names)
        cases = [case for case in cases if case.name in requested]

    results = [run_eval_case(case, repo_root=repo_root, update=update) for case in cases]
    return {
        "total": len(results),
        "passed": sum(1 for result in results if result["passed"]),
        "failed": [result["name"] for result in results if not result["passed"]],
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RepoMapper fixture-based evals.")
    parser.add_argument("--case", action="append", dest="cases", help="Run only the named eval case")
    parser.add_argument("--update", action="store_true", help="Rewrite golden files with current outputs")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    args = parser.parse_args()

    summary = run_evals(case_names=args.cases, update=args.update)
    if args.json:
        print(json.dumps(summary, indent=2))
        return

    print(f"Eval cases: {summary['passed']}/{summary['total']} passed")
    for result in summary["results"]:
        status = "PASS" if result["passed"] else "FAIL"
        print(f"- {status} {result['name']} ({result['mode']}, {result['duration_ms']} ms)")
    if summary["failed"]:
        print("")
        print("Failed cases:")
        for name in summary["failed"]:
            print(f"- {name}")


if __name__ == "__main__":
    main()
