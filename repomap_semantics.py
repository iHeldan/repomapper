"""Language-aware import/export semantics used by tracing and impact analysis."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set


JS_TS_EXTENSIONS = (".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".mts", ".cts")


@dataclass
class SemanticLink:
    source: str
    target: str
    relation: str
    line: Optional[int] = None
    source_symbol: Optional[str] = None
    target_symbol: Optional[str] = None
    detail: Optional[str] = None


def collect_semantic_links(
    rel_fname: str,
    text: str,
    known_rel_fnames: Set[str],
) -> List[SemanticLink]:
    """Extract semantic file-to-file links from imports, re-exports, and package boundaries."""
    normalized = Path(rel_fname).as_posix()
    suffix = Path(normalized).suffix.lower()

    links: List[SemanticLink] = []
    if suffix in JS_TS_EXTENSIONS:
        links.extend(_collect_js_ts_semantic_links(normalized, text, known_rel_fnames))
    elif suffix == ".py":
        links.extend(_collect_python_semantic_links(normalized, text, known_rel_fnames))

    deduped = []
    seen = set()
    for link in links:
        key = (
            link.source,
            link.target,
            link.relation,
            link.line,
            link.source_symbol,
            link.target_symbol,
            link.detail,
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(link)
    return deduped


def _normalize_known_path(path: Path) -> str:
    return os.path.normpath(path.as_posix()).replace("\\", "/").lstrip("./")


def _expand_js_ts_suffix_candidates(path: Path) -> List[Path]:
    """Expand compiled JS specifiers back to plausible source files."""
    candidates = [path]
    suffix = path.suffix.lower()
    if suffix not in JS_TS_EXTENSIONS:
        return candidates

    stem = path.with_suffix("")
    for alt_suffix in JS_TS_EXTENSIONS:
        candidate = Path(f"{stem.as_posix()}{alt_suffix}")
        if candidate != path:
            candidates.append(candidate)
    return candidates


def _candidate_symbols(spec: str) -> List[tuple[Optional[str], Optional[str]]]:
    spec = spec.strip()
    if not spec:
        return [(None, None)]
    if spec.startswith("type "):
        spec = spec[5:].strip()

    pieces: List[tuple[Optional[str], Optional[str]]] = []
    named_match = re.search(r"\{([^}]*)\}", spec)
    if named_match:
        prefix = spec[: named_match.start()].strip().rstrip(",").strip()
        if prefix:
            pieces.extend(_candidate_symbols(prefix))
        named_body = named_match.group(1).strip()
        if named_body:
            pieces.extend(_candidate_symbols(named_body))
        return pieces or [(None, None)]

    cleaned = spec.strip("{} ").strip()
    if not cleaned:
        return [(None, None)]

    if cleaned.startswith("*"):
        alias_match = re.search(r"\bas\s+([A-Za-z_][A-Za-z0-9_]*)", cleaned)
        alias = alias_match.group(1) if alias_match else "*"
        return [(alias, "*")]

    for item in cleaned.split(","):
        token = item.strip()
        if not token:
            continue
        if token.startswith("type "):
            token = token[5:].strip()
        alias_match = re.match(r"([A-Za-z_][A-Za-z0-9_]*)\s+as\s+([A-Za-z_][A-Za-z0-9_]*)$", token)
        if alias_match:
            pieces.append((alias_match.group(2), alias_match.group(1)))
            continue
        pieces.append((token, token))

    return pieces or [(None, None)]


def _resolve_js_ts_module(
    rel_fname: str,
    module_spec: str,
    known_rel_fnames: Set[str],
) -> Optional[str]:
    module_spec = module_spec.strip().strip("'\"")
    if not module_spec:
        return None

    parent = Path(rel_fname).parent
    spec_path = Path(module_spec)
    candidates: List[Path] = []

    if module_spec.startswith("."):
        base = (parent / spec_path).as_posix()
        base_path = Path(base)
        if base_path.suffix:
            candidates.extend(_expand_js_ts_suffix_candidates(base_path))
        else:
            candidates.append(base_path)
            for suffix in JS_TS_EXTENSIONS:
                candidates.append(Path(f"{base}{suffix}"))
            for suffix in JS_TS_EXTENSIONS:
                candidates.append(base_path / f"index{suffix}")
    else:
        base_path = Path(module_spec)
        if base_path.suffix:
            candidates.extend(_expand_js_ts_suffix_candidates(base_path))
        else:
            candidates.append(base_path)
            for suffix in JS_TS_EXTENSIONS:
                candidates.append(Path(f"{module_spec}{suffix}"))
            for suffix in JS_TS_EXTENSIONS:
                candidates.append(base_path / f"index{suffix}")

    for candidate in candidates:
        normalized = _normalize_known_path(candidate)
        if normalized in known_rel_fnames:
            return normalized
    return None


def _collect_js_ts_semantic_links(
    rel_fname: str,
    text: str,
    known_rel_fnames: Set[str],
) -> List[SemanticLink]:
    links: List[SemanticLink] = []

    import_from_pattern = re.compile(
        r'^\s*import\s+([\s\S]*?)\s+from\s+["\']([^"\']+)["\']\s*;?',
        re.MULTILINE,
    )
    export_from_pattern = re.compile(
        r'^\s*export\s+([\s\S]*?)\s+from\s+["\']([^"\']+)["\']\s*;?',
        re.MULTILINE,
    )
    require_pattern = re.compile(
        r'^\s*(?:const|let|var)\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*require\(\s*["\']([^"\']+)["\']\s*\)\s*;?\s*$'
    )

    for match in import_from_pattern.finditer(text):
        symbol_spec, module_spec = match.groups()
        target = _resolve_js_ts_module(rel_fname, module_spec, known_rel_fnames)
        if not target:
            continue
        line_num = text.count("\n", 0, match.start()) + 1
        normalized_spec = " ".join(symbol_spec.strip().split())
        for source_symbol, target_symbol in _candidate_symbols(symbol_spec):
            detail = f"import {normalized_spec} from {module_spec}"
            links.append(
                SemanticLink(
                    source=rel_fname,
                    target=target,
                    relation="imports",
                    line=line_num,
                    source_symbol=source_symbol,
                    target_symbol=target_symbol,
                    detail=detail,
                )
            )

    for match in export_from_pattern.finditer(text):
        symbol_spec, module_spec = match.groups()
        target = _resolve_js_ts_module(rel_fname, module_spec, known_rel_fnames)
        if not target:
            continue
        line_num = text.count("\n", 0, match.start()) + 1
        normalized_spec = " ".join(symbol_spec.strip().split())
        for source_symbol, target_symbol in _candidate_symbols(symbol_spec):
            detail = f"export {normalized_spec} from {module_spec}"
            links.append(
                SemanticLink(
                    source=rel_fname,
                    target=target,
                    relation="re_exports",
                    line=line_num,
                    source_symbol=source_symbol,
                    target_symbol=target_symbol,
                    detail=detail,
                )
            )

    for line_num, line in enumerate(text.splitlines(), start=1):
        match = require_pattern.match(line)
        if match:
            alias, module_spec = match.groups()
            target = _resolve_js_ts_module(rel_fname, module_spec, known_rel_fnames)
            if not target:
                continue
            links.append(
                SemanticLink(
                    source=rel_fname,
                    target=target,
                    relation="imports",
                    line=line_num,
                    source_symbol=alias,
                    target_symbol=alias,
                    detail=f"require({module_spec})",
                )
            )

    return links


def _resolve_python_module(
    rel_fname: str,
    module_spec: str,
    known_rel_fnames: Set[str],
    level: int = 0,
) -> Optional[str]:
    rel_path = Path(rel_fname)
    package_parts = list(rel_path.parent.parts)
    if level > 0:
        trim = max(level - 1, 0)
        if trim:
            package_parts = package_parts[:-trim] if trim <= len(package_parts) else []

    module_parts = [part for part in module_spec.split(".") if part] if module_spec else []
    combined_parts = package_parts + module_parts

    candidates: List[Path] = []
    if combined_parts:
        module_path = Path(*combined_parts)
        candidates.append(Path(f"{module_path.as_posix()}.py"))
        candidates.append(module_path / "__init__.py")
    elif package_parts:
        module_path = Path(*package_parts)
        candidates.append(module_path / "__init__.py")

    for candidate in candidates:
        normalized = _normalize_known_path(candidate)
        if normalized in known_rel_fnames:
            return normalized
    return None


def _resolve_python_relative_symbol_module(
    rel_fname: str,
    imported_symbol: str,
    known_rel_fnames: Set[str],
    level: int,
) -> Optional[str]:
    return _resolve_python_module(rel_fname, imported_symbol, known_rel_fnames, level=level)


def _split_import_names(spec: str) -> List[tuple[str, str]]:
    cleaned = spec.strip().strip("()")
    if not cleaned:
        return []

    names = []
    for item in cleaned.split(","):
        token = item.strip()
        if not token:
            continue
        alias_match = re.match(r"([A-Za-z_][A-Za-z0-9_]*)\s+as\s+([A-Za-z_][A-Za-z0-9_]*)$", token)
        if alias_match:
            names.append((alias_match.group(2), alias_match.group(1)))
            continue
        names.append((token, token))
    return names


def _collect_python_semantic_links(
    rel_fname: str,
    text: str,
    known_rel_fnames: Set[str],
) -> List[SemanticLink]:
    links: List[SemanticLink] = []

    from_pattern = re.compile(r'^\s*from\s+(\.*)([A-Za-z_][A-Za-z0-9_\.]*)?\s+import\s+(.+?)\s*$')
    import_pattern = re.compile(r'^\s*import\s+(.+?)\s*$')

    for line_num, line in enumerate(text.splitlines(), start=1):
        match = from_pattern.match(line)
        if match:
            dots, module_spec, imported_spec = match.groups()
            level = len(dots or "")
            imported_names = _split_import_names(imported_spec)
            module_target = _resolve_python_module(rel_fname, module_spec or "", known_rel_fnames, level=level)
            relation = "package_reexports" if Path(rel_fname).name == "__init__.py" else "imports"

            if module_target:
                for source_symbol, target_symbol in imported_names or [(None, None)]:
                    links.append(
                        SemanticLink(
                            source=rel_fname,
                            target=module_target,
                            relation=relation,
                            line=line_num,
                            source_symbol=source_symbol,
                            target_symbol=target_symbol,
                            detail=f"from {dots}{module_spec or ''} import {imported_spec.strip()}",
                        )
                    )

            if not module_spec:
                for source_symbol, target_symbol in imported_names:
                    symbol_target = _resolve_python_relative_symbol_module(
                        rel_fname,
                        target_symbol,
                        known_rel_fnames,
                        level=max(level, 1),
                    )
                    if not symbol_target:
                        continue
                    links.append(
                        SemanticLink(
                            source=rel_fname,
                            target=symbol_target,
                            relation=relation,
                            line=line_num,
                            source_symbol=source_symbol,
                            target_symbol=target_symbol,
                            detail=f"from {dots or '.'} import {target_symbol}",
                        )
                    )
            continue

        match = import_pattern.match(line)
        if not match:
            continue

        for raw_name in match.group(1).split(","):
            token = raw_name.strip()
            if not token:
                continue
            alias_match = re.match(r"([A-Za-z_][A-Za-z0-9_\.]*)\s+as\s+([A-Za-z_][A-Za-z0-9_]*)$", token)
            if alias_match:
                module_spec, alias = alias_match.groups()
                source_symbol = alias
            else:
                module_spec = token
                source_symbol = module_spec.split(".")[-1]

            target = _resolve_python_module(rel_fname, module_spec, known_rel_fnames, level=0)
            if not target:
                continue

            links.append(
                SemanticLink(
                    source=rel_fname,
                    target=target,
                    relation="imports",
                    line=line_num,
                    source_symbol=source_symbol,
                    target_symbol=module_spec.split(".")[-1],
                    detail=f"import {token}",
                )
            )

    return links
