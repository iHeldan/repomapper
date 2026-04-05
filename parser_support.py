"""Helpers for selecting and warming tree-sitter parser runtimes."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Tuple

from scm import get_scm_fname

try:
    from grep_ast import filename_to_lang
except ImportError:
    filename_to_lang = None

try:
    import tree_sitter_language_pack as tslp

    _HAS_LANGUAGE_PACK = True
except ImportError:
    tslp = None
    _HAS_LANGUAGE_PACK = False


@dataclass
class ParserWarmupResult:
    requested: List[str] = field(default_factory=list)
    available: List[str] = field(default_factory=list)
    downloaded: List[str] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)
    error: str | None = None


def resolve_parser_config(file_path: str, detected_lang: str) -> Tuple[str, str]:
    """Resolve the parser language and query language for a source file."""
    suffix = Path(file_path).suffix.lower()
    if suffix == ".tsx":
        return "tsx", "typescript"
    return detected_lang, detected_lang


def expand_runtime_languages(detected_lang: str, file_path: str | None = None) -> List[str]:
    """Expand a detected language into the parser runtimes RepoMap actually needs."""
    suffix = Path(file_path).suffix.lower() if file_path else ""
    if suffix == ".tsx":
        return ["tsx"]
    if detected_lang == "vue":
        return ["javascript", "typescript", "tsx"]
    return [detected_lang]


def infer_parser_languages(file_paths: Iterable[str]) -> List[str]:
    """Infer parser runtimes required for the given files."""
    if not filename_to_lang:
        return []

    inferred = set()
    for file_path in file_paths:
        detected_lang = filename_to_lang(str(file_path))
        if not detected_lang:
            continue
        for parser_lang in expand_runtime_languages(detected_lang, str(file_path)):
            query_lang = "typescript" if parser_lang == "tsx" else parser_lang
            if get_scm_fname(query_lang):
                inferred.add(parser_lang)

    return sorted(inferred)


def format_parser_runtime_error(parser_lang: str, err: Exception) -> str:
    """Return a user-facing error for parser bootstrap/runtime failures."""
    err_text = str(err)
    if "Failed to fetch manifest" in err_text or "Failed to download" in err_text or "host not found" in err_text:
        return (
            f"Parser runtime for '{parser_lang}' is not available locally. "
            "Run with --download-missing-parsers while online, or pre-warm parsers first."
        )
    return f"Parser runtime for '{parser_lang}' is unavailable: {err_text}"


def warm_languages(languages: Iterable[str]) -> ParserWarmupResult:
    """Ensure the requested parser runtimes are present in the local cache."""
    requested = sorted(dict.fromkeys(languages))
    result = ParserWarmupResult(requested=requested)

    if not requested:
        return result

    if not _HAS_LANGUAGE_PACK:
        result.missing = requested
        result.error = "tree-sitter-language-pack is not installed."
        return result

    already_downloaded = set(tslp.downloaded_languages())
    missing = [lang for lang in requested if lang not in already_downloaded]

    if missing:
        try:
            tslp.download(missing)
        except Exception as exc:
            result.error = format_parser_runtime_error(", ".join(missing), exc)

    now_downloaded = set(tslp.downloaded_languages())
    result.available = sorted(lang for lang in requested if lang in now_downloaded)
    result.downloaded = sorted(lang for lang in result.available if lang not in already_downloaded)
    result.missing = [lang for lang in requested if lang not in now_downloaded]

    if result.missing and result.error is None:
        result.error = f"Parser runtimes still missing after download attempt: {', '.join(result.missing)}"

    return result


def get_downloaded_parser_languages() -> List[str]:
    """Return the parser runtimes currently available in the local cache."""
    if not _HAS_LANGUAGE_PACK:
        return []
    return sorted(tslp.downloaded_languages())
