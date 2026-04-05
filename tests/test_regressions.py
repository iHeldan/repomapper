import asyncio
import contextlib
import io
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import parser_support
import repomap
import repomap_class
import repomap_server
from repomap_class import FileReport, RepoMap, Tag
from utils import find_src_files, is_within_directory


class RepoMapRankingTests(unittest.TestCase):
    def test_pagerank_runs_without_chat_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir).resolve()
            for name in ("a.py", "b.py", "c.py"):
                (root / name).write_text("# test\n", encoding="utf-8")

            tags_by_name = {
                "a.py": [
                    Tag("a.py", str(root / "a.py"), 1, "A", "def"),
                    Tag("a.py", str(root / "a.py"), 1, "B", "ref"),
                ],
                "b.py": [
                    Tag("b.py", str(root / "b.py"), 1, "B", "def"),
                    Tag("b.py", str(root / "b.py"), 1, "C", "ref"),
                ],
                "c.py": [
                    Tag("c.py", str(root / "c.py"), 1, "C", "def"),
                ],
            }

            repo_map = RepoMap(root=str(root), token_counter_func=lambda text: len(text.split()))
            repo_map.get_tags = lambda fname, rel_fname: tags_by_name[Path(fname).name]

            ranked_tags, _ = repo_map.get_ranked_tags(
                [],
                [str(root / "a.py"), str(root / "b.py"), str(root / "c.py")],
            )

            ranks_by_file = {tag.rel_fname: rank for rank, tag in ranked_tags}
            self.assertGreater(ranks_by_file["c.py"], ranks_by_file["b.py"])
            self.assertGreater(ranks_by_file["b.py"], ranks_by_file["a.py"])

    def test_important_files_without_parser_tags_still_appear_in_map(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir).resolve()
            readme = root / "README.md"
            source = root / "main.py"
            readme.write_text("# Project Title\n\nSome high-level description.\n", encoding="utf-8")
            source.write_text("def main():\n    return 1\n", encoding="utf-8")

            repo_map = RepoMap(root=str(root), token_counter_func=lambda text: len(text.split()))

            def fake_get_tags(fname, rel_fname):
                if rel_fname == "main.py":
                    return [Tag(rel_fname, fname, 1, "main", "def")]
                return []

            repo_map.get_tags = fake_get_tags

            ranked_tags, file_report = repo_map.get_ranked_tags(
                [],
                [str(readme), str(source)],
            )
            map_content, _ = repo_map.get_ranked_tags_map_uncached(
                [],
                [str(readme), str(source)],
                max_map_tokens=4096,
            )

            self.assertTrue(any(tag.rel_fname == "README.md" and tag.kind == "doc" for _, tag in ranked_tags))
            self.assertEqual(file_report.definition_matches, 1)
            self.assertIn("README.md:", map_content)
            self.assertIn("# Project Title", map_content)


class SearchIdentifierCacheTests(unittest.TestCase):
    def tearDown(self):
        repomap_server._REPO_MAP_CACHE.clear()

    def test_search_identifiers_refreshes_cached_tags_and_context(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir).resolve()
            source_file = root / "app.py"
            source_file.write_text("def foo():\n    return 1\n", encoding="utf-8")

            def fake_get_tags_raw(self, fname, rel_fname):
                text = Path(fname).read_text(encoding="utf-8")
                if "def foo" in text:
                    return [Tag(rel_fname, fname, 1, "foo", "def")]
                if "def bar" in text:
                    return [Tag(rel_fname, fname, 1, "bar", "def")]
                return []

            with mock.patch.object(repomap_server, "_check_project_root", return_value=None):
                with mock.patch.object(RepoMap, "get_tags_raw", new=fake_get_tags_raw):
                    first = asyncio.run(repomap_server.search_identifiers(str(root), "foo"))

                    time.sleep(0.01)
                    source_file.write_text("def bar():\n    return 2\n", encoding="utf-8")

                    second = asyncio.run(repomap_server.search_identifiers(str(root), "bar"))
                    third = asyncio.run(repomap_server.search_identifiers(str(root), "foo"))

            self.assertEqual(first["results"][0]["name"], "foo")
            self.assertEqual(second["results"][0]["name"], "bar")
            self.assertIn("def bar()", second["results"][0]["context"])
            self.assertEqual(third["results"], [])

    def test_search_identifiers_rebuilds_index_when_parser_state_changes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir).resolve()
            source_file = root / "app.py"
            source_file.write_text("def foo():\n    return 1\n", encoding="utf-8")

            parser_states = [[], ["python"]]

            def fake_get_tags_raw(self, fname, rel_fname):
                if parser_states[0]:
                    self._clear_tag_failure(fname)
                    return [Tag(rel_fname, fname, 1, "foo", "def")]
                self._record_tag_failure(fname, "parser missing", cacheable=False)
                return []

            with mock.patch.object(repomap_server, "_check_project_root", return_value=None):
                with mock.patch.object(repomap_server, "get_downloaded_parser_languages", side_effect=lambda: parser_states[0]):
                    with mock.patch.object(RepoMap, "get_tags_raw", new=fake_get_tags_raw):
                        first = asyncio.run(repomap_server.search_identifiers(str(root), "foo"))
                        parser_states[0] = ["python"]
                        second = asyncio.run(repomap_server.search_identifiers(str(root), "foo"))

            self.assertEqual(first["results"], [])
            self.assertEqual(second["results"][0]["name"], "foo")


class CliPathResolutionTests(unittest.TestCase):
    def test_cli_resolves_relative_paths_against_root(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir).resolve()
            pkg_dir = root / "pkg"
            pkg_dir.mkdir()
            chat_file = pkg_dir / "focus.py"
            other_file = pkg_dir / "mod.py"
            chat_file.write_text("def focus():\n    return 1\n", encoding="utf-8")
            other_file.write_text("def foo():\n    return 1\n", encoding="utf-8")

            captured = {}

            def fake_get_repo_map(self, chat_files=None, other_files=None, **kwargs):
                captured["chat_files"] = chat_files
                captured["other_files"] = other_files
                return None, FileReport({}, 0, 0, len(other_files or []))

            with mock.patch.object(repomap.RepoMap, "get_repo_map", new=fake_get_repo_map):
                with mock.patch.object(
                    sys,
                    "argv",
                    [
                        "repomap.py",
                        "--root",
                        str(root),
                        "--chat-files",
                        "pkg/focus.py",
                        "--other-files",
                        "pkg/mod.py",
                    ],
                ):
                    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                        repomap.main()

                self.assertEqual(captured["chat_files"], [str(chat_file.resolve())])
                self.assertEqual(captured["other_files"], [str(other_file.resolve())])

                captured.clear()
                with mock.patch.object(sys, "argv", ["repomap.py", "--root", str(root), "pkg/mod.py"]):
                    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                        repomap.main()

                self.assertEqual(captured["other_files"], [str(other_file.resolve())])

                captured.clear()
                with mock.patch.object(sys, "argv", ["repomap.py", "--root", str(root)]):
                    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                        repomap.main()

                self.assertEqual(
                    sorted(captured["other_files"]),
                    sorted([str(chat_file.resolve()), str(other_file.resolve())]),
                )

    def test_cli_can_warm_parsers_for_selected_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir).resolve()
            source_file = root / "app.py"
            source_file.write_text("def foo():\n    return 1\n", encoding="utf-8")

            def fake_get_repo_map(self, chat_files=None, other_files=None, **kwargs):
                return None, FileReport({}, 0, 0, len(other_files or []))

            warmup_result = parser_support.ParserWarmupResult(
                requested=["python"],
                available=["python"],
                downloaded=["python"],
            )

            with mock.patch.object(repomap.RepoMap, "get_repo_map", new=fake_get_repo_map):
                with mock.patch.object(repomap, "infer_parser_languages", return_value=["python"]) as infer_mock:
                    with mock.patch.object(repomap, "warm_languages", return_value=warmup_result) as warm_mock:
                        with mock.patch.object(
                            sys,
                            "argv",
                            ["repomap.py", "--root", str(root), "--download-missing-parsers", "app.py"],
                        ):
                            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                                repomap.main()

            infer_mock.assert_called_once()
            warm_mock.assert_called_once_with(["python"])


class ParserSupportTests(unittest.TestCase):
    def test_infer_parser_languages_handles_vue_and_tsx(self):
        inferred = parser_support.infer_parser_languages(
            ["src/app.tsx", "src/component.vue", "README.md", "src/util.py"]
        )
        self.assertEqual(inferred, ["javascript", "python", "tsx", "typescript"])

    def test_warm_languages_reports_downloaded_runtimes(self):
        fake_tslp = mock.Mock()
        fake_tslp.downloaded_languages.side_effect = [[], ["python"]]

        with mock.patch.object(parser_support, "_HAS_LANGUAGE_PACK", True):
            with mock.patch.object(parser_support, "tslp", fake_tslp):
                result = parser_support.warm_languages(["python"])

        fake_tslp.download.assert_called_once_with(["python"])
        self.assertEqual(result.downloaded, ["python"])
        self.assertEqual(result.missing, [])

    def test_tsx_files_use_tsx_parser(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir).resolve()
            source_file = root / "component.tsx"
            source_file.write_text("export const Component = <div />\n", encoding="utf-8")

            calls = []

            class DummyParser:
                def parse(self, code):
                    return type("Tree", (), {"root_node": object()})()

            class DummyQuery:
                def __init__(self, language, query_text):
                    self.language = language
                    self.query_text = query_text

            class DummyNode:
                start_point = (0, 0)
                text = b"Component"

            class DummyCursor:
                def __init__(self, query):
                    self.query = query

                def captures(self, root_node):
                    return {"name.definition": [DummyNode()]}

            def fake_get_language(name):
                calls.append(("language", name))
                return object()

            def fake_get_parser(name):
                calls.append(("parser", name))
                return DummyParser()

            repo_map = RepoMap(root=str(root))

            with mock.patch.object(repomap_class, "get_language", side_effect=fake_get_language):
                with mock.patch.object(repomap_class, "get_parser", side_effect=fake_get_parser):
                    with mock.patch.object(repomap_class, "Query", DummyQuery):
                        with mock.patch.object(repomap_class, "QueryCursor", DummyCursor):
                            with mock.patch.object(repomap_class, "read_text", return_value="(query)"):
                                tags = repo_map.get_tags_raw(str(source_file), "component.tsx")

            self.assertEqual(tags[0].name, "Component")
            self.assertIn(("language", "tsx"), calls)
            self.assertIn(("parser", "tsx"), calls)

    def test_parser_bootstrap_failures_are_not_cached(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir).resolve()
            source_file = root / "app.py"
            source_file.write_text("def foo():\n    return 1\n", encoding="utf-8")

            attempts = {"count": 0}

            class DummyParser:
                def parse(self, code):
                    return type("Tree", (), {"root_node": object()})()

            class DummyQuery:
                def __init__(self, language, query_text):
                    self.language = language
                    self.query_text = query_text

            class DummyNode:
                start_point = (0, 0)
                text = b"foo"

            class DummyCursor:
                def __init__(self, query):
                    self.query = query

                def captures(self, root_node):
                    return {"name.definition": [DummyNode()]}

            def fake_get_language(name):
                attempts["count"] += 1
                if attempts["count"] == 1:
                    raise Exception("Failed to fetch manifest from https://example.invalid/parsers.json")
                return object()

            repo_map = RepoMap(root=str(root))

            with mock.patch.object(repomap_class, "get_language", side_effect=fake_get_language):
                with mock.patch.object(repomap_class, "get_parser", return_value=DummyParser()):
                    with mock.patch.object(repomap_class, "Query", DummyQuery):
                        with mock.patch.object(repomap_class, "QueryCursor", DummyCursor):
                            with mock.patch.object(repomap_class, "read_text", return_value="(query)"):
                                first = repo_map.get_tags(str(source_file), "app.py")
                                second = repo_map.get_tags(str(source_file), "app.py")

            self.assertEqual(first, [])
            self.assertEqual(second[0].name, "foo")
            self.assertEqual(attempts["count"], 2)
            self.assertTrue(any("--download-missing-parsers" in message for message in repo_map.diagnostics))


class SymlinkContainmentTests(unittest.TestCase):
    def test_path_containment_rejects_symlink_escaping_root(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir).resolve()
            root = base / "root"
            outside = base / "root-evil"
            root.mkdir()
            outside.mkdir()
            (root / "sub").mkdir()

            secret = outside / "secret.py"
            secret.write_text("secret = 1\n", encoding="utf-8")
            link = root / "sub" / "link.py"
            link.symlink_to(secret)

            self.assertFalse(is_within_directory(str(link), str(root)))
            self.assertFalse(repomap_server._validate_path_containment("sub/link.py", str(root)))

    def test_find_src_files_skips_symlinked_files_outside_root(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir).resolve()
            root = base / "root"
            outside = base / "root-evil"
            root.mkdir()
            outside.mkdir()
            (root / "sub").mkdir()
            (root / "sub" / "real.py").write_text("value = 1\n", encoding="utf-8")

            secret = outside / "secret.py"
            secret.write_text("secret = 1\n", encoding="utf-8")
            (root / "sub" / "link.py").symlink_to(secret)

            found = sorted(str(Path(path).relative_to(root)) for path in find_src_files(str(root)))
            self.assertEqual(found, ["sub/real.py"])


if __name__ == "__main__":
    unittest.main()
