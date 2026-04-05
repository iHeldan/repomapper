import asyncio
import contextlib
import io
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import repomap
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

            repo_map = RepoMap(root=str(root))
            repo_map.get_tags = lambda fname, rel_fname: tags_by_name[Path(fname).name]

            ranked_tags, _ = repo_map.get_ranked_tags(
                [],
                [str(root / "a.py"), str(root / "b.py"), str(root / "c.py")],
            )

            ranks_by_file = {tag.rel_fname: rank for rank, tag in ranked_tags}
            self.assertGreater(ranks_by_file["c.py"], ranks_by_file["b.py"])
            self.assertGreater(ranks_by_file["b.py"], ranks_by_file["a.py"])


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
