import asyncio
import contextlib
import io
import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import git_support
import parser_support
import repomap
import repomap_class
import repomap_server
from repomap_class import FileReport, RankedFile, RankingReason, RepoMap, Tag
from utils import find_src_files, is_within_directory


def init_git_repo(root: Path) -> None:
    subprocess.run(["git", "-C", str(root), "init", "-b", "main"], check=True, capture_output=True, text=True)
    subprocess.run(["git", "-C", str(root), "config", "user.name", "RepoMapper Tests"], check=True, capture_output=True, text=True)
    subprocess.run(["git", "-C", str(root), "config", "user.email", "tests@example.com"], check=True, capture_output=True, text=True)


def git_commit_all(root: Path, message: str) -> None:
    subprocess.run(["git", "-C", str(root), "add", "."], check=True, capture_output=True, text=True)
    subprocess.run(["git", "-C", str(root), "commit", "-m", message], check=True, capture_output=True, text=True)


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

    def test_markdown_summary_is_exposed_in_report_and_map(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir).resolve()
            readme = root / "README.md"
            source = root / "main.py"
            readme.write_text(
                "# RepoMapper\n\n"
                "A compact repository map for AI agents.\n\n"
                "## Installation\n\n"
                "Run pip install -e .\n\n"
                "## Usage\n\n"
                "Use repomap --root .\n",
                encoding="utf-8",
            )
            source.write_text("def main():\n    return 1\n", encoding="utf-8")

            repo_map = RepoMap(root=str(root), token_counter_func=lambda text: len(text.split()))
            repo_map.get_tags = lambda fname, rel_fname: [Tag(rel_fname, fname, 1, "main", "def")] if rel_fname == "main.py" else []

            map_content, file_report = repo_map.get_ranked_tags_map_uncached(
                [],
                [str(readme), str(source)],
                max_map_tokens=4096,
            )

            by_path = {entry.path: entry for entry in file_report.ranked_files}
            self.assertEqual(by_path["README.md"].summary_kind, "doc")
            self.assertIn("heading: RepoMapper", by_path["README.md"].summary_items)
            self.assertIn("overview: A compact repository map for AI agents.", by_path["README.md"].summary_items)
            self.assertIn("doc_summary", {reason.code for reason in by_path["README.md"].reasons})
            self.assertIn("(Doc Highlights)", map_content)
            self.assertIn("- heading: RepoMapper", map_content)

    def test_package_json_summary_is_exposed_in_report_and_map(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir).resolve()
            package_json = root / "package.json"
            source = root / "app.ts"
            package_json.write_text(
                json.dumps(
                    {
                        "name": "demo-app",
                        "version": "1.2.3",
                        "scripts": {"dev": "vite", "test": "vitest", "build": "vite build"},
                        "dependencies": {"react": "^18.0.0", "vite": "^5.0.0"},
                        "devDependencies": {"vitest": "^1.0.0"},
                    }
                ),
                encoding="utf-8",
            )
            source.write_text("export function app() { return true }\n", encoding="utf-8")

            repo_map = RepoMap(root=str(root), token_counter_func=lambda text: len(text.split()))
            repo_map.get_tags = lambda fname, rel_fname: [Tag(rel_fname, fname, 1, "app", "def")] if rel_fname == "app.ts" else []

            map_content, file_report = repo_map.get_ranked_tags_map_uncached(
                [],
                [str(package_json), str(source)],
                max_map_tokens=4096,
            )

            by_path = {entry.path: entry for entry in file_report.ranked_files}
            self.assertEqual(by_path["package.json"].summary_kind, "config")
            self.assertIn("package: demo-app@1.2.3", by_path["package.json"].summary_items)
            self.assertIn("scripts: dev, test, build", by_path["package.json"].summary_items)
            self.assertIn("config_summary", {reason.code for reason in by_path["package.json"].reasons})
            self.assertIn("(Config Highlights)", map_content)
            self.assertIn("- package: demo-app@1.2.3", map_content)

    def test_file_report_includes_ranked_files_reasons_and_selection(self):
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

            _, file_report = repo_map.get_ranked_tags_map_uncached(
                [],
                [str(root / "a.py"), str(root / "b.py"), str(root / "c.py")],
                max_map_tokens=4096,
            )

            by_path = {entry.path: entry for entry in file_report.ranked_files}
            self.assertEqual(file_report.selected_files, ["c.py", "b.py", "a.py"])
            self.assertGreater(file_report.map_tokens, 0)
            self.assertTrue(by_path["c.py"].included_in_map)
            self.assertEqual(by_path["c.py"].inbound_neighbors, ["b.py"])
            self.assertEqual(by_path["a.py"].outbound_neighbors, ["b.py"])
            self.assertIn("referenced_by", {reason.code for reason in by_path["c.py"].reasons})
            self.assertEqual(by_path["b.py"].sample_symbols, ["B"])

    def test_query_terms_boost_matching_paths_and_symbols(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir).resolve()
            auth_file = root / "auth_service.py"
            billing_file = root / "billing.py"
            auth_file.write_text("# auth\n", encoding="utf-8")
            billing_file.write_text("# billing\n", encoding="utf-8")

            tags_by_name = {
                "auth_service.py": [
                    Tag("auth_service.py", str(auth_file), 1, "AuthService", "def"),
                    Tag("auth_service.py", str(auth_file), 2, "login_user", "def"),
                ],
                "billing.py": [
                    Tag("billing.py", str(billing_file), 1, "BillingService", "def"),
                ],
            }

            repo_map = RepoMap(root=str(root), token_counter_func=lambda text: len(text.split()))
            repo_map.get_tags = lambda fname, rel_fname: tags_by_name[Path(fname).name]

            ranked_tags, file_report = repo_map.get_ranked_tags(
                [],
                [str(auth_file), str(billing_file)],
                query="auth login flow",
            )

            ranks_by_file = {}
            for rank, tag in ranked_tags:
                ranks_by_file[tag.rel_fname] = max(rank, ranks_by_file.get(tag.rel_fname, 0))

            by_path = {entry.path: entry for entry in file_report.ranked_files}
            self.assertEqual(file_report.query_terms, ["auth", "login", "flow"])
            self.assertGreater(ranks_by_file["auth_service.py"], ranks_by_file["billing.py"])
            self.assertEqual(by_path["auth_service.py"].matched_query_path_terms, ["auth"])
            self.assertEqual(by_path["auth_service.py"].matched_query_symbol_terms, ["auth", "login"])
            self.assertIn("query_path_match", {reason.code for reason in by_path["auth_service.py"].reasons})
            self.assertIn("query_symbol_match", {reason.code for reason in by_path["auth_service.py"].reasons})

    def test_related_tests_are_surfaced_for_high_ranked_source_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir).resolve()
            source_dir = root / "src"
            test_dir = root / "tests"
            source_dir.mkdir()
            test_dir.mkdir()
            source_file = source_dir / "auth.py"
            test_file = test_dir / "test_auth.py"
            helper_file = source_dir / "billing.py"
            source_file.write_text("def auth_login():\n    return True\n", encoding="utf-8")
            test_file.write_text("def test_auth_login():\n    assert True\n", encoding="utf-8")
            helper_file.write_text("def charge():\n    return True\n", encoding="utf-8")

            tags_by_name = {
                "auth.py": [Tag("src/auth.py", str(source_file), 1, "auth_login", "def")],
                "test_auth.py": [],
                "billing.py": [Tag("src/billing.py", str(helper_file), 1, "charge", "def")],
            }

            repo_map = RepoMap(root=str(root), token_counter_func=lambda text: len(text.split()))
            repo_map.get_tags = lambda fname, rel_fname: tags_by_name[Path(fname).name]

            map_content, file_report = repo_map.get_ranked_tags_map_uncached(
                [str(source_file)],
                [str(source_file), str(test_file), str(helper_file)],
                max_map_tokens=4096,
            )

            by_path = {entry.path: entry for entry in file_report.ranked_files}
            self.assertIn("tests/test_auth.py", by_path)
            self.assertTrue(by_path["tests/test_auth.py"].is_test_file)
            self.assertEqual(by_path["tests/test_auth.py"].related_sources, ["src/auth.py"])
            self.assertEqual(by_path["src/auth.py"].related_tests, ["tests/test_auth.py"])
            self.assertIn("related_source", {reason.code for reason in by_path["tests/test_auth.py"].reasons})
            self.assertIn("related_tests", {reason.code for reason in by_path["src/auth.py"].reasons})
            self.assertIn("tests/test_auth.py:", map_content)
            self.assertIn("def test_auth_login()", map_content)

    def test_entrypoint_and_public_api_files_are_surfaced(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir).resolve()
            api_dir = root / "api"
            api_dir.mkdir()
            app_file = root / "app.py"
            routes_file = api_dir / "routes.py"
            helper_file = root / "worker.py"
            app_file.write_text(
                "from api.routes import router\n\n"
                "def run():\n"
                "    return router\n\n"
                "if __name__ == \"__main__\":\n"
                "    run()\n",
                encoding="utf-8",
            )
            routes_file.write_text(
                "router = APIRouter()\n\n"
                "@router.get('/health')\n"
                "def health():\n"
                "    return {'ok': True}\n",
                encoding="utf-8",
            )
            helper_file.write_text("def worker():\n    return 1\n", encoding="utf-8")

            tags_by_name = {
                "app.py": [Tag("app.py", str(app_file), 3, "run", "def"), Tag("app.py", str(app_file), 1, "router", "ref")],
                "routes.py": [],
                "worker.py": [Tag("worker.py", str(helper_file), 1, "worker", "def")],
            }

            repo_map = RepoMap(root=str(root), token_counter_func=lambda text: len(text.split()))
            repo_map.get_tags = lambda fname, rel_fname: tags_by_name[Path(fname).name]

            map_content, file_report = repo_map.get_ranked_tags_map_uncached(
                [],
                [str(app_file), str(routes_file), str(helper_file)],
                max_map_tokens=4096,
            )

            by_path = {entry.path: entry for entry in file_report.ranked_files}
            self.assertTrue(by_path["app.py"].is_entrypoint_file)
            self.assertIn("entrypoint_file", {reason.code for reason in by_path["app.py"].reasons})
            self.assertIn("python_main_guard", by_path["app.py"].entrypoint_signals)
            self.assertTrue(by_path["api/routes.py"].is_public_api_file)
            self.assertIn("public_api_file", {reason.code for reason in by_path["api/routes.py"].reasons})
            self.assertIn("route_definition", by_path["api/routes.py"].public_api_signals)
            self.assertIn("if __name__ == \"__main__\":", map_content)
            self.assertIn("@router.get('/health')", map_content)

    def test_changed_neighbor_mode_surfaces_only_changed_context(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir).resolve()
            changed_file = root / "changed.py"
            neighbor_file = root / "neighbor.py"
            unrelated_file = root / "unrelated.py"
            for path in (changed_file, neighbor_file, unrelated_file):
                path.write_text("# test\n", encoding="utf-8")

            tags_by_name = {
                "changed.py": [
                    Tag("changed.py", str(changed_file), 1, "Changed", "def"),
                    Tag("changed.py", str(changed_file), 2, "Shared", "ref"),
                ],
                "neighbor.py": [
                    Tag("neighbor.py", str(neighbor_file), 1, "Shared", "def"),
                ],
                "unrelated.py": [
                    Tag("unrelated.py", str(unrelated_file), 1, "Other", "def"),
                ],
            }

            repo_map = RepoMap(root=str(root), token_counter_func=lambda text: len(text.split()))
            repo_map.get_tags = lambda fname, rel_fname: tags_by_name[Path(fname).name]

            map_content, file_report = repo_map.get_ranked_tags_map_uncached(
                [],
                [str(changed_file), str(neighbor_file), str(unrelated_file)],
                max_map_tokens=4096,
                changed_fnames={str(changed_file)},
                changed_neighbor_depth=1,
            )

            by_path = {entry.path: entry for entry in file_report.ranked_files}
            self.assertEqual(file_report.changed_files, ["changed.py"])
            self.assertEqual(file_report.changed_neighbor_depth, 1)
            self.assertIn("changed.py", by_path)
            self.assertIn("neighbor.py", by_path)
            self.assertNotIn("unrelated.py", by_path)
            self.assertTrue(by_path["changed.py"].is_changed_file)
            self.assertEqual(by_path["changed.py"].changed_neighbor_distance, 0)
            self.assertEqual(by_path["neighbor.py"].changed_neighbor_distance, 1)
            self.assertEqual(by_path["neighbor.py"].related_changed_files, ["changed.py"])
            self.assertIn("changed_file", {reason.code for reason in by_path["changed.py"].reasons})
            self.assertIn("changed_neighbor", {reason.code for reason in by_path["neighbor.py"].reasons})
            self.assertIn("neighbor.py:", map_content)
            self.assertNotIn("unrelated.py:", map_content)

    def test_trace_file_path_finds_symbol_and_test_edges(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir).resolve()
            app_file = root / "app.py"
            service_file = root / "service.py"
            test_dir = root / "tests"
            test_dir.mkdir()
            test_file = test_dir / "test_service.py"
            for path in (app_file, service_file, test_file):
                path.write_text("# test\n", encoding="utf-8")

            tags_by_name = {
                "app.py": [
                    Tag("app.py", str(app_file), 1, "run_app", "def"),
                    Tag("app.py", str(app_file), 2, "Service", "ref"),
                ],
                "service.py": [
                    Tag("service.py", str(service_file), 1, "Service", "def"),
                ],
                "test_service.py": [],
            }

            repo_map = RepoMap(root=str(root), token_counter_func=lambda text: len(text.split()))
            repo_map.get_tags = lambda fname, rel_fname: tags_by_name[Path(fname).name]

            report = repo_map.trace_file_path(
                str(app_file),
                str(test_file),
                files=[str(app_file), str(service_file), str(test_file)],
                max_hops=4,
            )

            self.assertIsNone(report.error)
            self.assertEqual(report.path, ["app.py", "service.py", "tests/test_service.py"])
            self.assertEqual(report.steps[0].relation, "references")
            self.assertEqual(report.steps[0].symbols, ["Service"])
            self.assertEqual(report.steps[1].relation, "related_test")

    def test_analyze_file_impact_surfaces_neighbors_and_tests(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir).resolve()
            app_file = root / "app.py"
            service_file = root / "service.py"
            db_file = root / "db.py"
            test_dir = root / "tests"
            test_dir.mkdir()
            test_file = test_dir / "test_service.py"
            for path in (app_file, service_file, db_file, test_file):
                path.write_text("# test\n", encoding="utf-8")

            tags_by_name = {
                "app.py": [
                    Tag("app.py", str(app_file), 1, "run_app", "def"),
                    Tag("app.py", str(app_file), 2, "Service", "ref"),
                ],
                "service.py": [
                    Tag("service.py", str(service_file), 1, "Service", "def"),
                    Tag("service.py", str(service_file), 2, "db_query", "ref"),
                ],
                "db.py": [
                    Tag("db.py", str(db_file), 1, "db_query", "def"),
                ],
                "test_service.py": [],
            }

            repo_map = RepoMap(root=str(root), token_counter_func=lambda text: len(text.split()))
            repo_map.get_tags = lambda fname, rel_fname: tags_by_name[Path(fname).name]

            report = repo_map.analyze_file_impact(
                [str(app_file)],
                files=[str(app_file), str(service_file), str(db_file), str(test_file)],
                max_depth=2,
                max_results=5,
            )

            self.assertIsNone(report.error)
            self.assertEqual(report.seed_files, ["app.py"])
            by_path = {entry.path: entry for entry in report.impacted_files}
            self.assertEqual(by_path["service.py"].distance, 1)
            self.assertEqual(by_path["service.py"].path_from_seed, ["app.py", "service.py"])
            self.assertEqual(by_path["service.py"].steps[0].relation, "references")
            self.assertIn("impact_path", {reason.code for reason in by_path["service.py"].reasons})
            self.assertIn("db.py", by_path)
            self.assertEqual(by_path["db.py"].path_from_seed, ["app.py", "service.py", "db.py"])
            self.assertIn("tests/test_service.py", by_path)
            self.assertTrue(by_path["tests/test_service.py"].is_test_file)
            self.assertIn("impact_test", {reason.code for reason in by_path["tests/test_service.py"].reasons})


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

    def test_cli_changed_mode_uses_git_worktree_selection(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir).resolve()
            init_git_repo(root)
            stable = root / "stable.py"
            changed = root / "changed.py"
            stable.write_text("STABLE = 1\n", encoding="utf-8")
            changed.write_text("CHANGED = 1\n", encoding="utf-8")
            git_commit_all(root, "initial")

            changed.write_text("CHANGED = 2\n", encoding="utf-8")
            (root / "new.py").write_text("NEW = 1\n", encoding="utf-8")

            captured = {}

            def fake_get_repo_map(self, chat_files=None, other_files=None, **kwargs):
                captured["other_files"] = other_files
                return None, FileReport({}, 0, 0, len(other_files or []))

            with mock.patch.object(repomap.RepoMap, "get_repo_map", new=fake_get_repo_map):
                with mock.patch.object(sys, "argv", ["repomap.py", "--root", str(root), "--changed"]):
                    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                        repomap.main()

            self.assertEqual(
                sorted(Path(path).name for path in captured["other_files"]),
                ["changed.py", "new.py"],
            )

    def test_cli_changed_mode_respects_base_ref(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir).resolve()
            init_git_repo(root)
            stable = root / "stable.py"
            changed = root / "changed.py"
            stable.write_text("STABLE = 1\n", encoding="utf-8")
            changed.write_text("CHANGED = 1\n", encoding="utf-8")
            git_commit_all(root, "initial")

            changed.write_text("CHANGED = 2\n", encoding="utf-8")
            git_commit_all(root, "update changed file")

            captured = {}

            def fake_get_repo_map(self, chat_files=None, other_files=None, **kwargs):
                captured["other_files"] = other_files
                return None, FileReport({}, 0, 0, len(other_files or []))

            with mock.patch.object(repomap.RepoMap, "get_repo_map", new=fake_get_repo_map):
                with mock.patch.object(sys, "argv", ["repomap.py", "--root", str(root), "--changed", "--base-ref", "HEAD~1"]):
                    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                        repomap.main()

            self.assertEqual([Path(path).name for path in captured["other_files"]], ["changed.py"])

    def test_cli_changed_neighbors_passes_full_scope_and_changed_focus(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir).resolve()
            init_git_repo(root)
            changed = root / "changed.py"
            neighbor = root / "neighbor.py"
            changed.write_text("CHANGED = 1\n", encoding="utf-8")
            neighbor.write_text("NEIGHBOR = 1\n", encoding="utf-8")
            git_commit_all(root, "initial")
            changed.write_text("CHANGED = 2\n", encoding="utf-8")

            captured = {}

            def fake_get_repo_map(self, chat_files=None, other_files=None, **kwargs):
                captured["other_files"] = other_files
                captured["changed_fnames"] = kwargs.get("changed_fnames")
                captured["changed_neighbor_depth"] = kwargs.get("changed_neighbor_depth")
                return None, FileReport({}, 0, 0, len(other_files or []))

            with mock.patch.object(repomap.RepoMap, "get_repo_map", new=fake_get_repo_map):
                with mock.patch.object(
                    sys,
                    "argv",
                    ["repomap.py", "--root", str(root), "--changed-neighbors", "1"],
                ):
                    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                        repomap.main()

            self.assertEqual(
                sorted(Path(path).name for path in captured["other_files"]),
                ["changed.py", "neighbor.py"],
            )
            self.assertEqual({Path(path).name for path in captured["changed_fnames"]}, {"changed.py"})
            self.assertEqual(captured["changed_neighbor_depth"], 1)

    def test_cli_can_emit_json_output_with_ranked_file_reasons(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir).resolve()
            source_file = root / "app.py"
            source_file.write_text("def foo():\n    return 1\n", encoding="utf-8")

            captured = {}
            fake_report = FileReport(
                excluded={},
                definition_matches=1,
                reference_matches=0,
                total_files_considered=1,
                query="auth login",
                query_terms=["auth", "login"],
                ranked_files=[
                    RankedFile(
                        path="app.py",
                        rank=1.0,
                        base_rank=1.0,
                        included_in_map=True,
                        definitions=1,
                        matched_query_terms=["auth"],
                        matched_query_symbol_terms=["auth"],
                        sample_symbols=["foo"],
                        reasons=[RankingReason("definitions", "Defines 1 symbol.")],
                    )
                ],
                selected_files=["app.py"],
                map_tokens=12,
            )

            def fake_get_repo_map(self, chat_files=None, other_files=None, **kwargs):
                captured["query"] = kwargs.get("query")
                return "app.py:\n(Rank value: 1.0000)\n\n1: def foo():", fake_report

            with mock.patch.object(repomap.RepoMap, "get_repo_map", new=fake_get_repo_map):
                with mock.patch.object(
                    sys,
                    "argv",
                    ["repomap.py", "--root", str(root), "--output-format", "json", "--query", "auth login", "app.py"],
                ):
                    stdout = io.StringIO()
                    stderr = io.StringIO()
                    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                        repomap.main()

            payload = json.loads(stdout.getvalue())
            self.assertEqual(captured["query"], "auth login")
            self.assertEqual(payload["report"]["query_terms"], ["auth", "login"])
            self.assertEqual(payload["report"]["selected_files"], ["app.py"])
            self.assertEqual(payload["report"]["ranked_files"][0]["path"], "app.py")
            self.assertEqual(payload["report"]["ranked_files"][0]["reasons"][0]["code"], "definitions")
            self.assertEqual(stderr.getvalue(), "")

    def test_cli_trace_mode_can_emit_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir).resolve()
            app_file = root / "app.py"
            service_file = root / "service.py"
            app_file.write_text("value = 1\n", encoding="utf-8")
            service_file.write_text("value = 2\n", encoding="utf-8")

            captured = {}

            def fake_trace_file_path(self, start_file, end_file, files=None, max_hops=6):
                captured["start_file"] = start_file
                captured["end_file"] = end_file
                captured["files"] = files
                captured["max_hops"] = max_hops
                return repomap_class.ConnectionReport(
                    start_file="app.py",
                    end_file="service.py",
                    path=["app.py", "service.py"],
                    steps=[repomap_class.ConnectionStep("app.py", "service.py", "references", ["Service"])],
                )

            with mock.patch.object(repomap.RepoMap, "trace_file_path", new=fake_trace_file_path):
                with mock.patch.object(
                    sys,
                    "argv",
                    [
                        "repomap.py",
                        "--root",
                        str(root),
                        "--trace-from",
                        "app.py",
                        "--trace-to",
                        "service.py",
                        "--output-format",
                        "json",
                    ],
                ):
                    stdout = io.StringIO()
                    stderr = io.StringIO()
                    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                        repomap.main()

            payload = json.loads(stdout.getvalue())
            self.assertEqual(Path(captured["start_file"]).name, "app.py")
            self.assertEqual(Path(captured["end_file"]).name, "service.py")
            self.assertEqual(payload["path"], ["app.py", "service.py"])
            self.assertEqual(payload["steps"][0]["relation"], "references")
            self.assertEqual(stderr.getvalue(), "")

    def test_cli_impact_mode_can_emit_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir).resolve()
            app_file = root / "app.py"
            service_file = root / "service.py"
            app_file.write_text("value = 1\n", encoding="utf-8")
            service_file.write_text("value = 2\n", encoding="utf-8")

            captured = {}

            def fake_analyze_file_impact(self, seed_files, files=None, max_depth=2, max_results=10):
                captured["seed_files"] = seed_files
                captured["files"] = files
                captured["max_depth"] = max_depth
                captured["max_results"] = max_results
                return repomap_class.ImpactReport(
                    seed_files=["app.py"],
                    max_depth=max_depth,
                    max_results=max_results,
                    impacted_files=[
                        repomap_class.ImpactTarget(
                            path="service.py",
                            seed_file="app.py",
                            distance=1,
                            path_from_seed=["app.py", "service.py"],
                            steps=[repomap_class.ConnectionStep("app.py", "service.py", "references", ["Service"])],
                            reasons=[RankingReason("impact_path", "Reachable from app.py in 1 hop.")],
                        )
                    ],
                    diagnostics=["Found 1 impacted file."],
                )

            with mock.patch.object(repomap.RepoMap, "analyze_file_impact", new=fake_analyze_file_impact):
                with mock.patch.object(
                    sys,
                    "argv",
                    [
                        "repomap.py",
                        "--root",
                        str(root),
                        "--impact-from",
                        "app.py",
                        "--impact-max-depth",
                        "3",
                        "--impact-max-results",
                        "5",
                        "--output-format",
                        "json",
                    ],
                ):
                    stdout = io.StringIO()
                    stderr = io.StringIO()
                    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                        repomap.main()

            payload = json.loads(stdout.getvalue())
            self.assertEqual([Path(path).name for path in captured["seed_files"]], ["app.py"])
            self.assertEqual(captured["max_depth"], 3)
            self.assertEqual(captured["max_results"], 5)
            self.assertEqual(payload["seed_files"], ["app.py"])
            self.assertEqual(payload["impacted_files"][0]["path"], "service.py")
            self.assertEqual(payload["impacted_files"][0]["steps"][0]["relation"], "references")
            self.assertEqual(stderr.getvalue(), "")

    def test_cli_impact_changed_uses_git_seeds_without_narrowing_scope(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir).resolve()
            init_git_repo(root)
            changed_file = root / "changed.py"
            neighbor_file = root / "neighbor.py"
            changed_file.write_text("CHANGED = 1\n", encoding="utf-8")
            neighbor_file.write_text("NEIGHBOR = 1\n", encoding="utf-8")
            git_commit_all(root, "initial")
            changed_file.write_text("CHANGED = 2\n", encoding="utf-8")

            captured = {}

            def fake_analyze_file_impact(self, seed_files, files=None, max_depth=2, max_results=10):
                captured["seed_files"] = seed_files
                captured["files"] = files
                return repomap_class.ImpactReport(
                    seed_files=["changed.py"],
                    max_depth=max_depth,
                    max_results=max_results,
                    impacted_files=[],
                    diagnostics=["No impacted files."],
                )

            with mock.patch.object(repomap.RepoMap, "analyze_file_impact", new=fake_analyze_file_impact):
                with mock.patch.object(
                    sys,
                    "argv",
                    [
                        "repomap.py",
                        "--root",
                        str(root),
                        "--impact-changed",
                        "--output-format",
                        "json",
                    ],
                ):
                    stdout = io.StringIO()
                    stderr = io.StringIO()
                    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                        repomap.main()

            payload = json.loads(stdout.getvalue())
            self.assertEqual([Path(path).name for path in captured["seed_files"]], ["changed.py"])
            self.assertEqual(sorted(Path(path).name for path in captured["files"]), ["changed.py", "neighbor.py"])
            self.assertEqual(payload["seed_files"], ["changed.py"])
            self.assertIn("working tree", payload["diagnostics"][0].lower())
            self.assertEqual(stderr.getvalue(), "")


class RepoMapServerTests(unittest.TestCase):
    def tearDown(self):
        repomap_server._REPO_MAP_CACHE.clear()

    def test_repo_map_passes_query_to_repo_mapper(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir).resolve()
            source_file = root / "app.py"
            source_file.write_text("def foo():\n    return 1\n", encoding="utf-8")

            captured = {}

            def fake_get_repo_map(self, **kwargs):
                captured["query"] = kwargs.get("query")
                return "app.py:\n(Rank value: 1.0000)\n\n1: def foo():", FileReport(
                    excluded={},
                    definition_matches=1,
                    reference_matches=0,
                    total_files_considered=1,
                    query=kwargs.get("query"),
                    query_terms=["auth"],
                )

            with mock.patch.object(repomap_server, "_check_project_root", return_value=None):
                with mock.patch.object(repomap_server, "find_src_files", return_value=[str(source_file)]):
                    with mock.patch.object(RepoMap, "get_repo_map", new=fake_get_repo_map):
                        result = asyncio.run(repomap_server.repo_map(str(root), query="auth"))

            self.assertEqual(captured["query"], "auth")
            self.assertEqual(result["report"]["query"], "auth")
            self.assertEqual(result["report"]["query_terms"], ["auth"])

    def test_repo_map_passes_changed_neighbors_to_repo_mapper(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir).resolve()
            changed_file = root / "changed.py"
            neighbor_file = root / "neighbor.py"
            changed_file.write_text("value = 1\n", encoding="utf-8")
            neighbor_file.write_text("value = 2\n", encoding="utf-8")

            captured = {}

            def fake_get_repo_map(self, **kwargs):
                captured["changed_fnames"] = kwargs.get("changed_fnames")
                captured["changed_neighbor_depth"] = kwargs.get("changed_neighbor_depth")
                return "changed.py:\n(Rank value: 1.0000)\n\n1: value = 1", FileReport(
                    excluded={},
                    definition_matches=0,
                    reference_matches=0,
                    total_files_considered=2,
                    changed_files=["changed.py"],
                    changed_neighbor_depth=1,
                )

            with mock.patch.object(repomap_server, "_check_project_root", return_value=None):
                with mock.patch.object(repomap_server, "find_src_files", return_value=[str(changed_file), str(neighbor_file)]):
                    with mock.patch.object(repomap_server, "get_changed_files", return_value=git_support.GitFileSelectionResult(files=[str(changed_file)])):
                        with mock.patch.object(RepoMap, "get_repo_map", new=fake_get_repo_map):
                            result = asyncio.run(repomap_server.repo_map(str(root), changed_neighbors=1))

            self.assertEqual({Path(path).name for path in captured["changed_fnames"]}, {"changed.py"})
            self.assertEqual(captured["changed_neighbor_depth"], 1)
            self.assertEqual(result["report"]["changed_files"], ["changed.py"])
            self.assertEqual(result["report"]["changed_neighbor_depth"], 1)

    def test_trace_file_path_tool_returns_serialized_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir).resolve()
            app_file = root / "app.py"
            service_file = root / "service.py"
            app_file.write_text("value = 1\n", encoding="utf-8")
            service_file.write_text("value = 2\n", encoding="utf-8")

            def fake_trace_file_path(self, start_file, end_file, files=None, max_hops=6):
                return repomap_class.ConnectionReport(
                    start_file="app.py",
                    end_file="service.py",
                    path=["app.py", "service.py"],
                    steps=[repomap_class.ConnectionStep("app.py", "service.py", "references", ["Service"])],
                    diagnostics=["Found a 1-hop path."],
                )

            with mock.patch.object(repomap_server, "_check_project_root", return_value=None):
                with mock.patch.object(repomap_server, "find_src_files", return_value=[str(app_file), str(service_file)]):
                    with mock.patch.object(RepoMap, "trace_file_path", new=fake_trace_file_path):
                        result = asyncio.run(
                            repomap_server.trace_file_path(
                                str(root),
                                "app.py",
                                "service.py",
                                max_hops=4,
                            )
                        )

            self.assertEqual(result["path"], ["app.py", "service.py"])
            self.assertEqual(result["steps"][0]["relation"], "references")
            self.assertEqual(result["diagnostics"], ["Found a 1-hop path."])

    def test_analyze_file_impact_tool_returns_serialized_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir).resolve()
            app_file = root / "app.py"
            service_file = root / "service.py"
            app_file.write_text("value = 1\n", encoding="utf-8")
            service_file.write_text("value = 2\n", encoding="utf-8")

            def fake_analyze_file_impact(self, seed_files, files=None, max_depth=2, max_results=10):
                return repomap_class.ImpactReport(
                    seed_files=["app.py"],
                    max_depth=max_depth,
                    max_results=max_results,
                    impacted_files=[
                        repomap_class.ImpactTarget(
                            path="service.py",
                            seed_file="app.py",
                            distance=1,
                            path_from_seed=["app.py", "service.py"],
                            steps=[repomap_class.ConnectionStep("app.py", "service.py", "references", ["Service"])],
                            reasons=[RankingReason("impact_path", "Reachable from app.py in 1 hop.")],
                        )
                    ],
                    diagnostics=["Found 1 impacted file."],
                )

            with mock.patch.object(repomap_server, "_check_project_root", return_value=None):
                with mock.patch.object(repomap_server, "find_src_files", return_value=[str(app_file), str(service_file)]):
                    with mock.patch.object(RepoMap, "analyze_file_impact", new=fake_analyze_file_impact):
                        result = asyncio.run(
                            repomap_server.analyze_file_impact(
                                str(root),
                                ["app.py"],
                                max_depth=3,
                                max_results=5,
                            )
                        )

            self.assertEqual(result["seed_files"], ["app.py"])
            self.assertEqual(result["impacted_files"][0]["path"], "service.py")
            self.assertEqual(result["impacted_files"][0]["steps"][0]["relation"], "references")
            self.assertEqual(result["diagnostics"], ["Found 1 impacted file."])

    def test_analyze_file_impact_tool_can_use_git_changed_seeds(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir).resolve()
            changed_file = root / "changed.py"
            neighbor_file = root / "neighbor.py"
            changed_file.write_text("value = 1\n", encoding="utf-8")
            neighbor_file.write_text("value = 2\n", encoding="utf-8")

            captured = {}

            def fake_analyze_file_impact(self, seed_files, files=None, max_depth=2, max_results=10):
                captured["seed_files"] = seed_files
                captured["files"] = files
                return repomap_class.ImpactReport(
                    seed_files=["changed.py"],
                    max_depth=max_depth,
                    max_results=max_results,
                    impacted_files=[],
                    diagnostics=["Found changed impact context."],
                )

            with mock.patch.object(repomap_server, "_check_project_root", return_value=None):
                with mock.patch.object(repomap_server, "find_src_files", return_value=[str(changed_file), str(neighbor_file)]):
                    with mock.patch.object(repomap_server, "get_changed_files", return_value=git_support.GitFileSelectionResult(files=[str(changed_file)], diagnostics=["Collected 1 changed file."])):
                        with mock.patch.object(RepoMap, "analyze_file_impact", new=fake_analyze_file_impact):
                            result = asyncio.run(
                                repomap_server.analyze_file_impact(
                                    str(root),
                                    changed_only=True,
                                    max_depth=3,
                                    max_results=5,
                                )
                            )

            self.assertEqual([Path(path).name for path in captured["seed_files"]], ["changed.py"])
            self.assertEqual(sorted(Path(path).name for path in captured["files"]), ["changed.py", "neighbor.py"])
            self.assertEqual(result["seed_files"], ["changed.py"])
            self.assertEqual(result["diagnostics"][0], "Collected 1 changed file.")


class GitSupportTests(unittest.TestCase):
    def test_get_changed_files_collects_worktree_changes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir).resolve()
            init_git_repo(root)
            stable = root / "stable.py"
            changed = root / "changed.py"
            stable.write_text("STABLE = 1\n", encoding="utf-8")
            changed.write_text("CHANGED = 1\n", encoding="utf-8")
            git_commit_all(root, "initial")

            changed.write_text("CHANGED = 2\n", encoding="utf-8")
            (root / "new.py").write_text("NEW = 1\n", encoding="utf-8")

            result = git_support.get_changed_files(str(root))

            self.assertIsNone(result.error)
            self.assertIn("working tree", result.diagnostics[0].lower())
            self.assertEqual(sorted(Path(path).name for path in result.files), ["changed.py", "new.py"])

    def test_get_changed_files_supports_base_ref(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir).resolve()
            init_git_repo(root)
            stable = root / "stable.py"
            changed = root / "changed.py"
            stable.write_text("STABLE = 1\n", encoding="utf-8")
            changed.write_text("CHANGED = 1\n", encoding="utf-8")
            git_commit_all(root, "initial")

            changed.write_text("CHANGED = 2\n", encoding="utf-8")
            git_commit_all(root, "update changed file")

            result = git_support.get_changed_files(str(root), "HEAD~1")

            self.assertIsNone(result.error)
            self.assertIn("HEAD~1", result.diagnostics[0])
            self.assertEqual([Path(path).name for path in result.files], ["changed.py"])


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
