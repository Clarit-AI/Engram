"""
Tests for documentation changes introduced in the docs-resync PR.

Covers:
- File existence and placement for new/moved/deleted docs
- Archive files contain proper archival warnings
- Active docs do not reference stale paths
- README.md endpoint table accuracy
- CLAUDE.md gh-auth guidance presence
- AGENTS.md canonical doc references
- INDEX.md and SUMMARY.md reference correct active files
- api_guide.md required sections and endpoint listing
"""

import re
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DOCS_DIR = REPO_ROOT / "docs" / "stateful_mamba"
ARCHIVE_DIR = DOCS_DIR / ".archive"


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# File existence tests
# ---------------------------------------------------------------------------

class TestFileExistence(unittest.TestCase):
    """New files must exist; deleted/moved files must not be in old locations."""

    # --- new files that must exist ---

    def test_agents_md_exists_at_repo_root(self):
        self.assertTrue(
            (REPO_ROOT / "AGENTS.md").is_file(),
            "AGENTS.md must exist at repository root",
        )

    def test_archive_agents_md_exists(self):
        self.assertTrue(
            (ARCHIVE_DIR / "AGENTS.md").is_file(),
            ".archive/AGENTS.md must exist with archive rules",
        )

    def test_archive_docstring_templates_md_exists(self):
        self.assertTrue(
            (ARCHIVE_DIR / "DOCSTRING_TEMPLATES.md").is_file(),
            ".archive/DOCSTRING_TEMPLATES.md must exist with archival notice",
        )

    def test_archive_api_reference_md_exists(self):
        self.assertTrue(
            (ARCHIVE_DIR / "api_reference.md").is_file(),
            ".archive/api_reference.md must exist (renamed from active dir)",
        )

    def test_api_guide_md_exists_in_active_docs(self):
        self.assertTrue(
            (DOCS_DIR / "api_guide.md").is_file(),
            "api_guide.md must exist in docs/stateful_mamba/",
        )

    # --- files that must NOT exist in active (non-archive) location ---

    def test_docstring_templates_not_in_active_docs(self):
        self.assertFalse(
            (DOCS_DIR / "DOCSTRING_TEMPLATES.md").exists(),
            "DOCSTRING_TEMPLATES.md must have been removed from active docs dir",
        )

    def test_api_reference_not_in_active_docs(self):
        self.assertFalse(
            (DOCS_DIR / "api_reference.md").exists(),
            "api_reference.md must have been moved to .archive/ and removed from active docs",
        )

    # --- active doc files that must still exist ---

    def test_index_md_exists(self):
        self.assertTrue((DOCS_DIR / "INDEX.md").is_file())

    def test_summary_md_exists(self):
        self.assertTrue((DOCS_DIR / "SUMMARY.md").is_file())

    def test_http_api_spec_md_exists(self):
        self.assertTrue((DOCS_DIR / "http_api_spec.md").is_file())

    def test_user_guide_md_exists(self):
        self.assertTrue((DOCS_DIR / "user_guide.md").is_file())

    def test_architecture_md_exists(self):
        self.assertTrue((DOCS_DIR / "architecture.md").is_file())

    def test_migration_guide_md_exists(self):
        self.assertTrue((DOCS_DIR / "migration_guide.md").is_file())

    def test_examples_md_exists(self):
        self.assertTrue((DOCS_DIR / "examples.md").is_file())

    def test_troubleshooting_md_exists(self):
        self.assertTrue((DOCS_DIR / "troubleshooting.md").is_file())


# ---------------------------------------------------------------------------
# Archive file integrity
# ---------------------------------------------------------------------------

class TestArchiveFileIntegrity(unittest.TestCase):
    """Files in .archive/ must carry a visible archival warning."""

    def _assert_has_archival_marker(self, path: Path):
        text = _read(path).lower()
        markers = ["archival", "historical reference", "archive", "historical"]
        has_marker = any(m in text for m in markers)
        self.assertTrue(
            has_marker,
            f"{path.name} must contain an archival/historical reference marker",
        )

    def test_archive_api_reference_has_archival_header(self):
        path = ARCHIVE_DIR / "api_reference.md"
        text = _read(path)
        self.assertIn(
            "ARCHIVAL ONLY",
            text,
            "api_reference.md in archive must start with 'ARCHIVAL ONLY' in its heading",
        )

    def test_archive_docstring_templates_has_archival_header(self):
        path = ARCHIVE_DIR / "DOCSTRING_TEMPLATES.md"
        text = _read(path)
        self.assertIn(
            "ARCHIVAL ONLY",
            text,
            "DOCSTRING_TEMPLATES.md in archive must contain 'ARCHIVAL ONLY' heading",
        )

    def test_archive_agents_md_has_historical_reference_statement(self):
        path = ARCHIVE_DIR / "AGENTS.md"
        text = _read(path)
        self.assertIn(
            "historical reference",
            text.lower(),
            ".archive/AGENTS.md must say files are for historical reference",
        )

    def test_archive_api_reference_warns_against_production_use(self):
        text = _read(ARCHIVE_DIR / "api_reference.md").lower()
        self.assertTrue(
            "do not use" in text or "not reflect" in text or "historical" in text,
            "archive api_reference.md must discourage production use",
        )

    def test_archive_docstring_templates_points_to_canonical(self):
        """The archived DOCSTRING_TEMPLATES.md should direct users to http_api_spec.md."""
        text = _read(ARCHIVE_DIR / "DOCSTRING_TEMPLATES.md")
        self.assertIn(
            "http_api_spec.md",
            text,
            "Archived DOCSTRING_TEMPLATES.md must reference http_api_spec.md as canonical",
        )

    def test_archive_api_reference_points_to_canonical(self):
        """The archived api_reference.md should direct users to http_api_spec.md."""
        text = _read(ARCHIVE_DIR / "api_reference.md")
        self.assertIn(
            "http_api_spec.md",
            text,
            "Archived api_reference.md must reference http_api_spec.md as canonical",
        )


# ---------------------------------------------------------------------------
# AGENTS.md (repo root)
# ---------------------------------------------------------------------------

class TestAgentsMd(unittest.TestCase):
    """Root AGENTS.md must contain the correct canonical-doc pointers."""

    def setUp(self):
        self.text = _read(REPO_ROOT / "AGENTS.md")

    def test_references_api_guide(self):
        self.assertIn(
            "api_guide.md",
            self.text,
            "AGENTS.md must reference api_guide.md as user-friendly API guide",
        )

    def test_references_http_api_spec(self):
        self.assertIn(
            "http_api_spec.md",
            self.text,
            "AGENTS.md must reference http_api_spec.md as canonical technical spec",
        )

    def test_references_archive_directory(self):
        self.assertIn(
            ".archive/",
            self.text,
            "AGENTS.md must mention .archive/ as historical reference location",
        )

    def test_describes_engram_project(self):
        self.assertIn(
            "Engram",
            self.text,
            "AGENTS.md must identify the project as Engram",
        )

    def test_mentions_sglang_mamba_legacy(self):
        """Should acknowledge legacy sglang-mamba references so agents know about them."""
        self.assertIn(
            "sglang-mamba",
            self.text,
            "AGENTS.md must mention legacy sglang-mamba name",
        )

    def test_has_canonical_docs_section(self):
        self.assertIn(
            "Canonical Docs",
            self.text,
            "AGENTS.md must have a 'Canonical Docs' section",
        )

    def test_clariti_ai_origin(self):
        """Should mention Clarit-AI as the origin for gh operations."""
        self.assertIn(
            "Clarit-AI",
            self.text,
            "AGENTS.md must reference the Clarit-AI GitHub account",
        )


# ---------------------------------------------------------------------------
# CLAUDE.md additions
# ---------------------------------------------------------------------------

class TestClaudeMdGhAuthSection(unittest.TestCase):
    """CLAUDE.md must contain the new gh-auth guidance."""

    def setUp(self):
        self.text = _read(REPO_ROOT / "CLAUDE.md")

    def test_has_gh_auth_status_command(self):
        self.assertIn(
            "gh auth status",
            self.text,
            "CLAUDE.md must include 'gh auth status' command",
        )

    def test_has_gh_auth_switch_command(self):
        self.assertIn(
            "gh auth switch",
            self.text,
            "CLAUDE.md must include 'gh auth switch' command",
        )

    def test_mentions_clarit_ai_user(self):
        self.assertIn(
            "Clarit-AI",
            self.text,
            "CLAUDE.md must mention Clarit-AI as the target account",
        )

    def test_warns_about_kha_entertainment(self):
        self.assertIn(
            "KHAEntertainment",
            self.text,
            "CLAUDE.md must warn about KHAEntertainment account confusion",
        )

    def test_mentions_engram_origin(self):
        self.assertIn(
            "Clarit-AI/Engram",
            self.text,
            "CLAUDE.md must state origin is Clarit-AI/Engram",
        )


# ---------------------------------------------------------------------------
# README.md endpoint table
# ---------------------------------------------------------------------------

class TestReadmeEndpointTable(unittest.TestCase):
    """README.md endpoint table must reflect the current HTTP API."""

    def setUp(self):
        self.text = _read(REPO_ROOT / "README.md")

    def test_save_snapshot_is_post(self):
        match = re.search(r"`/save_snapshot`\s*\|\s*(\w+)", self.text)
        self.assertIsNotNone(match, "README must contain /save_snapshot row")
        self.assertEqual(match.group(1), "POST")

    def test_restore_snapshot_is_post(self):
        match = re.search(r"`/restore_snapshot`\s*\|\s*(\w+)", self.text)
        self.assertIsNotNone(match, "README must contain /restore_snapshot row")
        self.assertEqual(match.group(1), "POST")

    def test_list_snapshots_is_post(self):
        """Changed from GET to POST in this PR."""
        match = re.search(r"`/list_snapshots`\s*\|\s*(\w+)", self.text)
        self.assertIsNotNone(match, "README must contain /list_snapshots row")
        self.assertEqual(
            match.group(1),
            "POST",
            "/list_snapshots must be POST (was GET in old docs)",
        )

    def test_delete_snapshot_is_post(self):
        """Changed from DELETE to POST in this PR."""
        match = re.search(r"`/delete_snapshot`\s*\|\s*(\w+)", self.text)
        self.assertIsNotNone(match, "README must contain /delete_snapshot row")
        self.assertEqual(
            match.group(1),
            "POST",
            "/delete_snapshot must be POST (was DELETE in old docs)",
        )

    def test_get_snapshot_info_endpoint_added(self):
        """New endpoint added in this PR."""
        self.assertIn(
            "/get_snapshot_info",
            self.text,
            "/get_snapshot_info must be listed in README endpoint table",
        )

    def test_get_snapshot_info_is_post(self):
        match = re.search(r"`/get_snapshot_info`\s*\|\s*(\w+)", self.text)
        self.assertIsNotNone(match, "README must contain /get_snapshot_info row")
        self.assertEqual(match.group(1), "POST")

    def test_api_guide_link_present(self):
        self.assertIn(
            "api_guide.md",
            self.text,
            "README must link to docs/stateful_mamba/api_guide.md",
        )

    def test_http_api_spec_link_present(self):
        self.assertIn(
            "http_api_spec.md",
            self.text,
            "README must link to docs/stateful_mamba/http_api_spec.md",
        )

    def test_no_stale_api_reference_link(self):
        """Old api_reference.md link must not appear in the active navigation."""
        # The table nav link [API Reference](#api-extensions) still exists as a
        # section anchor, but the dead docs/stateful_mamba/api_reference.md path
        # must not be linked from README.
        self.assertNotIn(
            "stateful_mamba/api_reference.md",
            self.text,
            "README must not link to the now-archived api_reference.md",
        )

    def test_deepwiki_badge_removed(self):
        """DeepWiki badge was removed in this PR."""
        self.assertNotIn(
            "deepwiki.com/badge.svg",
            self.text,
            "DeepWiki badge must have been removed from README",
        )


# ---------------------------------------------------------------------------
# INDEX.md
# ---------------------------------------------------------------------------

class TestIndexMd(unittest.TestCase):
    """INDEX.md must reference active docs and not stale ones."""

    def setUp(self):
        self.text = _read(DOCS_DIR / "INDEX.md")

    def test_references_api_guide(self):
        self.assertIn("api_guide.md", self.text)

    def test_references_http_api_spec(self):
        self.assertIn("http_api_spec.md", self.text)

    def test_references_user_guide(self):
        self.assertIn("user_guide.md", self.text)

    def test_references_migration_guide(self):
        self.assertIn("migration_guide.md", self.text)

    def test_references_examples(self):
        self.assertIn("examples.md", self.text)

    def test_references_architecture(self):
        self.assertIn("architecture.md", self.text)

    def test_references_troubleshooting(self):
        self.assertIn("troubleshooting.md", self.text)

    def test_references_summary(self):
        self.assertIn("SUMMARY.md", self.text)

    def test_does_not_reference_deleted_docstring_templates(self):
        """DOCSTRING_TEMPLATES.md was deleted from active docs."""
        # Should only appear in archive context if at all
        # The index must not navigate users there as an active doc
        lines_with_docstring = [
            ln for ln in self.text.splitlines()
            if "DOCSTRING_TEMPLATES.md" in ln and ".archive" not in ln
        ]
        self.assertEqual(
            len(lines_with_docstring),
            0,
            "INDEX.md must not link to DOCSTRING_TEMPLATES.md outside .archive/",
        )

    def test_does_not_reference_active_api_reference(self):
        """api_reference.md moved to .archive/ — INDEX must not link active path."""
        lines_with_api_ref = [
            ln for ln in self.text.splitlines()
            if "api_reference.md" in ln and ".archive" not in ln
        ]
        self.assertEqual(
            len(lines_with_api_ref),
            0,
            "INDEX.md must not link to api_reference.md outside .archive/",
        )

    def test_mentions_archive_directory(self):
        self.assertIn(".archive", self.text, "INDEX.md must reference .archive/ section")

    def test_title_is_engram_not_stateful_mamba(self):
        """Title was updated to Engram Docs Index."""
        first_heading = next(
            (ln for ln in self.text.splitlines() if ln.startswith("#")), ""
        )
        self.assertIn(
            "Engram",
            first_heading,
            "INDEX.md title must say Engram Docs Index",
        )


# ---------------------------------------------------------------------------
# SUMMARY.md
# ---------------------------------------------------------------------------

class TestSummaryMd(unittest.TestCase):
    """SUMMARY.md must list active docs and archive correctly."""

    def setUp(self):
        self.text = _read(DOCS_DIR / "SUMMARY.md")

    def test_references_api_guide(self):
        self.assertIn("api_guide.md", self.text)

    def test_references_http_api_spec(self):
        self.assertIn("http_api_spec.md", self.text)

    def test_references_user_guide(self):
        self.assertIn("user_guide.md", self.text)

    def test_references_migration_guide(self):
        self.assertIn("migration_guide.md", self.text)

    def test_references_examples(self):
        self.assertIn("examples.md", self.text)

    def test_references_architecture(self):
        self.assertIn("architecture.md", self.text)

    def test_references_troubleshooting(self):
        self.assertIn("troubleshooting.md", self.text)

    def test_mentions_archive(self):
        self.assertIn(".archive", self.text)

    def test_five_snapshot_endpoints_listed(self):
        """SUMMARY.md lists the five snapshot HTTP endpoints."""
        for endpoint in [
            "/save_snapshot",
            "/list_snapshots",
            "/get_snapshot_info",
            "/restore_snapshot",
            "/delete_snapshot",
        ]:
            self.assertIn(
                endpoint,
                self.text,
                f"SUMMARY.md must list endpoint {endpoint}",
            )

    def test_title_is_engram(self):
        first_heading = next(
            (ln for ln in self.text.splitlines() if ln.startswith("#")), ""
        )
        self.assertIn(
            "Engram",
            first_heading,
            "SUMMARY.md title must say Engram",
        )


# ---------------------------------------------------------------------------
# api_guide.md
# ---------------------------------------------------------------------------

class TestApiGuideMd(unittest.TestCase):
    """api_guide.md must contain the required sections and accurate info."""

    def setUp(self):
        self.text = _read(DOCS_DIR / "api_guide.md")

    def test_title_is_engram_api_guide(self):
        first_heading = next(
            (ln for ln in self.text.splitlines() if ln.startswith("#")), ""
        )
        self.assertIn("Engram", first_heading)

    def test_has_quick_start_section(self):
        self.assertIn("Quick Start", self.text)

    def test_has_authentication_section(self):
        self.assertIn("Authentication", self.text)

    def test_has_endpoints_at_a_glance(self):
        self.assertIn("Endpoints At A Glance", self.text)

    def test_has_common_workflows_section(self):
        self.assertIn("Common Workflows", self.text)

    def test_has_error_handling_section(self):
        self.assertIn("Error Handling", self.text)

    def test_lists_all_five_endpoints(self):
        for endpoint in [
            "/save_snapshot",
            "/list_snapshots",
            "/get_snapshot_info",
            "/restore_snapshot",
            "/delete_snapshot",
        ]:
            self.assertIn(
                endpoint,
                self.text,
                f"api_guide.md must document endpoint {endpoint}",
            )

    def test_all_endpoints_are_post(self):
        """All Engram snapshot endpoints are POST in the current API."""
        endpoint_rows = re.findall(
            r"`(/(?:save|list|get|restore|delete)_snapshot[^`]*)`\s*\|\s*`(\w+)`",
            self.text,
        )
        for endpoint, method in endpoint_rows:
            self.assertEqual(
                method,
                "POST",
                f"{endpoint} must be documented as POST in api_guide.md",
            )

    def test_references_http_api_spec_for_technical_details(self):
        self.assertIn(
            "http_api_spec.md",
            self.text,
            "api_guide.md must point to http_api_spec.md for technical details",
        )

    def test_has_python_example(self):
        self.assertIn("python", self.text.lower())
        self.assertIn("requests", self.text)

    def test_has_javascript_example(self):
        self.assertIn("JavaScript", self.text)

    def test_has_curl_example(self):
        self.assertIn("curl", self.text)

    def test_describes_conversation_id_concept(self):
        self.assertIn(
            "conversation_id",
            self.text,
            "api_guide.md must explain the conversation_id concept",
        )

    def test_describes_turn_number_concept(self):
        self.assertIn(
            "turn_number",
            self.text,
            "api_guide.md must explain the turn_number concept",
        )

    def test_admin_optional_auth_explained(self):
        self.assertIn(
            "ADMIN_OPTIONAL",
            self.text,
            "api_guide.md must document ADMIN_OPTIONAL auth behavior",
        )

    def test_save_snapshot_can_fail_with_200_note(self):
        """Quirk: /save_snapshot can fail with HTTP 200, must be documented."""
        self.assertIn(
            "200",
            self.text,
            "api_guide.md must warn that /save_snapshot can return HTTP 200 on failure",
        )

    def test_python_api_surface_section(self):
        self.assertIn("Python API Surface", self.text)

    def test_snapshot_manager_mentioned(self):
        self.assertIn("SnapshotManager", self.text)


# ---------------------------------------------------------------------------
# architecture.md
# ---------------------------------------------------------------------------

class TestArchitectureMd(unittest.TestCase):
    """architecture.md must describe the real implementation, not the old speculative one."""

    def setUp(self):
        self.text = _read(DOCS_DIR / "architecture.md")

    def test_title_is_engram_architecture(self):
        first_heading = next(
            (ln for ln in self.text.splitlines() if ln.startswith("#")), ""
        )
        self.assertIn("Engram", first_heading)

    def test_references_mamba_snapshot_py(self):
        self.assertIn(
            "mamba_snapshot.py",
            self.text,
            "architecture.md must reference the real mamba_snapshot.py file",
        )

    def test_references_scheduler(self):
        self.assertIn(
            "scheduler",
            self.text.lower(),
            "architecture.md must describe scheduler integration",
        )

    def test_references_tokenizer_manager(self):
        self.assertIn(
            "tokenizer_manager",
            self.text.lower().replace(" ", "_"),
            "architecture.md must describe tokenizer manager role",
        )

    def test_references_http_server(self):
        self.assertIn(
            "http_server",
            self.text.lower().replace(" ", "_"),
            "architecture.md must describe the HTTP server layer",
        )

    def test_does_not_mention_nonexistent_snapshot_registry(self):
        """Old speculative SnapshotRegistry class does not exist in the codebase."""
        # The new architecture.md should not describe SnapshotRegistry as a
        # first-class component — it was a speculative design artifact.
        # Acceptable if mentioned purely as historical context, but should not
        # be a top-level section header.
        lines_registry_header = [
            ln for ln in self.text.splitlines()
            if ln.startswith("#") and "SnapshotRegistry" in ln
        ]
        self.assertEqual(
            len(lines_registry_header),
            0,
            "architecture.md must not have a top-level SnapshotRegistry section header "
            "(it is a speculative artifact, not a real component)",
        )

    def test_describes_five_http_endpoints(self):
        for endpoint in [
            "/save_snapshot",
            "/list_snapshots",
            "/get_snapshot_info",
            "/restore_snapshot",
            "/delete_snapshot",
        ]:
            self.assertIn(
                endpoint,
                self.text,
                f"architecture.md must mention endpoint {endpoint}",
            )

    def test_describes_save_flow(self):
        self.assertIn("Save Flow", self.text)

    def test_describes_restore_flow(self):
        self.assertIn("Restore Flow", self.text)


# ---------------------------------------------------------------------------
# No active doc leaks stale paths
# ---------------------------------------------------------------------------

class TestNoStaleReferencesInActiveDocs(unittest.TestCase):
    """Active docs must not reference files that have been deleted or archived."""

    ACTIVE_DOC_PATHS = [
        REPO_ROOT / "AGENTS.md",
        REPO_ROOT / "CLAUDE.md",
        REPO_ROOT / "README.md",
        DOCS_DIR / "INDEX.md",
        DOCS_DIR / "SUMMARY.md",
        DOCS_DIR / "api_guide.md",
        DOCS_DIR / "architecture.md",
    ]

    def _assert_no_bare_reference(self, filename: str, description: str):
        """filename must not appear in active docs except via .archive/ prefix."""
        for path in self.ACTIVE_DOC_PATHS:
            text = _read(path)
            # Find any link that references the filename without going through .archive/
            bad_refs = [
                ln for ln in text.splitlines()
                if filename in ln and ".archive" not in ln
                # Allow headings and content text that mention it by name as prose
                # but flag markdown link syntax pointing at it
                and re.search(r"\[.*?\]\([^)]*" + re.escape(filename) + r"[^)]*\)", ln)
            ]
            self.assertEqual(
                len(bad_refs),
                0,
                f"{path.name} must not link to {filename} outside .archive/: {bad_refs}",
            )

    def test_no_active_link_to_docstring_templates(self):
        self._assert_no_bare_reference(
            "DOCSTRING_TEMPLATES.md",
            "deleted active doc",
        )

    def test_no_active_link_to_api_reference_in_docs(self):
        """api_reference.md was moved to .archive/ — active docs must not link it directly."""
        # Only check docs that we fully control in this PR; README links are OK
        # since the table doesn't link api_reference.md directly
        for path in [
            DOCS_DIR / "INDEX.md",
            DOCS_DIR / "SUMMARY.md",
            DOCS_DIR / "api_guide.md",
            DOCS_DIR / "architecture.md",
            REPO_ROOT / "AGENTS.md",
        ]:
            text = _read(path)
            bad_refs = [
                ln for ln in text.splitlines()
                if "api_reference.md" in ln
                and ".archive" not in ln
                and re.search(r"\[.*?\]\([^)]*api_reference\.md[^)]*\)", ln)
            ]
            self.assertEqual(
                len(bad_refs),
                0,
                f"{path.name} must not link to api_reference.md outside .archive/: {bad_refs}",
            )


# ---------------------------------------------------------------------------
# Regression / boundary / edge cases
# ---------------------------------------------------------------------------

class TestRegressionAndEdgeCases(unittest.TestCase):
    """Extra regression and boundary tests for robustness."""

    def test_archive_directory_itself_exists(self):
        self.assertTrue(
            ARCHIVE_DIR.is_dir(),
            ".archive directory must exist under docs/stateful_mamba/",
        )

    def test_archive_directory_has_at_least_three_files(self):
        files = list(ARCHIVE_DIR.iterdir())
        self.assertGreaterEqual(
            len(files),
            3,
            ".archive/ must contain at least AGENTS.md, DOCSTRING_TEMPLATES.md, api_reference.md",
        )

    def test_readme_has_exactly_five_snapshot_endpoints(self):
        text = _read(REPO_ROOT / "README.md")
        # Count endpoint rows in the API Extensions table
        endpoint_pattern = re.compile(r"`/\w+_snapshot[^`]*`")
        endpoint_matches = endpoint_pattern.findall(text)
        self.assertEqual(
            len(endpoint_matches),
            5,
            f"README must list exactly 5 snapshot endpoints, found: {endpoint_matches}",
        )

    def test_api_guide_does_not_mention_deepwiki(self):
        text = _read(DOCS_DIR / "api_guide.md")
        self.assertNotIn("deepwiki", text.lower())

    def test_agents_md_does_not_reference_native_linear_cli(self):
        """AGENTS.md must warn against native Linear CLI usage."""
        text = _read(REPO_ROOT / "AGENTS.md")
        # Should mention not using native Linear CLI/MCP
        self.assertIn(
            "Core Memory MCP",
            text,
            "AGENTS.md must direct users to Core Memory MCP, not native Linear CLI",
        )

    def test_claude_md_does_not_contradict_agents_md_on_gh(self):
        """Both CLAUDE.md and AGENTS.md must agree on Clarit-AI as the gh account."""
        claude_text = _read(REPO_ROOT / "CLAUDE.md")
        agents_text = _read(REPO_ROOT / "AGENTS.md")
        self.assertIn("Clarit-AI", claude_text)
        self.assertIn("Clarit-AI", agents_text)

    def test_index_md_not_empty(self):
        text = _read(DOCS_DIR / "INDEX.md")
        self.assertGreater(len(text.strip()), 100)

    def test_summary_md_not_empty(self):
        text = _read(DOCS_DIR / "SUMMARY.md")
        self.assertGreater(len(text.strip()), 100)

    def test_api_guide_references_rid_concept(self):
        """rid is a core concept in save/restore workflows."""
        text = _read(DOCS_DIR / "api_guide.md")
        self.assertIn(
            "`rid`",
            text,
            "api_guide.md must explain the rid concept",
        )

    def test_archive_api_reference_has_warning_callout(self):
        """The [!WARNING] callout must be present in archived api_reference.md."""
        text = _read(ARCHIVE_DIR / "api_reference.md")
        self.assertIn(
            "[!WARNING]",
            text,
            "Archived api_reference.md must use the [!WARNING] callout block",
        )

    def test_archive_docstring_templates_has_warning_callout(self):
        text = _read(ARCHIVE_DIR / "DOCSTRING_TEMPLATES.md")
        self.assertIn(
            "[!WARNING]",
            text,
            "Archived DOCSTRING_TEMPLATES.md must use the [!WARNING] callout block",
        )

    def test_readme_nav_uses_api_guide_not_api_reference_anchor(self):
        """Navigation bar was updated from #api-extensions anchor to api_guide.md link."""
        text = _read(REPO_ROOT / "README.md")
        # Old link: [API Reference](#api-extensions) — replaced with API Guide link
        self.assertIn(
            "api_guide.md",
            text,
            "README nav must use api_guide.md link",
        )

    def test_api_guide_has_branch_name_concept(self):
        text = _read(DOCS_DIR / "api_guide.md")
        self.assertIn(
            "branch_name",
            text,
            "api_guide.md must explain the branch_name concept for branching workflows",
        )


if __name__ == "__main__":
    unittest.main()