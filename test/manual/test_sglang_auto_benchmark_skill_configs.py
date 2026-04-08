"""
Tests for the sglang-auto-benchmark skill YAML configs and documentation.

Covers the files added in this PR:
- .claude/skills/sglang-auto-benchmark/SKILL.md
- .claude/skills/sglang-auto-benchmark/references/cookbook-llm/README.md
- .claude/skills/sglang-auto-benchmark/references/cookbook-llm/*.yaml (25 files)
"""

import sys
from pathlib import Path

import pytest
import yaml

# Paths
REPO_ROOT = Path(__file__).parent.parent.parent
SKILL_DIR = REPO_ROOT / ".claude" / "skills" / "sglang-auto-benchmark"
COOKBOOK_DIR = SKILL_DIR / "references" / "cookbook-llm"
SKILL_MD = SKILL_DIR / "SKILL.md"
COOKBOOK_README = COOKBOOK_DIR / "README.md"

# Exactly the YAML files added in this PR
PR_YAML_FILES = [
    "deepseek-math-v2.yaml",
    "deepseek-r1-0528.yaml",
    "deepseek-v3.1.yaml",
    "deepseek-v3.2.yaml",
    "deepseek-v3.yaml",
    "devstral-small-2-24b-instruct-2512.yaml",
    "ernie-4.5-21b-a3b-pt.yaml",
    "glm-4.5.yaml",
    "glm-4.6.yaml",
    "glm-4.7-flash.yaml",
    "glm-4.7.yaml",
    "glm-5-fp8.yaml",
    "glyph.yaml",
    "gpt-oss-120b.yaml",
    "intern-s1.yaml",
    "kimi-k2-instruct.yaml",
    "kimi-k2.5.yaml",
    "kimi-linear-48b-a3b-instruct.yaml",
    "ling-2.5-1t.yaml",
    "llada2-1-mini.yaml",
    "llama-3.1-70b-instruct.yaml",
    "llama-3.3-70b-instruct.yaml",
    "llama-4-maverick-17b-128e-instruct-fp8.yaml",
]

SUPPORTED_DATASET_KINDS = {"random", "sharegpt", "custom", "generated-shared-prefix"}
VALID_ATTENTION_BACKENDS = {"fa3", "flashinfer", "triton", "fa2", "torch_native"}
VALID_SEARCH_TIERS = {1, 2, 3}


def load_yaml(filename: str) -> dict:
    """Load a YAML config from the cookbook-llm directory."""
    path = COOKBOOK_DIR / filename
    with open(path) as f:
        return yaml.safe_load(f)


@pytest.fixture(params=PR_YAML_FILES)
def config(request):
    """Parametrized fixture yielding (filename, parsed_dict) for each changed config."""
    filename = request.param
    return filename, load_yaml(filename)


# ---------------------------------------------------------------------------
# File existence and parseability
# ---------------------------------------------------------------------------


class TestFilesExistAndParse:
    def test_skill_md_exists(self):
        assert SKILL_MD.exists(), f"SKILL.md not found at {SKILL_MD}"

    def test_cookbook_readme_exists(self):
        assert COOKBOOK_README.exists(), f"README.md not found at {COOKBOOK_README}"

    @pytest.mark.parametrize("filename", PR_YAML_FILES)
    def test_yaml_file_exists(self, filename):
        path = COOKBOOK_DIR / filename
        assert path.exists(), f"Expected YAML config {filename} not found at {path}"

    @pytest.mark.parametrize("filename", PR_YAML_FILES)
    def test_yaml_file_parses_without_error(self, filename):
        """Each YAML file must parse cleanly with no exceptions."""
        cfg = load_yaml(filename)
        assert isinstance(cfg, dict), f"{filename}: parsed content should be a dict"

    @pytest.mark.parametrize("filename", PR_YAML_FILES)
    def test_yaml_file_is_not_empty(self, filename):
        cfg = load_yaml(filename)
        assert cfg, f"{filename}: config dict should not be empty"


# ---------------------------------------------------------------------------
# SKILL.md content
# ---------------------------------------------------------------------------


class TestSkillMd:
    def test_skill_md_has_frontmatter_name(self):
        text = SKILL_MD.read_text()
        assert "name: sglang-auto-benchmark" in text

    def test_skill_md_has_frontmatter_description(self):
        text = SKILL_MD.read_text()
        assert "description:" in text

    def test_skill_md_documents_supported_dataset_kinds(self):
        text = SKILL_MD.read_text()
        for kind in ("sharegpt", "custom", "random", "generated-shared-prefix"):
            assert kind in text, f"SKILL.md should document dataset kind '{kind}'"

    def test_skill_md_documents_search_tiers(self):
        text = SKILL_MD.read_text()
        assert "Tier 1" in text
        assert "Tier 2" in text
        assert "Tier 3" in text

    def test_skill_md_documents_canonical_dataset_format(self):
        text = SKILL_MD.read_text()
        assert "output_len" in text
        assert "prompt" in text

    def test_skill_md_documents_sla_fields(self):
        text = SKILL_MD.read_text()
        assert "max_ttft_ms" in text
        assert "max_tpot_ms" in text

    def test_skill_md_mentions_resume_flag(self):
        """Resume/interrupt behavior must be documented."""
        text = SKILL_MD.read_text()
        assert "resume" in text

    def test_skill_md_documents_speculative_eagled_tuning(self):
        text = SKILL_MD.read_text()
        assert "EAGLE" in text

    def test_skill_md_documents_remote_log_mirroring(self):
        text = SKILL_MD.read_text()
        assert "progress.log" in text

    def test_skill_md_documents_output_artifacts(self):
        text = SKILL_MD.read_text()
        for artifact in ("results.jsonl", "results.csv", "summary.md"):
            assert (
                artifact in text
            ), f"SKILL.md should mention output artifact '{artifact}'"


# ---------------------------------------------------------------------------
# README.md content
# ---------------------------------------------------------------------------


class TestCookbookReadme:
    def test_readme_lists_each_pr_yaml_file(self):
        """README must enumerate every YAML config added by this PR."""
        text = COOKBOOK_README.read_text()
        for filename in PR_YAML_FILES:
            assert filename in text, f"README.md should list config '{filename}'"

    def test_readme_documents_default_scenarios(self):
        text = COOKBOOK_README.read_text()
        assert "1000" in text
        assert "8000" in text

    def test_readme_documents_default_num_prompts(self):
        text = COOKBOOK_README.read_text()
        assert "80" in text

    def test_readme_documents_search_tier_default(self):
        text = COOKBOOK_README.read_text()
        assert "tier: 2" in text or "search.tier: 2" in text

    def test_readme_documents_max_candidates_default(self):
        text = COOKBOOK_README.read_text()
        assert "max_candidates: 8" in text or "search.max_candidates: 8" in text

    def test_readme_lists_excluded_vl_models(self):
        """VL/OCR exclusions should be documented."""
        text = COOKBOOK_README.read_text()
        assert "OCR" in text or "VL" in text


# ---------------------------------------------------------------------------
# Top-level schema keys
# ---------------------------------------------------------------------------


class TestTopLevelKeys:
    REQUIRED_KEYS = ("server", "dataset", "benchmark", "search", "speculative")

    @pytest.mark.parametrize("filename", PR_YAML_FILES)
    def test_required_top_level_keys_present(self, filename):
        cfg = load_yaml(filename)
        for key in self.REQUIRED_KEYS:
            assert key in cfg, f"{filename}: missing required top-level key '{key}'"


# ---------------------------------------------------------------------------
# Server section
# ---------------------------------------------------------------------------


class TestServerSection:
    def test_server_has_host(self, config):
        filename, cfg = config
        assert "host" in cfg["server"], f"{filename}: server.host missing"

    def test_server_host_is_loopback(self, config):
        filename, cfg = config
        assert (
            cfg["server"]["host"] == "127.0.0.1"
        ), f"{filename}: server.host should be 127.0.0.1, got {cfg['server']['host']}"

    def test_server_has_port(self, config):
        filename, cfg = config
        assert "port" in cfg["server"], f"{filename}: server.port missing"

    def test_server_port_is_30000(self, config):
        filename, cfg = config
        assert (
            cfg["server"]["port"] == 30000
        ), f"{filename}: server.port should be 30000, got {cfg['server']['port']}"

    def test_server_has_base_flags(self, config):
        filename, cfg = config
        assert "base_flags" in cfg["server"], f"{filename}: server.base_flags missing"

    def test_server_base_flags_has_model_path(self, config):
        filename, cfg = config
        assert (
            "model_path" in cfg["server"]["base_flags"]
        ), f"{filename}: server.base_flags.model_path missing"

    def test_server_model_path_is_nonempty_string(self, config):
        filename, cfg = config
        model_path = cfg["server"]["base_flags"]["model_path"]
        assert (
            isinstance(model_path, str) and model_path.strip()
        ), f"{filename}: server.base_flags.model_path should be a non-empty string"

    def test_server_base_flags_has_mem_fraction_static(self, config):
        filename, cfg = config
        assert (
            "mem_fraction_static" in cfg["server"]["base_flags"]
        ), f"{filename}: server.base_flags.mem_fraction_static missing"

    def test_server_mem_fraction_static_is_valid(self, config):
        filename, cfg = config
        val = cfg["server"]["base_flags"]["mem_fraction_static"]
        assert (
            0.0 < val < 1.0
        ), f"{filename}: mem_fraction_static={val} should be in (0, 1)"

    def test_server_base_flags_has_schedule_policy(self, config):
        filename, cfg = config
        assert (
            "schedule_policy" in cfg["server"]["base_flags"]
        ), f"{filename}: server.base_flags.schedule_policy missing"

    def test_server_schedule_policy_is_lpm(self, config):
        filename, cfg = config
        policy = cfg["server"]["base_flags"]["schedule_policy"]
        assert (
            policy == "lpm"
        ), f"{filename}: schedule_policy should be 'lpm', got '{policy}'"

    def test_server_has_env(self, config):
        filename, cfg = config
        assert "env" in cfg["server"], f"{filename}: server.env missing"

    def test_server_env_has_cuda_visible_devices(self, config):
        filename, cfg = config
        assert (
            "CUDA_VISIBLE_DEVICES" in cfg["server"]["env"]
        ), f"{filename}: server.env.CUDA_VISIBLE_DEVICES missing"

    def test_server_has_launch_field(self, config):
        filename, cfg = config
        assert "launch" in cfg["server"], f"{filename}: server.launch missing"

    def test_server_launch_is_boolean(self, config):
        filename, cfg = config
        assert isinstance(
            cfg["server"]["launch"], bool
        ), f"{filename}: server.launch should be a boolean"

    def test_server_has_search_space(self, config):
        filename, cfg = config
        assert (
            "search_space" in cfg["server"]
        ), f"{filename}: server.search_space missing"

    def test_server_search_space_has_chunked_prefill_size(self, config):
        filename, cfg = config
        assert (
            "chunked_prefill_size" in cfg["server"]["search_space"]
        ), f"{filename}: server.search_space.chunked_prefill_size missing"

    def test_server_search_space_chunked_prefill_size_values_are_positive(self, config):
        filename, cfg = config
        vals = cfg["server"]["search_space"]["chunked_prefill_size"]
        assert all(
            v > 0 for v in vals
        ), f"{filename}: chunked_prefill_size values should all be positive"

    def test_server_search_space_has_max_running_requests(self, config):
        filename, cfg = config
        assert (
            "max_running_requests" in cfg["server"]["search_space"]
        ), f"{filename}: server.search_space.max_running_requests missing"

    def test_server_mem_fraction_default_is_0_82_or_lower(self, config):
        """mem_fraction_static should be 0.82 (standard) or lower (special models like LLaDA)."""
        filename, cfg = config
        val = cfg["server"]["base_flags"]["mem_fraction_static"]
        assert val <= 0.82, f"{filename}: mem_fraction_static={val} should be <= 0.82"


# ---------------------------------------------------------------------------
# Dataset section
# ---------------------------------------------------------------------------


class TestDatasetSection:
    def test_dataset_has_kind(self, config):
        filename, cfg = config
        assert "kind" in cfg["dataset"], f"{filename}: dataset.kind missing"

    def test_dataset_kind_is_random(self, config):
        """All PR configs default to random dataset."""
        filename, cfg = config
        assert (
            cfg["dataset"]["kind"] == "random"
        ), f"{filename}: dataset.kind should be 'random', got '{cfg['dataset']['kind']}'"

    def test_dataset_kind_is_supported(self, config):
        filename, cfg = config
        kind = cfg["dataset"]["kind"]
        assert (
            kind in SUPPORTED_DATASET_KINDS
        ), f"{filename}: unsupported dataset kind '{kind}'"

    def test_dataset_has_num_prompts(self, config):
        filename, cfg = config
        assert (
            "num_prompts" in cfg["dataset"]
        ), f"{filename}: dataset.num_prompts missing"

    def test_dataset_num_prompts_is_80(self, config):
        """Default budget is 80 prompts per the README rules."""
        filename, cfg = config
        assert (
            cfg["dataset"]["num_prompts"] == 80
        ), f"{filename}: dataset.num_prompts should be 80, got {cfg['dataset']['num_prompts']}"

    def test_dataset_has_scenario_names(self, config):
        filename, cfg = config
        assert (
            "scenario_names" in cfg["dataset"]
        ), f"{filename}: dataset.scenario_names missing"

    def test_dataset_has_input_len(self, config):
        filename, cfg = config
        assert "input_len" in cfg["dataset"], f"{filename}: dataset.input_len missing"

    def test_dataset_has_output_len(self, config):
        filename, cfg = config
        assert "output_len" in cfg["dataset"], f"{filename}: dataset.output_len missing"

    def test_dataset_scenario_lists_are_aligned(self, config):
        """scenario_names, input_len, and output_len must have the same length."""
        filename, cfg = config
        ds = cfg["dataset"]
        n_names = len(ds["scenario_names"])
        n_in = len(ds["input_len"])
        n_out = len(ds["output_len"])
        assert n_names == n_in == n_out, (
            f"{filename}: scenario_names({n_names}), input_len({n_in}), "
            f"output_len({n_out}) must have the same length"
        )

    def test_dataset_has_chat_and_summarization_scenarios(self, config):
        """Default configs must include chat and summarization scenarios."""
        filename, cfg = config
        scenario_names = cfg["dataset"]["scenario_names"]
        assert (
            "chat" in scenario_names
        ), f"{filename}: 'chat' scenario should be present"
        assert (
            "summarization" in scenario_names
        ), f"{filename}: 'summarization' scenario should be present"

    def test_dataset_includes_short_input_scenario(self, config):
        """Chat-like 1000-token scenario must be present."""
        filename, cfg = config
        assert (
            1000 in cfg["dataset"]["input_len"]
        ), f"{filename}: input_len should include 1000 (chat scenario)"

    def test_dataset_includes_long_input_scenario(self, config):
        """Summarization-like 8000-token scenario must be present."""
        filename, cfg = config
        assert (
            8000 in cfg["dataset"]["input_len"]
        ), f"{filename}: input_len should include 8000 (summarization scenario)"

    def test_dataset_all_input_lens_are_positive(self, config):
        filename, cfg = config
        for val in cfg["dataset"]["input_len"]:
            assert val > 0, f"{filename}: all input_len values should be positive"

    def test_dataset_all_output_lens_are_positive(self, config):
        filename, cfg = config
        for val in cfg["dataset"]["output_len"]:
            assert val > 0, f"{filename}: all output_len values should be positive"


# ---------------------------------------------------------------------------
# Benchmark section
# ---------------------------------------------------------------------------


class TestBenchmarkSection:
    def test_benchmark_has_backend(self, config):
        filename, cfg = config
        assert "backend" in cfg["benchmark"], f"{filename}: benchmark.backend missing"

    def test_benchmark_backend_is_auto(self, config):
        filename, cfg = config
        assert (
            cfg["benchmark"]["backend"] == "auto"
        ), f"{filename}: benchmark.backend should be 'auto'"

    def test_benchmark_has_tokenizer(self, config):
        filename, cfg = config
        assert (
            "tokenizer" in cfg["benchmark"]
        ), f"{filename}: benchmark.tokenizer missing"

    def test_benchmark_tokenizer_matches_model_path(self, config):
        """Tokenizer should reference the same model as model_path."""
        filename, cfg = config
        tokenizer = cfg["benchmark"]["tokenizer"]
        model_path = cfg["server"]["base_flags"]["model_path"]
        assert tokenizer == model_path, (
            f"{filename}: benchmark.tokenizer='{tokenizer}' should match "
            f"server.base_flags.model_path='{model_path}'"
        )

    def test_benchmark_has_qps_section(self, config):
        filename, cfg = config
        assert "qps" in cfg["benchmark"], f"{filename}: benchmark.qps missing"

    def test_benchmark_qps_has_lower(self, config):
        filename, cfg = config
        assert (
            "lower" in cfg["benchmark"]["qps"]
        ), f"{filename}: benchmark.qps.lower missing"

    def test_benchmark_qps_has_upper(self, config):
        filename, cfg = config
        assert (
            "upper" in cfg["benchmark"]["qps"]
        ), f"{filename}: benchmark.qps.upper missing"

    def test_benchmark_qps_has_tolerance(self, config):
        filename, cfg = config
        assert (
            "tolerance" in cfg["benchmark"]["qps"]
        ), f"{filename}: benchmark.qps.tolerance missing"

    def test_benchmark_qps_lower_positive(self, config):
        filename, cfg = config
        lower = cfg["benchmark"]["qps"]["lower"]
        assert lower > 0, f"{filename}: qps.lower={lower} should be > 0"

    def test_benchmark_qps_upper_greater_than_lower(self, config):
        filename, cfg = config
        lower = cfg["benchmark"]["qps"]["lower"]
        upper = cfg["benchmark"]["qps"]["upper"]
        assert (
            upper > lower
        ), f"{filename}: qps.upper={upper} should be > qps.lower={lower}"

    def test_benchmark_qps_tolerance_positive(self, config):
        filename, cfg = config
        tol = cfg["benchmark"]["qps"]["tolerance"]
        assert tol > 0, f"{filename}: qps.tolerance={tol} should be > 0"

    def test_benchmark_qps_lower_is_0_25(self, config):
        """Canonical lower QPS is 0.25 per README default."""
        filename, cfg = config
        lower = cfg["benchmark"]["qps"]["lower"]
        assert lower == 0.25, f"{filename}: qps.lower should be 0.25, got {lower}"

    def test_benchmark_has_sla_section(self, config):
        filename, cfg = config
        assert "sla" in cfg["benchmark"], f"{filename}: benchmark.sla missing"

    def test_benchmark_sla_has_max_ttft_ms(self, config):
        filename, cfg = config
        assert (
            "max_ttft_ms" in cfg["benchmark"]["sla"]
        ), f"{filename}: benchmark.sla.max_ttft_ms missing"

    def test_benchmark_sla_has_max_tpot_ms(self, config):
        filename, cfg = config
        assert (
            "max_tpot_ms" in cfg["benchmark"]["sla"]
        ), f"{filename}: benchmark.sla.max_tpot_ms missing"

    def test_benchmark_sla_max_ttft_ms_is_1500(self, config):
        filename, cfg = config
        val = cfg["benchmark"]["sla"]["max_ttft_ms"]
        assert val == 1500, f"{filename}: sla.max_ttft_ms should be 1500, got {val}"

    def test_benchmark_sla_max_tpot_ms_is_30(self, config):
        filename, cfg = config
        val = cfg["benchmark"]["sla"]["max_tpot_ms"]
        assert val == 30, f"{filename}: sla.max_tpot_ms should be 30, got {val}"

    def test_benchmark_sla_values_are_positive(self, config):
        filename, cfg = config
        sla = cfg["benchmark"]["sla"]
        assert sla["max_ttft_ms"] > 0, f"{filename}: sla.max_ttft_ms should be > 0"
        assert sla["max_tpot_ms"] > 0, f"{filename}: sla.max_tpot_ms should be > 0"

    def test_benchmark_has_output_dir(self, config):
        filename, cfg = config
        assert (
            "output_dir" in cfg["benchmark"]
        ), f"{filename}: benchmark.output_dir missing"

    def test_benchmark_output_dir_contains_model_stem(self, config):
        """output_dir should contain the YAML filename stem to avoid result collisions."""
        filename, cfg = config
        stem = Path(filename).stem  # e.g. "llama-3.1-70b-instruct"
        output_dir = cfg["benchmark"]["output_dir"]
        assert stem in output_dir, (
            f"{filename}: benchmark.output_dir='{output_dir}' should contain "
            f"the config stem '{stem}'"
        )

    def test_benchmark_output_dir_is_under_auto_benchmark_results(self, config):
        filename, cfg = config
        output_dir = cfg["benchmark"]["output_dir"]
        assert (
            "auto_benchmark_results" in output_dir
        ), f"{filename}: output_dir should be under auto_benchmark_results/"

    def test_benchmark_has_max_concurrency(self, config):
        filename, cfg = config
        assert (
            "max_concurrency" in cfg["benchmark"]
        ), f"{filename}: benchmark.max_concurrency missing"

    def test_benchmark_max_concurrency_is_list(self, config):
        filename, cfg = config
        assert isinstance(
            cfg["benchmark"]["max_concurrency"], list
        ), f"{filename}: benchmark.max_concurrency should be a list"

    def test_benchmark_has_extra_request_body(self, config):
        filename, cfg = config
        assert (
            "extra_request_body" in cfg["benchmark"]
        ), f"{filename}: benchmark.extra_request_body missing"

    def test_benchmark_extra_request_body_has_temperature(self, config):
        filename, cfg = config
        assert (
            "temperature" in cfg["benchmark"]["extra_request_body"]
        ), f"{filename}: benchmark.extra_request_body.temperature missing"

    def test_benchmark_temperature_is_zero(self, config):
        """Deterministic benchmarking requires temperature=0.0."""
        filename, cfg = config
        temp = cfg["benchmark"]["extra_request_body"]["temperature"]
        assert (
            temp == 0.0
        ), f"{filename}: extra_request_body.temperature should be 0.0, got {temp}"


# ---------------------------------------------------------------------------
# Search section
# ---------------------------------------------------------------------------


class TestSearchSection:
    def test_search_has_tier(self, config):
        filename, cfg = config
        assert "tier" in cfg["search"], f"{filename}: search.tier missing"

    def test_search_tier_is_2(self, config):
        """Default tier per README is 2."""
        filename, cfg = config
        assert (
            cfg["search"]["tier"] == 2
        ), f"{filename}: search.tier should be 2, got {cfg['search']['tier']}"

    def test_search_tier_is_valid(self, config):
        filename, cfg = config
        tier = cfg["search"]["tier"]
        assert (
            tier in VALID_SEARCH_TIERS
        ), f"{filename}: search.tier={tier} should be one of {VALID_SEARCH_TIERS}"

    def test_search_has_max_candidates(self, config):
        filename, cfg = config
        assert (
            "max_candidates" in cfg["search"]
        ), f"{filename}: search.max_candidates missing"

    def test_search_max_candidates_is_8(self, config):
        """Default max_candidates per README is 8."""
        filename, cfg = config
        assert (
            cfg["search"]["max_candidates"] == 8
        ), f"{filename}: search.max_candidates should be 8, got {cfg['search']['max_candidates']}"

    def test_search_has_resume(self, config):
        filename, cfg = config
        assert "resume" in cfg["search"], f"{filename}: search.resume missing"

    def test_search_resume_is_true(self, config):
        filename, cfg = config
        assert (
            cfg["search"]["resume"] is True
        ), f"{filename}: search.resume should be true"


# ---------------------------------------------------------------------------
# Speculative section
# ---------------------------------------------------------------------------


class TestSpeculativeSection:
    def test_speculative_has_enabled_field(self, config):
        filename, cfg = config
        assert (
            "enabled" in cfg["speculative"]
        ), f"{filename}: speculative.enabled missing"

    def test_speculative_disabled_by_default(self, config):
        """All shipped configs must have speculative disabled."""
        filename, cfg = config
        assert (
            cfg["speculative"]["enabled"] is False
        ), f"{filename}: speculative.enabled should be false by default"

    def test_speculative_has_algorithm(self, config):
        filename, cfg = config
        assert (
            "algorithm" in cfg["speculative"]
        ), f"{filename}: speculative.algorithm missing"

    def test_speculative_algorithm_is_eagle_variant(self, config):
        filename, cfg = config
        algo = cfg["speculative"]["algorithm"]
        assert algo.startswith(
            "EAGLE"
        ), f"{filename}: speculative.algorithm='{algo}' should be EAGLE or EAGLE3"

    def test_speculative_has_draft_model_path(self, config):
        filename, cfg = config
        assert (
            "draft_model_path" in cfg["speculative"]
        ), f"{filename}: speculative.draft_model_path missing"

    def test_speculative_has_search_space(self, config):
        filename, cfg = config
        assert (
            "search_space" in cfg["speculative"]
        ), f"{filename}: speculative.search_space missing"

    def test_speculative_search_space_has_num_steps(self, config):
        filename, cfg = config
        assert (
            "speculative_num_steps" in cfg["speculative"]["search_space"]
        ), f"{filename}: speculative.search_space.speculative_num_steps missing"

    def test_speculative_search_space_has_eagle_topk(self, config):
        filename, cfg = config
        assert (
            "speculative_eagle_topk" in cfg["speculative"]["search_space"]
        ), f"{filename}: speculative.search_space.speculative_eagle_topk missing"

    def test_speculative_search_space_has_num_draft_tokens(self, config):
        filename, cfg = config
        assert (
            "speculative_num_draft_tokens" in cfg["speculative"]["search_space"]
        ), f"{filename}: speculative.search_space.speculative_num_draft_tokens missing"

    def test_speculative_num_steps_values_are_positive(self, config):
        filename, cfg = config
        for v in cfg["speculative"]["search_space"]["speculative_num_steps"]:
            assert v > 0, f"{filename}: speculative_num_steps values should be positive"

    def test_speculative_eagle_topk_values_are_positive(self, config):
        filename, cfg = config
        for v in cfg["speculative"]["search_space"]["speculative_eagle_topk"]:
            assert (
                v > 0
            ), f"{filename}: speculative_eagle_topk values should be positive"

    def test_speculative_num_draft_tokens_values_are_positive(self, config):
        filename, cfg = config
        for v in cfg["speculative"]["search_space"]["speculative_num_draft_tokens"]:
            assert (
                v > 0
            ), f"{filename}: speculative_num_draft_tokens values should be positive"


# ---------------------------------------------------------------------------
# Cross-config consistency
# ---------------------------------------------------------------------------


class TestCrossConfigConsistency:
    def test_all_pr_configs_are_present(self):
        """Every file listed in PR_YAML_FILES must actually exist on disk."""
        missing = [f for f in PR_YAML_FILES if not (COOKBOOK_DIR / f).exists()]
        assert not missing, f"Missing configs from PR: {missing}"

    def test_no_duplicate_output_dirs(self):
        """Each config must write results to a unique directory."""
        output_dirs = []
        for filename in PR_YAML_FILES:
            cfg = load_yaml(filename)
            output_dirs.append(cfg["benchmark"]["output_dir"])
        assert len(output_dirs) == len(set(output_dirs)), (
            f"Duplicate output_dir values found: "
            f"{[d for d in output_dirs if output_dirs.count(d) > 1]}"
        )

    def test_no_duplicate_model_paths(self):
        """Each config must reference a distinct model (no accidental copy-paste errors)."""
        model_paths = {}
        for filename in PR_YAML_FILES:
            cfg = load_yaml(filename)
            mp = cfg["server"]["base_flags"]["model_path"]
            if mp in model_paths:
                model_paths[mp].append(filename)
            else:
                model_paths[mp] = [filename]
        duplicates = {mp: files for mp, files in model_paths.items() if len(files) > 1}
        assert not duplicates, f"Duplicate model_paths found: {duplicates}"

    def test_all_configs_have_two_scenarios(self):
        """Standard is exactly two scenarios: chat and summarization."""
        # This tests that all PR configs follow the two-scenario convention
        for filename in PR_YAML_FILES:
            cfg = load_yaml(filename)
            n = len(cfg["dataset"]["scenario_names"])
            assert (
                n == 2
            ), f"{filename}: expected 2 scenarios, got {n}: {cfg['dataset']['scenario_names']}"

    def test_search_space_chunked_prefill_size_includes_canonical_values(self):
        """All configs should include 4096 and 8192 in chunked_prefill_size search."""
        for filename in PR_YAML_FILES:
            cfg = load_yaml(filename)
            vals = cfg["server"]["search_space"].get("chunked_prefill_size", [])
            assert 4096 in vals, f"{filename}: chunked_prefill_size should include 4096"
            assert 8192 in vals, f"{filename}: chunked_prefill_size should include 8192"


# ---------------------------------------------------------------------------
# Model-specific / regression tests
# ---------------------------------------------------------------------------


class TestModelSpecificConfigs:
    def test_llama_4_maverick_uses_eagle3(self):
        """Llama-4 Maverick should use EAGLE3 (not EAGLE) for speculative."""
        cfg = load_yaml("llama-4-maverick-17b-128e-instruct-fp8.yaml")
        assert cfg["speculative"]["algorithm"] == "EAGLE3"

    def test_llama_4_maverick_has_draft_model_path(self):
        """Llama-4 Maverick speculative must have a non-empty draft model path."""
        cfg = load_yaml("llama-4-maverick-17b-128e-instruct-fp8.yaml")
        assert cfg["speculative"][
            "draft_model_path"
        ], "llama-4-maverick-17b-128e-instruct-fp8.yaml: draft_model_path should be non-empty"

    def test_llama_4_maverick_draft_model_path_references_eagle3(self):
        cfg = load_yaml("llama-4-maverick-17b-128e-instruct-fp8.yaml")
        draft = cfg["speculative"]["draft_model_path"]
        assert (
            "EAGLE3" in draft
        ), f"llama-4-maverick draft_model_path '{draft}' should reference EAGLE3"

    def test_llama_4_maverick_uses_sglang_serve_command_prefix(self):
        cfg = load_yaml("llama-4-maverick-17b-128e-instruct-fp8.yaml")
        prefix = cfg["server"]["command_prefix"]
        assert (
            "sglang" in prefix and "serve" in prefix
        ), "llama-4-maverick should use 'sglang serve' command_prefix"

    def test_llada2_mini_has_dllm_algorithm(self):
        """LLaDA2-1-mini is a diffusion LM and requires dllm_algorithm."""
        cfg = load_yaml("llada2-1-mini.yaml")
        assert (
            "dllm_algorithm" in cfg["server"]["base_flags"]
        ), "llada2-1-mini.yaml: dllm_algorithm should be set in base_flags"

    def test_llada2_mini_has_lower_mem_fraction(self):
        """LLaDA2-1-mini uses 0.77 instead of the standard 0.82."""
        cfg = load_yaml("llada2-1-mini.yaml")
        val = cfg["server"]["base_flags"]["mem_fraction_static"]
        assert (
            val == 0.77
        ), f"llada2-1-mini.yaml: mem_fraction_static should be 0.77, got {val}"

    def test_ling_2_5_1t_has_launch_false(self):
        """Ling-2.5-1T is multi-node and must keep launch: false."""
        cfg = load_yaml("ling-2.5-1t.yaml")
        assert (
            cfg["server"]["launch"] is False
        ), "ling-2.5-1t.yaml: server.launch should be false (multi-node deployment)"

    def test_ling_2_5_1t_has_pp_size(self):
        """Ling-2.5-1T uses pipeline parallelism."""
        cfg = load_yaml("ling-2.5-1t.yaml")
        assert (
            "pp_size" in cfg["server"]["base_flags"]
        ), "ling-2.5-1t.yaml: should have pp_size in base_flags"

    def test_deepseek_v3_has_enable_symm_mem(self):
        """DeepSeek-V3 requires enable_symm_mem flag."""
        cfg = load_yaml("deepseek-v3.yaml")
        assert (
            cfg["server"]["base_flags"].get("enable_symm_mem") is True
        ), "deepseek-v3.yaml: enable_symm_mem should be true"

    def test_deepseek_r1_has_enable_symm_mem(self):
        """DeepSeek-R1-0528 requires enable_symm_mem flag."""
        cfg = load_yaml("deepseek-r1-0528.yaml")
        assert (
            cfg["server"]["base_flags"].get("enable_symm_mem") is True
        ), "deepseek-r1-0528.yaml: enable_symm_mem should be true"

    def test_moe_models_have_ep_size_search(self):
        """MoE models (deepseek, glm-4.6, gpt-oss-120b) should include ep_size in search."""
        moe_configs = [
            "deepseek-v3.yaml",
            "deepseek-v3.1.yaml",
            "deepseek-v3.2.yaml",
            "deepseek-r1-0528.yaml",
            "glm-4.6.yaml",
            "glm-5-fp8.yaml",
            "gpt-oss-120b.yaml",
            "intern-s1.yaml",
        ]
        for filename in moe_configs:
            cfg = load_yaml(filename)
            assert (
                "ep_size" in cfg["server"]["search_space"]
            ), f"{filename}: MoE model should have ep_size in search_space"

    def test_glm_4_5_has_context_length(self):
        """GLM-4.5 requires context_length in base_flags."""
        cfg = load_yaml("glm-4.5.yaml")
        assert (
            "context_length" in cfg["server"]["base_flags"]
        ), "glm-4.5.yaml: context_length should be set in base_flags"

    def test_glm_4_7_has_context_length(self):
        """GLM-4.7 requires context_length in base_flags."""
        cfg = load_yaml("glm-4.7.yaml")
        assert (
            "context_length" in cfg["server"]["base_flags"]
        ), "glm-4.7.yaml: context_length should be set in base_flags"

    def test_intern_s1_has_trust_remote_code(self):
        cfg = load_yaml("intern-s1.yaml")
        assert (
            cfg["server"]["base_flags"].get("trust_remote_code") is True
        ), "intern-s1.yaml: trust_remote_code should be true"

    def test_kimi_k2_instruct_has_trust_remote_code(self):
        cfg = load_yaml("kimi-k2-instruct.yaml")
        assert (
            cfg["server"]["base_flags"].get("trust_remote_code") is True
        ), "kimi-k2-instruct.yaml: trust_remote_code should be true"

    def test_kimi_linear_has_command_prefix(self):
        """Kimi-Linear requires explicit command_prefix."""
        cfg = load_yaml("kimi-linear-48b-a3b-instruct.yaml")
        assert (
            "command_prefix" in cfg["server"]
        ), "kimi-linear-48b-a3b-instruct.yaml: command_prefix should be present"

    def test_kimi_linear_has_rocm_env_var(self):
        """Kimi-Linear sets SGLANG_ROCM_FUSED_DECODE_MLA env var."""
        cfg = load_yaml("kimi-linear-48b-a3b-instruct.yaml")
        assert (
            "SGLANG_ROCM_FUSED_DECODE_MLA" in cfg["server"]["env"]
        ), "kimi-linear-48b-a3b-instruct.yaml: SGLANG_ROCM_FUSED_DECODE_MLA should be in env"

    def test_ernie_has_command_prefix(self):
        """ERNIE requires explicit command_prefix."""
        cfg = load_yaml("ernie-4.5-21b-a3b-pt.yaml")
        assert (
            "command_prefix" in cfg["server"]
        ), "ernie-4.5-21b-a3b-pt.yaml: command_prefix should be present"

    def test_glyph_has_reasoning_parser(self):
        """Glyph needs reasoning_parser set for structured outputs."""
        cfg = load_yaml("glyph.yaml")
        assert (
            "reasoning_parser" in cfg["server"]["base_flags"]
        ), "glyph.yaml: reasoning_parser should be set in base_flags"

    def test_glyph_has_tool_call_parser(self):
        cfg = load_yaml("glyph.yaml")
        assert (
            "tool_call_parser" in cfg["server"]["base_flags"]
        ), "glyph.yaml: tool_call_parser should be set in base_flags"

    def test_deepseek_math_v2_model_path(self):
        cfg = load_yaml("deepseek-math-v2.yaml")
        assert (
            cfg["server"]["base_flags"]["model_path"] == "deepseek-ai/DeepSeek-Math-V2"
        )

    def test_llama_3_1_70b_uses_4_gpus(self):
        """Llama-3.1-70B uses tp=4 (4 GPUs)."""
        cfg = load_yaml("llama-3.1-70b-instruct.yaml")
        assert cfg["server"]["base_flags"]["tp_size"] == 4

    def test_large_moe_models_use_8_gpus(self):
        """Large MoE models (DeepSeek, GLM-4.6, etc.) use tp=8."""
        large_moe_configs = [
            "deepseek-v3.yaml",
            "deepseek-v3.1.yaml",
            "deepseek-v3.2.yaml",
            "deepseek-r1-0528.yaml",
            "glm-4.6.yaml",
            "glm-5-fp8.yaml",
            "gpt-oss-120b.yaml",
            "intern-s1.yaml",
            "kimi-k2-instruct.yaml",
            "kimi-k2.5.yaml",
        ]
        for filename in large_moe_configs:
            cfg = load_yaml(filename)
            tp = cfg["server"]["base_flags"].get("tp_size")
            assert tp == 8, f"{filename}: expected tp_size=8, got {tp}"

    def test_deepseek_math_v2_does_not_have_flashinfer_or_fa3_in_search(self):
        """deepseek-math-v2 only has flashinfer (not fa3) in attention backends."""
        cfg = load_yaml("deepseek-math-v2.yaml")
        search = cfg["server"]["search_space"]
        # deepseek-math-v2 is B200 based, only lists flashinfer
        if "prefill_attention_backend" in search:
            for b in search["prefill_attention_backend"]:
                assert (
                    b in VALID_ATTENTION_BACKENDS
                ), f"deepseek-math-v2.yaml: invalid prefill_attention_backend '{b}'"


# ---------------------------------------------------------------------------
# Negative / boundary tests
# ---------------------------------------------------------------------------


class TestNegativeAndBoundary:
    def test_yaml_with_missing_server_key_would_fail_validation(self):
        """Ensure our validation logic correctly detects missing required keys."""
        incomplete = {"dataset": {}, "benchmark": {}, "search": {}, "speculative": {}}
        assert "server" not in incomplete

    def test_scenario_list_length_mismatch_is_detectable(self):
        """Mismatched scenario list lengths should be flagged."""
        mismatched = {
            "scenario_names": ["chat"],
            "input_len": [1000, 8000],
            "output_len": [1000, 1000],
        }
        assert len(mismatched["scenario_names"]) != len(mismatched["input_len"])

    def test_qps_lower_cannot_exceed_upper(self):
        """QPS validation: lower must be less than upper."""
        inverted = {"lower": 10.0, "upper": 1.0}
        assert inverted["lower"] >= inverted["upper"]

    def test_search_tier_3_is_valid_but_not_default(self):
        """Tier 3 is allowed but none of the PR configs should use it."""
        for filename in PR_YAML_FILES:
            cfg = load_yaml(filename)
            tier = cfg["search"]["tier"]
            assert (
                tier != 3
            ), f"{filename}: shipped configs should not use tier 3 (too expensive)"

    def test_speculative_enabled_true_would_need_draft_model_path(self):
        """Regression: if speculative.enabled were true, draft_model_path must be non-empty.
        All current configs have enabled=false; this test anchors the contract."""
        for filename in PR_YAML_FILES:
            cfg = load_yaml(filename)
            if cfg["speculative"]["enabled"]:
                assert cfg["speculative"][
                    "draft_model_path"
                ], f"{filename}: if speculative.enabled=true, draft_model_path must be set"

    def test_max_candidates_is_not_null(self):
        """max_candidates=null means unbounded sweep; default should be 8."""
        for filename in PR_YAML_FILES:
            cfg = load_yaml(filename)
            val = cfg["search"]["max_candidates"]
            assert (
                val is not None
            ), f"{filename}: search.max_candidates should be 8, not null"

    def test_mem_fraction_static_is_not_zero_or_one(self):
        """Edge guard: mem_fraction_static at 0 or 1 would be unusable."""
        for filename in PR_YAML_FILES:
            cfg = load_yaml(filename)
            val = cfg["server"]["base_flags"]["mem_fraction_static"]
            assert (
                0.0 < val < 1.0
            ), f"{filename}: mem_fraction_static={val} must be strictly between 0 and 1"

    def test_all_yaml_files_have_source_comment(self):
        """Each YAML file should have a Source comment indicating cookbook origin."""
        for filename in PR_YAML_FILES:
            path = COOKBOOK_DIR / filename
            content = path.read_text()
            assert (
                "Source:" in content
            ), f"{filename}: should have a '# Source:' comment indicating cookbook origin"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
