"""``scripts/run_milestone_eval.py``'s benchmark/tool-scope wiring --
imported by path since scripts/ isn't a package (matches
test_run_milestone_eval_warmup.py's pattern).
"""

import importlib.util
import sys
from pathlib import Path

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_milestone_eval.py"
_spec = importlib.util.spec_from_file_location(
    "run_milestone_eval_benchmarks", _SCRIPT_PATH
)
assert _spec is not None and _spec.loader is not None
run_milestone_eval = importlib.util.module_from_spec(_spec)
sys.modules["run_milestone_eval_benchmarks"] = run_milestone_eval
_spec.loader.exec_module(run_milestone_eval)


def test_browsecomp_dropped_from_benchmarks() -> None:
    assert "browsecomp" not in run_milestone_eval._BENCHMARK_LOADERS
    assert "browsecomp" not in run_milestone_eval._BENCHMARK_TOOL_SCOPE


def test_every_benchmark_has_a_tool_scope_and_vice_versa() -> None:
    assert (
        run_milestone_eval._BENCHMARK_LOADERS.keys()
        == run_milestone_eval._BENCHMARK_TOOL_SCOPE.keys()
    )


def test_wiki_paper_scope_benchmarks() -> None:
    scope = run_milestone_eval._BENCHMARK_TOOL_SCOPE
    assert scope["frames"] == "wiki_paper"
    assert scope["research_qa"] == "wiki_paper"
    assert scope["health_bench"] == "wiki_paper"


def test_general_scope_benchmarks() -> None:
    assert run_milestone_eval._BENCHMARK_TOOL_SCOPE["research_rubrics"] == "general"


def test_parse_benchmark_limits_basic() -> None:
    assert run_milestone_eval._parse_benchmark_limits("health_bench=200") == {
        "health_bench": 200
    }


def test_parse_benchmark_limits_multiple() -> None:
    assert run_milestone_eval._parse_benchmark_limits("health_bench=200,frames=50") == {
        "health_bench": 200,
        "frames": 50,
    }


def test_parse_benchmark_limits_empty_string_gives_no_caps() -> None:
    assert run_milestone_eval._parse_benchmark_limits("") == {}


def test_parse_benchmark_limits_rejects_unknown_benchmark() -> None:
    try:
        run_milestone_eval._parse_benchmark_limits("not_a_real_benchmark=10")
    except ValueError as e:
        assert "not_a_real_benchmark" in str(e)
    else:
        raise AssertionError("expected ValueError for an unknown benchmark name")
