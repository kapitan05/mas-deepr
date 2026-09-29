import pytest

from mas_deepr.config import TOOL_SCOPES, get_tool_scope


def test_wiki_paper_scope_excludes_web_search_and_tavily() -> None:
    tools = set(get_tool_scope("wiki_paper").split(","))
    assert "web_search" not in tools
    assert "tavily_search" not in tools
    assert tools == {
        "wikipedia_search",
        "fetch_page",
        "semantic_scholar_search",
        "pubmed_search",
    }


def test_general_scope_includes_tavily_as_paid_fallback() -> None:
    """Confirmed live 2026-09-17: SearXNG's free engines (Bing/DuckDuckGo)
    degrade (CAPTCHA/connection errors) even at smoke-test volume --
    general-scope eval (bounded, one benchmark per milestone, not GRPO
    training volume) gets Tavily as a reliability fallback."""
    tools = set(get_tool_scope("general").split(","))
    assert tools == {
        "web_search",
        "fetch_page",
        "wikipedia_search",
        "semantic_scholar_search",
        "pubmed_search",
        "tavily_search",
    }


def test_get_tool_scope_raises_with_known_names_on_bad_input() -> None:
    with pytest.raises(KeyError, match="general"):
        get_tool_scope("nonexistent")


def test_tool_scopes_registry_has_exactly_two_named_scopes() -> None:
    assert set(TOOL_SCOPES) == {"general", "wiki_paper"}
