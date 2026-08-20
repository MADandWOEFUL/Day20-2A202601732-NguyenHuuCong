"""Unit tests for Supervisor routing policy and workflow transitions."""

from multi_agent_research_lab.agents.supervisor import SupervisorAgent
from multi_agent_research_lab.core.schemas import ResearchQuery, SourceDocument
from multi_agent_research_lab.core.state import ResearchState


def test_supervisor_routes_to_researcher_when_empty() -> None:
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    supervisor = SupervisorAgent(max_iterations=6)
    updated = supervisor.run(state)
    assert updated.route_history == ["researcher"]
    assert updated.iteration == 1


def test_supervisor_routes_to_analyst_when_sources_present() -> None:
    state = ResearchState(
        request=ResearchQuery(query="Explain multi-agent systems"),
        sources=[SourceDocument(title="Doc 1", snippet="Content 1")],
        research_notes="Found notes on agents.",
    )
    supervisor = SupervisorAgent(max_iterations=6)
    updated = supervisor.run(state)
    assert updated.route_history == ["analyst"]


def test_supervisor_routes_to_writer_when_analysis_present() -> None:
    state = ResearchState(
        request=ResearchQuery(query="Explain multi-agent systems"),
        sources=[SourceDocument(title="Doc 1", snippet="Content 1")],
        research_notes="Found notes.",
        analysis_notes="Analyzed insights.",
    )
    supervisor = SupervisorAgent(max_iterations=6)
    updated = supervisor.run(state)
    assert updated.route_history == ["writer"]


def test_supervisor_routes_to_done_when_final_answer_present() -> None:
    state = ResearchState(
        request=ResearchQuery(query="Explain multi-agent systems"),
        final_answer="Comprehensive final report.",
    )
    supervisor = SupervisorAgent(max_iterations=6)
    updated = supervisor.run(state)
    assert updated.route_history == ["done"]


def test_supervisor_stops_at_max_iterations() -> None:
    state = ResearchState(
        request=ResearchQuery(query="Explain multi-agent systems"),
        iteration=6,
    )
    supervisor = SupervisorAgent(max_iterations=6)
    updated = supervisor.run(state)
    assert updated.route_history == ["done"]
