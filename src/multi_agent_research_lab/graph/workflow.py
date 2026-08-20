"""LangGraph workflow orchestrating Supervisor, Researcher, Analyst, Writer, and Critic agents."""

import logging
from typing import Any

from langgraph.graph import END, START, StateGraph

from multi_agent_research_lab.agents.analyst import AnalystAgent
from multi_agent_research_lab.agents.critic import CriticAgent
from multi_agent_research_lab.agents.researcher import ResearcherAgent
from multi_agent_research_lab.agents.supervisor import SupervisorAgent
from multi_agent_research_lab.agents.writer import WriterAgent
from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient

logger = logging.getLogger(__name__)


class MultiAgentWorkflow:
    """Builds, compiles, and executes the multi-agent graph with LangGraph."""

    def __init__(
        self,
        settings: Settings | None = None,
        llm_client: LLMClient | None = None,
        search_client: SearchClient | None = None,
        enable_critic: bool = True,
    ) -> None:
        self.settings = settings or get_settings()
        self.llm_client = llm_client or LLMClient(settings=self.settings)
        self.search_client = search_client or SearchClient(settings=self.settings)
        self.enable_critic = enable_critic

        self.supervisor = SupervisorAgent(settings=self.settings)
        self.researcher = ResearcherAgent(
            search_client=self.search_client,
            llm_client=self.llm_client,
        )
        self.analyst = AnalystAgent(llm_client=self.llm_client)
        self.writer = WriterAgent(llm_client=self.llm_client)
        self.critic = CriticAgent(llm_client=self.llm_client)

        self._compiled_graph: Any = None

    def _ensure_state(self, state_input: Any) -> ResearchState:
        if isinstance(state_input, ResearchState):
            return state_input
        if isinstance(state_input, dict):
            return ResearchState.model_validate(state_input)
        raise TypeError(f"Expected ResearchState or dict, got {type(state_input)}")

    def _supervisor_node(self, state: Any) -> dict[str, Any]:
        s = self._ensure_state(state)
        updated = self.supervisor.run(s)
        return updated.model_dump()

    def _researcher_node(self, state: Any) -> dict[str, Any]:
        s = self._ensure_state(state)
        updated = self.researcher.run(s)
        return updated.model_dump()

    def _analyst_node(self, state: Any) -> dict[str, Any]:
        s = self._ensure_state(state)
        updated = self.analyst.run(s)
        return updated.model_dump()

    def _writer_node(self, state: Any) -> dict[str, Any]:
        s = self._ensure_state(state)
        updated = self.writer.run(s)
        return updated.model_dump()

    def _critic_node(self, state: Any) -> dict[str, Any]:
        s = self._ensure_state(state)
        if self.enable_critic:
            updated = self.critic.run(s)
            return updated.model_dump()
        return s.model_dump()

    def _route_after_supervisor(self, state: Any) -> str:
        s = self._ensure_state(state)
        if s.route_history:
            last_route = s.route_history[-1]
            if last_route in ("researcher", "analyst", "writer"):
                return last_route
        return "done"

    def build(self) -> Any:
        """Create a LangGraph StateGraph connecting supervisor to specialized workers."""
        builder = StateGraph(ResearchState)

        # Add agent nodes
        builder.add_node("supervisor", self._supervisor_node)
        builder.add_node("researcher", self._researcher_node)
        builder.add_node("analyst", self._analyst_node)
        builder.add_node("writer", self._writer_node)
        if self.enable_critic:
            builder.add_node("critic", self._critic_node)

        # Flow starting at supervisor
        builder.add_edge(START, "supervisor")

        # Supervisor routes conditionally to workers or finishes
        builder.add_conditional_edges(
            "supervisor",
            self._route_after_supervisor,
            {
                "researcher": "researcher",
                "analyst": "analyst",
                "writer": "writer",
                "done": END,
            },
        )

        # Workers loop back to supervisor (centralized orchestrator pattern)
        builder.add_edge("researcher", "supervisor")
        builder.add_edge("analyst", "supervisor")

        if self.enable_critic:
            # Writer hands off to Critic, then back to supervisor
            builder.add_edge("writer", "critic")
            builder.add_edge("critic", "supervisor")
        else:
            builder.add_edge("writer", "supervisor")

        return builder

    def compile(self) -> Any:
        if self._compiled_graph is None:
            builder = self.build()
            self._compiled_graph = builder.compile()
        return self._compiled_graph

    def run(self, state: ResearchState) -> ResearchState:
        """Execute the workflow graph and return the final state."""
        app = self.compile()
        raw_result = app.invoke(state)
        return self._ensure_state(raw_result)
