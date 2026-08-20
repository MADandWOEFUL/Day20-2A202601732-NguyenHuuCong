"""Supervisor / router for multi-agent workflow."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.state import ResearchState

logger = logging.getLogger(__name__)


class SupervisorAgent(BaseAgent):
    """Decides which worker should run next and when to stop."""

    name = "supervisor"

    def __init__(self, settings: Settings | None = None, max_iterations: int | None = None) -> None:
        self.settings = settings or get_settings()
        self.max_iterations = (
            max_iterations if max_iterations is not None else self.settings.max_iterations
        )

    def decide_next(self, state: ResearchState) -> str:
        """Inspect state and return next node ('researcher', 'analyst', 'writer', 'done')."""
        # Guardrail: stop if max iterations reached
        if state.iteration >= self.max_iterations:
            logger.warning("Max iterations (%d) reached. Stopping.", self.max_iterations)
            return "done"

        # If final answer already produced, we are done
        if state.final_answer:
            return "done"

        # If we have analysis notes, next step is writing the final report
        if state.analysis_notes:
            return "writer"

        # If we have sources or research notes, next step is analysis
        if state.research_notes or state.sources:
            return "analyst"

        # Initial state: need research first
        return "researcher"

    def run(self, state: ResearchState) -> ResearchState:
        """Evaluate state, determine next route, and record the decision."""
        next_route = self.decide_next(state)
        state.record_route(next_route)
        state.add_trace_event(
            "supervisor_decision",
            {
                "next_route": next_route,
                "iteration": state.iteration,
                "has_sources": len(state.sources) > 0,
                "has_research_notes": state.research_notes is not None,
                "has_analysis_notes": state.analysis_notes is not None,
                "has_final_answer": state.final_answer is not None,
            },
        )
        return state
