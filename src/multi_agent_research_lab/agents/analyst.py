"""Analyst agent for evaluating findings, assessing evidence, and extracting insights."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)


ANALYST_SYSTEM_PROMPT = """You are an expert Systems Analyst in a multi-agent workflow.
Your job is to analyze research notes and raw sources into structured analytical insights.

Specifically:
1. Synthesize Key Themes & Core Mechanisms: Group findings into coherent structural themes.
2. Comparative Analysis & Trade-offs: Compare alternatives (latency vs cost, accuracy vs speed).
3. Evidence Strength & Gaps: Evaluate quality; flag weak assumptions or missing points.
4. Failure Modes & Guardrails: Identify practical operational risks and mitigations.

Format your output with clear headings and rigorous technical precision."""


class AnalystAgent(BaseAgent):
    """Turns research notes and sources into structured analytical insights."""

    name = "analyst"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.analysis_notes`."""
        logger.info("Analyst evaluating research notes for: %s", state.request.query)

        research_content = state.research_notes or "No explicit research notes available."
        sources_summary = "\n".join(
            f"- [{idx}] {s.title}: {s.snippet[:150]}..." for idx, s in enumerate(state.sources, 1)
        )

        user_prompt = (
            f"User Query: {state.request.query}\n"
            f"Target Audience: {state.request.audience}\n\n"
            f"Research Notes from Researcher:\n{research_content}\n\n"
            f"Available Sources:\n"
            + (sources_summary if sources_summary else "No external sources list.")
            + "\n\nPlease conduct a thorough technical analysis according to instructions."
        )

        response = self.llm_client.complete(
            system_prompt=ANALYST_SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )

        state.analysis_notes = response.content
        state.agent_results.append(
            AgentResult(
                agent=AgentName.ANALYST,
                content=response.content,
                metadata={
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "cost_usd": response.cost_usd,
                },
            )
        )
        state.add_trace_event(
            "analyst_completed",
            {
                "cost_usd": response.cost_usd,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
            },
        )
        return state
