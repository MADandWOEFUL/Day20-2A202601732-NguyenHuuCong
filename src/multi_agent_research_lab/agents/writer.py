"""Writer agent for synthesizing research and analysis into a polished report."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)


WRITER_SYSTEM_PROMPT = """You are an expert Technical Research Writer.
Your job is to synthesize research and analytical insights into a high-quality report.

Requirements:
1. Grounding & Citations: Every key factual claim MUST cite its source (e.g. [Source 1], [ID]).
2. Structure:
   - # Title & Executive Summary
   - ## 1. Problem Framing & Objectives
   - ## 2. Core Technical Architecture & Mechanisms
   - ## 3. Comparative Analysis & Trade-offs (include a comparison table)
   - ## 4. Production Failure Modes & Guardrails
   - ## 5. Actionable Recommendations & Takeaways
   - ## References (List all cited sources with titles and URLs/IDs)
3. Tone: Tailor explanations to the requested audience with an objective engineering perspective.
4. Completeness: Provide substantial depth with concrete examples and trade-off discussions."""


class WriterAgent(BaseAgent):
    """Produces the final polished research report with citations."""

    name = "writer"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.final_answer`."""
        logger.info("Writer synthesizing final report for: %s", state.request.query)

        sources_text = "\n".join(
            f"[Source {idx}] {s.title}"
            + (f" ({s.url})" if s.url else "")
            + f"\nSnippet: {s.snippet}"
            for idx, s in enumerate(state.sources, 1)
        )

        user_prompt = (
            f"Target Research Query: {state.request.query}\n"
            f"Target Audience: {state.request.audience}\n\n"
            f"=== RESEARCH NOTES ===\n{state.research_notes or 'None'}\n\n"
            f"=== ANALYSIS NOTES ===\n{state.analysis_notes or 'None'}\n\n"
            f"=== SOURCE DOCUMENTS ===\n{sources_text if sources_text else 'None'}\n\n"
            "Please produce the complete, citation-rich research report."
        )

        response = self.llm_client.complete(
            system_prompt=WRITER_SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )

        state.final_answer = response.content
        state.agent_results.append(
            AgentResult(
                agent=AgentName.WRITER,
                content=response.content,
                metadata={
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "cost_usd": response.cost_usd,
                },
            )
        )
        state.add_trace_event(
            "writer_completed",
            {
                "cost_usd": response.cost_usd,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
            },
        )
        return state
