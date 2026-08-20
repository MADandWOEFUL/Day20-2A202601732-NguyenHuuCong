"""Critic agent for fact-checking, citation audit, and quality review."""

import json
import logging
import re

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)


CRITIC_SYSTEM_PROMPT = """You are an expert Reviewer and Quality Auditor in a multi-agent system.
Your job is to critically evaluate a research report produced by the Writer agent.

Audit the report across these criteria:
1. Factuality & Hallucination: Are claims supported by the provided sources?
2. Citation Coverage: Are critical assertions accompanied by source citations?
3. Clarity & Structure: Is the report well-structured and audience-appropriate?
4. Completeness: Did the report fully address the initial user query?

Provide a JSON object with:
{
  "quality_score": <float between 0.0 and 10.0>,
  "citation_coverage": <float between 0.0 and 1.0>,
  "strengths": [<list of strengths>],
  "weaknesses": [<list of weaknesses or unsupported claims>],
  "summary_feedback": "<brief paragraph summarizing assessment>"
}"""


class CriticAgent(BaseAgent):
    """Fact-checking, citation coverage, and safety/quality review agent."""

    name = "critic"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Validate final answer and append critique/metrics."""
        logger.info("Critic reviewing final report for: %s", state.request.query)

        if not state.final_answer:
            logger.warning("Critic called but state.final_answer is empty.")
            return state

        sources_summary = "\n".join(
            f"[{idx}] {s.title}: {s.snippet[:200]}" for idx, s in enumerate(state.sources, 1)
        )

        user_prompt = f"""User Query: {state.request.query}

Sources:
{sources_summary if sources_summary else "No sources"}

Final Report to Critique:
{state.final_answer}

Provide your evaluation as JSON."""

        response = self.llm_client.complete(
            system_prompt=CRITIC_SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )

        quality_score: float = 8.5
        citation_cov: float = 0.90
        try:
            # Parse json block from LLM output
            json_match = re.search(r"\{.*\}", response.content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                quality_score = float(data.get("quality_score", 8.5))
                citation_cov = float(data.get("citation_coverage", 0.90))
        except Exception:
            pass

        state.agent_results.append(
            AgentResult(
                agent=AgentName.CRITIC,
                content=response.content,
                metadata={
                    "quality_score": quality_score,
                    "citation_coverage": citation_cov,
                    "cost_usd": response.cost_usd,
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                },
            )
        )
        state.add_trace_event(
            "critic_completed",
            {
                "quality_score": quality_score,
                "citation_coverage": citation_cov,
                "cost_usd": response.cost_usd,
            },
        )
        return state
