"""Researcher agent for gathering and summarizing relevant information."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient

logger = logging.getLogger(__name__)


RESEARCHER_SYSTEM_PROMPT = """You are an expert Research Agent in a multi-agent system.
Your job is to review retrieved source documents and produce dense research notes.
Focus on:
1. Identifying key definitions, core concepts, and state-of-the-art developments.
2. Citing each piece of evidence with its source title/ID.
3. Highlighting empirical results, quantitative metrics, architectures, or protocols.
4. Extracting controversies, limitations, or differing perspectives.

Keep your notes structured with clear bullet points and explicit citations."""


class ResearcherAgent(BaseAgent):
    """Collects sources and creates concise, citation-backed research notes."""

    name = "researcher"

    def __init__(
        self,
        search_client: SearchClient | None = None,
        llm_client: LLMClient | None = None,
    ) -> None:
        self.search_client = search_client or SearchClient()
        self.llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.sources` and `state.research_notes`."""
        logger.info("Researcher running for query: %s", state.request.query)

        # 1. Search for sources
        docs = self.search_client.search(
            query=state.request.query,
            max_results=state.request.max_sources,
        )
        state.sources = docs

        # 2. Format sources for LLM context
        sources_text = ""
        for idx, doc in enumerate(docs, 1):
            url_str = f" ({doc.url})" if doc.url else ""
            sources_text += f"\n[Source {idx}] {doc.title}{url_str}\n{doc.snippet}\n"

        user_prompt = (
            f"Target Query: {state.request.query}\n"
            f"Audience: {state.request.audience}\n\n"
            f"Retrieved Sources ({len(docs)} found):\n"
            + (
                sources_text
                if docs
                else "No external sources retrieved. Use foundational knowledge.\n"
            )
            + "\nPlease produce comprehensive, citation-rich research notes addressing the query."
        )

        # 3. Generate research notes via LLM
        response = self.llm_client.complete(
            system_prompt=RESEARCHER_SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )

        state.research_notes = response.content
        state.agent_results.append(
            AgentResult(
                agent=AgentName.RESEARCHER,
                content=response.content,
                metadata={
                    "sources_count": len(docs),
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "cost_usd": response.cost_usd,
                },
            )
        )
        state.add_trace_event(
            "researcher_completed",
            {
                "sources_retrieved": len(docs),
                "cost_usd": response.cost_usd,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
            },
        )
        return state
