"""Tracing hooks and observability integrations for LangSmith, Langfuse, and local trace spans."""

import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from time import perf_counter
from typing import Any

from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.state import ResearchState

logger = logging.getLogger(__name__)


def setup_tracing(settings: Settings | None = None) -> None:
    """Initialize LangSmith / Langfuse tracing environment variables if configured."""
    s = settings or get_settings()
    if s.langsmith_api_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = s.langsmith_api_key
        os.environ["LANGCHAIN_PROJECT"] = s.langsmith_project
        logger.info("LangSmith tracing enabled for project: %s", s.langsmith_project)


@contextmanager
def trace_span(name: str, attributes: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    """Context manager for measuring execution span duration and capturing metadata."""
    started = perf_counter()
    span: dict[str, Any] = {
        "name": name,
        "attributes": attributes or {},
        "duration_seconds": None,
        "status": "running",
    }
    try:
        yield span
        span["status"] = "success"
    except Exception as exc:
        span["status"] = "error"
        span["error"] = str(exc)
        raise
    finally:
        span["duration_seconds"] = perf_counter() - started


def export_trace_summary(state: ResearchState) -> dict[str, Any]:
    """Generate a structured summary of the workflow execution trace."""
    total_cost = sum(float(res.metadata.get("cost_usd", 0.0) or 0.0) for res in state.agent_results)
    total_in_tokens = sum(
        int(res.metadata.get("input_tokens", 0) or 0) for res in state.agent_results
    )
    total_out_tokens = sum(
        int(res.metadata.get("output_tokens", 0) or 0) for res in state.agent_results
    )

    return {
        "query": state.request.query,
        "iterations": state.iteration,
        "route_history": state.route_history,
        "sources_count": len(state.sources),
        "total_cost_usd": total_cost,
        "total_input_tokens": total_in_tokens,
        "total_output_tokens": total_out_tokens,
        "events_count": len(state.trace),
        "events": state.trace,
    }
