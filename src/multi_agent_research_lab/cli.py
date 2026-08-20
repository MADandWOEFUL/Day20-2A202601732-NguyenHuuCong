"""Command-line entrypoint for the lab starter."""

from pathlib import Path
from time import perf_counter
from typing import Annotated

import typer
import yaml  # type: ignore[import-untyped]
from pydantic import ValidationError
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import (
    AgentName,
    AgentResult,
    BenchmarkMetrics,
    ResearchQuery,
)
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import run_benchmark
from multi_agent_research_lab.evaluation.report import render_markdown_report
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.observability.tracing import setup_tracing
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient
from multi_agent_research_lab.services.storage import LocalArtifactStore

app = typer.Typer(help="Multi-Agent Research Lab CLI")
console = Console()


def _init() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    setup_tracing(settings)


def _parse_query(query: str) -> ResearchQuery:
    try:
        return ResearchQuery(query=query)
    except ValidationError as exc:
        console.print(
            Panel.fit(
                f"Invalid query: {exc.errors()[0]['msg']}",
                title="Input Error",
                style="red",
            )
        )
        raise typer.Exit(code=1) from exc


def _run_single_agent_baseline(query_str: str) -> ResearchState:
    """Execute single-agent baseline pipeline (single direct prompt with search context)."""
    settings = get_settings()
    search_client = SearchClient(settings=settings)
    llm_client = LLMClient(settings=settings)

    request = ResearchQuery(query=query_str)
    state = ResearchState(request=request)

    # 1. Search
    docs = search_client.search(query=query_str, max_results=request.max_sources)
    state.sources = docs

    sources_context = "\n".join(f"[{idx}] {d.title}: {d.snippet}" for idx, d in enumerate(docs, 1))

    # 2. Single LLM call
    system_prompt = (
        "You are a generalist AI research assistant. Given a query and retrieved sources, "
        "write a concise, factual summary addressing the query with citations."
    )
    user_prompt = (
        f"Query: {query_str}\n\nSources:\n{sources_context}\n\nPlease provide your answer."
    )

    resp = llm_client.complete(system_prompt=system_prompt, user_prompt=user_prompt)
    state.final_answer = resp.content
    state.record_route("single_agent")
    state.agent_results.append(
        AgentResult(
            agent=AgentName.SUPERVISOR,
            content=resp.content,
            metadata={
                "cost_usd": resp.cost_usd,
                "input_tokens": resp.input_tokens,
                "output_tokens": resp.output_tokens,
            },
        )
    )
    return state


@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run a single-agent baseline."""
    _init()
    request = _parse_query(query)
    console.print(f"[bold cyan]Running Single-Agent Baseline for:[/bold cyan] {request.query}\n")

    started = perf_counter()
    state = _run_single_agent_baseline(request.query)
    duration = perf_counter() - started

    cost = sum(float(r.metadata.get("cost_usd", 0) or 0) for r in state.agent_results)
    console.print(Panel(Markdown(state.final_answer or ""), title="Single-Agent Baseline Output"))
    console.print(
        f"[green]Done in {duration:.2f}s | Cost: ${cost:.5f} | Sources: "
        f"{len(state.sources)}[/green]"
    )


@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run the multi-agent workflow (Supervisor + Researcher + Analyst + Writer + Critic)."""
    _init()
    request = _parse_query(query)
    console.print(f"[bold cyan]Running Multi-Agent Workflow for:[/bold cyan] {request.query}\n")

    state = ResearchState(request=request)
    workflow = MultiAgentWorkflow()

    started = perf_counter()
    result = workflow.run(state)
    duration = perf_counter() - started

    # Print route progression
    console.print(f"[bold yellow]Route Sequence:[/bold yellow] {' -> '.join(result.route_history)}")
    console.print(f"[bold yellow]Retrieved Sources:[/bold yellow] {len(result.sources)} documents")

    # Print final answer
    if result.final_answer:
        console.print(
            Panel(Markdown(result.final_answer), title="Multi-Agent Final Report", expand=True)
        )

    # Print critic review if present
    critic_res = [r for r in result.agent_results if r.agent == AgentName.CRITIC]
    if critic_res:
        meta = critic_res[-1].metadata
        score = meta.get("quality_score", "N/A")
        cov = meta.get("citation_coverage", "N/A")
        review_text = (
            f"[bold]Quality Score:[/bold] {score}/10 | "
            f"[bold]Citation Coverage:[/bold] {cov}\n\n{critic_res[-1].content}"
        )
        console.print(
            Panel(
                review_text,
                title="Critic Review & Audit",
                style="cyan",
            )
        )

    total_cost = sum(float(r.metadata.get("cost_usd", 0) or 0) for r in result.agent_results)
    console.print(
        f"[bold green]Multi-Agent Completed in {duration:.2f}s | "
        f"Total Cost: ${total_cost:.5f} | Iterations: {result.iteration}[/bold green]"
    )


@app.command()
def benchmark(
    config_path: Annotated[
        str, typer.Option("--config", "-c", help="Config file path")
    ] = "configs/lab_default.yaml",
) -> None:
    """Run benchmark comparing single-agent baseline and multi-agent workflow."""
    _init()
    cfg_file = Path(config_path)
    queries = [
        "Research GraphRAG state-of-the-art and write a 500-word summary",
        "Compare single-agent and multi-agent workflows for customer support",
        "Summarize production guardrails for LLM agents",
    ]
    if cfg_file.exists():
        try:
            with open(cfg_file, encoding="utf-8") as f:
                data = yaml.safe_load(f)
                queries = data.get("benchmark", {}).get("queries", queries)
        except Exception:
            pass

    console.print(f"[bold green]Starting Benchmark on {len(queries)} queries...[/bold green]\n")
    all_metrics: list[BenchmarkMetrics] = []
    workflow = MultiAgentWorkflow()

    for idx, q in enumerate(queries, 1):
        console.print(f"[bold cyan]Test Query {idx}/{len(queries)}:[/bold cyan] {q}")

        # 1. Benchmark Baseline
        console.print("  -> Running Baseline...")
        _, b_metrics = run_benchmark(
            run_name=f"Baseline [Q{idx}]",
            query=q,
            runner=_run_single_agent_baseline,
        )
        all_metrics.append(b_metrics)

        # 2. Benchmark Multi-Agent
        console.print("  -> Running Multi-Agent Workflow...")
        _, m_metrics = run_benchmark(
            run_name=f"Multi-Agent [Q{idx}]",
            query=q,
            runner=lambda query_text: workflow.run(
                ResearchState(request=ResearchQuery(query=query_text))
            ),
        )
        all_metrics.append(m_metrics)
        console.print()

    # Render and save report
    report_md = render_markdown_report(all_metrics)
    store = LocalArtifactStore(root=Path("reports"))
    report_path = store.write_text("benchmark_report.md", report_md)
    console.print(f"[bold green]Benchmark report written to:[/bold green] {report_path}\n")

    # Display table in console
    table = Table(title="Benchmark Comparison")
    table.add_column("Run", style="cyan")
    table.add_column("Latency (s)", justify="right")
    table.add_column("Cost (USD)", justify="right")
    table.add_column("Quality", justify="right")
    table.add_column("Citation Cov.", justify="right")
    table.add_column("Failure Rate", justify="right")

    for m in all_metrics:
        cost_str = f"${m.estimated_cost_usd:.4f}" if m.estimated_cost_usd else "N/A"
        q_str = f"{m.quality_score:.1f}" if m.quality_score is not None else "N/A"
        cit_str = f"{m.citation_coverage:.0%}" if m.citation_coverage is not None else "N/A"
        fail_str = f"{m.failure_rate:.0%}" if m.failure_rate is not None else "0%"
        table.add_row(m.run_name, f"{m.latency_seconds:.2f}", cost_str, q_str, cit_str, fail_str)

    console.print(table)


if __name__ == "__main__":
    app()
