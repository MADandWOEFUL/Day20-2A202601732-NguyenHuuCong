"""Search client abstraction for ResearcherAgent."""

import json
import logging
import ssl
import urllib.request
from pathlib import Path

import certifi

from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.schemas import SourceDocument

logger = logging.getLogger(__name__)


class SearchClient:
    """Search client supporting Tavily API, local offline corpus search, and mock fallback."""

    def __init__(
        self,
        settings: Settings | None = None,
        corpus_dir: Path | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.corpus_dir = corpus_dir or Path("ai_agent_offline_research_corpus_v2/topics")

    def _search_tavily(self, query: str, max_results: int) -> list[SourceDocument]:
        """Search via Tavily API."""
        if not self.settings.tavily_api_key:
            return []

        url = "https://api.tavily.com/search"
        payload = {
            "api_key": self.settings.tavily_api_key,
            "query": query,
            "max_results": max_results,
            "search_depth": "basic",
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json", "User-Agent": "MultiAgentResearchLab/1.0"},
        )
        ctx = ssl.create_default_context(cafile=certifi.where())

        with urllib.request.urlopen(req, context=ctx, timeout=10) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            raw_results = body.get("results", [])

        docs: list[SourceDocument] = []
        for item in raw_results:
            title = item.get("title", "Untitled Source")
            snippet = item.get("content", "")
            source_url = item.get("url")
            score = item.get("score")
            docs.append(
                SourceDocument(
                    title=title,
                    url=source_url,
                    snippet=snippet,
                    metadata={"score": score, "source": "tavily"},
                )
            )
        return docs

    def _search_corpus(self, query: str, max_results: int) -> list[SourceDocument]:
        """Search in the local JSON corpus files."""
        if not self.corpus_dir.exists():
            return []

        query_terms = [t.lower() for t in query.split() if len(t) > 2]
        scored_docs: list[tuple[float, SourceDocument]] = []

        for json_file in self.corpus_dir.glob("*.json"):
            try:
                data = json.loads(json_file.read_text(encoding="utf-8"))
            except Exception:
                continue

            topic_info = data.get("topic", {})
            topic_name = topic_info.get("name", "")

            # Check knowledge articles
            for article in data.get("knowledge_base", {}).get("knowledge_articles", []):
                title = article.get("title", "")
                content = article.get("content", "")
                article_id = article.get("article_id", "A0")

                score = sum(
                    1.5 for t in query_terms if t in title.lower() or t in topic_name.lower()
                ) + sum(0.5 for t in query_terms if t in content.lower())

                if score > 0:
                    scored_docs.append(
                        (
                            score,
                            SourceDocument(
                                title=f"[{article_id}] {topic_name}: {title}",
                                url=f"offline://corpus/{json_file.stem}#{article_id}",
                                snippet=content[:500] + "...",
                                metadata={
                                    "article_id": article_id,
                                    "topic": topic_name,
                                    "score": score,
                                },
                            ),
                        )
                    )

            # Check public and synthetic source documents in corpus
            for src in data.get("source_documents", {}).get("public_reference_summaries", []):
                title = src.get("title", "")
                content = src.get("summary", "")
                src_id = src.get("source_id", "src")
                score = sum(1.0 for t in query_terms if t in title.lower() or t in content.lower())
                if score > 0:
                    scored_docs.append(
                        (
                            score,
                            SourceDocument(
                                title=f"[{src_id}] {title}",
                                url=src.get("url") or f"offline://corpus/{json_file.stem}#{src_id}",
                                snippet=content[:500] + "...",
                                metadata={"source_id": src_id, "score": score},
                            ),
                        )
                    )

        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored_docs[:max_results]]

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        """Search for documents relevant to a query.

        Tries Tavily API first, falling back to local corpus or synthesized mock documents.
        """
        # 1. Try Tavily search if configured
        try:
            results = self._search_tavily(query, max_results)
            if results:
                return results
        except Exception as exc:
            logger.warning("Tavily search failed, falling back to local corpus: %s", exc)

        # 2. Try Local corpus search
        try:
            corpus_results = self._search_corpus(query, max_results)
            if corpus_results:
                return corpus_results
        except Exception as exc:
            logger.warning("Local corpus search failed: %s", exc)

        # 3. Fallback mock document
        return [
            SourceDocument(
                title=f"Research notes on {query}",
                url="https://example.org/research/knowledge-base",
                snippet=(
                    f"Key architectural principles regarding: {query}. "
                    "Includes comparative analysis of performance, trade-offs, and state."
                ),
                metadata={"source": "mock_fallback"},
            )
        ]
