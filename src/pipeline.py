"""Core orchestration logic for the newsletter workflow."""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from . import utils


@dataclass
class Source:
    id: str
    name: str
    rss: Optional[str]
    sitemap: Optional[str]
    homepage: Optional[str]
    trust_score: float
    sector_impact: float
    group: str  # "turkish" or "global"


@dataclass
class RawItem:
    source: Source
    url: str
    fetched_at: dt.datetime
    payload: Dict[str, Any]


@dataclass
class NormalizedArticle:
    source: Source
    url: str
    title: str
    text: str
    byline: Optional[str]
    published_at: dt.datetime
    summary: Optional[str] = None
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EnrichedArticle(NormalizedArticle):
    entities: List[str] = field(default_factory=list)
    countries: List[str] = field(default_factory=list)
    themes: List[str] = field(default_factory=list)
    embedding_id: Optional[str] = None
    novelty: float = 0.5
    diversity: float = 0.5
    tr_relevance: float = 0.5
    policy: Dict[str, Any] = field(default_factory=dict)
    cluster_id: Optional[str] = None
    score: float = 0.0


class NewsletterPipeline:
    """Execute the end-to-end workflow from collection to publishing."""

    def __init__(
        self,
        toolkit: utils.MCPToolkit,
        resources: Mapping[str, Any],
        prompts: Mapping[str, Any],
        cache_root: Path = utils.CACHE_ROOT,
    ) -> None:
        self.toolkit = toolkit
        self.resources = resources
        self.prompts = prompts
        self.cache_root = cache_root
        self.log = logging.getLogger(self.__class__.__name__)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(
        self,
        days: int,
        max_items: int,
        output_dir: Path,
        turkish_weight: float = 0.6,
        dry_run: bool = False,
    ) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        now = dt.datetime.now(dt.timezone.utc)
        collected = self.collect_sources(days)
        normalized = self.normalize(collected, days)
        enriched = self.enrich(normalized, now, turkish_weight)
        ranked = self.dedupe_and_rank(enriched, max_items, turkish_weight)
        policy_checked = self.apply_policy_guard(ranked)
        markdown = self.render(policy_checked, now.date())
        output_path = output_dir / f"newsletter_{now.date()}.md"
        output_path.write_text(markdown, encoding="utf-8")
        self.log.info("Newsletter written to %s", output_path)
        if not dry_run:
            self.publish(markdown)
        else:
            self.log.info("Dry run enabled, skipping publisher calls")
        return output_path

    # ------------------------------------------------------------------
    # Step: Collect
    # ------------------------------------------------------------------
    def collect_sources(self, days: int) -> List[RawItem]:
        sources = self._load_sources()
        items: List[RawItem] = []
        for source in sources:
            try:
                if source.rss:
                    payload = {"rss": source.rss, "days": days}
                    response = self._call_cached("rss_server", "fetch_since", payload)
                    urls = self._extract_urls(response)
                else:
                    sitemap_payload = {"url": source.sitemap or source.homepage, "days": days}
                    sitemap_data = self._call_cached("web_fetch", "sitemap_or_index", sitemap_payload)
                    urls = self._urls_from_sitemap(sitemap_data)
                    # fetch article bodies for caching
                    for url in urls:
                        self._call_cached("web_fetch", "get", {"url": url})
                for url in urls:
                    items.append(
                        RawItem(
                            source=source,
                            url=url,
                            fetched_at=dt.datetime.now(dt.timezone.utc),
                            payload={"url": url},
                        )
                    )
            except utils.ToolUnavailableError as exc:
                self.log.warning("Skipping source %s: %s", source.name, exc)
            except utils.ToolExecutionError as exc:
                self.log.error("Failed to collect from %s: %s", source.name, exc)
        self.log.info("Collected %s raw items", len(items))
        return items

    # ------------------------------------------------------------------
    # Step: Normalize
    # ------------------------------------------------------------------
    def normalize(self, items: Iterable[RawItem], days: int) -> List[NormalizedArticle]:
        normalized: List[NormalizedArticle] = []
        for item in items:
            cached = self._load_normalized(item.url)
            if cached:
                article = cached
            else:
                try:
                    response = self._call_tool("readability", "extract", {"url": item.url})
                except (utils.ToolUnavailableError, utils.ToolExecutionError) as exc:
                    self.log.debug("Readability failed for %s: %s", item.url, exc)
                    continue
                if not response:
                    continue
                article = self._to_normalized(item, response)
                if article:
                    self._save_normalized(article)
            if not article:
                continue
            if not utils.within_days(article.published_at, days):
                continue
            if not article.text:
                continue
            normalized.append(article)
        self.log.info("Normalized %s articles", len(normalized))
        return normalized

    # ------------------------------------------------------------------
    # Step: Enrich
    # ------------------------------------------------------------------
    def enrich(self, articles: Iterable[NormalizedArticle], now: dt.datetime, turkish_weight: float) -> List[EnrichedArticle]:
        enriched: List[EnrichedArticle] = []
        for article in articles:
            entity_data: Dict[str, Any] = {}
            try:
                entity_data = self._call_tool("ner_tag", "run", {"text": article.text}) or {}
            except (utils.ToolUnavailableError, utils.ToolExecutionError) as exc:
                self.log.debug("NER tagging failed for %s: %s", article.url, exc)
            embedding_id = None
            try:
                embed_payload = {
                    "id": utils.hash_dict({"url": article.url}),
                    "text": article.text,
                    "metadata": {"source": article.source.id},
                }
                embed_response = self._call_tool("embed_store", "upsert", embed_payload)
                if isinstance(embed_response, Mapping):
                    embedding_id = str(embed_response.get("id"))
            except (utils.ToolUnavailableError, utils.ToolExecutionError) as exc:
                self.log.debug("Embedding upsert failed for %s: %s", article.url, exc)
            recency_hours = max((now - article.published_at).total_seconds() / 3600.0, 1.0)
            novelty = min(1.0, 72.0 / recency_hours)
            tr_relevance = turkish_weight if article.source.group == "turkish" else 1 - turkish_weight
            enriched.append(
                EnrichedArticle(
                    **article.__dict__,
                    entities=list(entity_data.get("entities", [])),
                    countries=list(entity_data.get("countries", [])),
                    themes=list(entity_data.get("themes", [])),
                    embedding_id=embedding_id,
                    novelty=novelty,
                    tr_relevance=tr_relevance,
                )
            )
        return enriched

    # ------------------------------------------------------------------
    # Step: Deduplicate and rank
    # ------------------------------------------------------------------
    def dedupe_and_rank(
        self,
        articles: Iterable[EnrichedArticle],
        max_items: int,
        turkish_weight: float,
    ) -> List[EnrichedArticle]:
        articles = list(articles)
        if not articles:
            return []
        payload = {
            "items": [
                {
                    "id": utils.hash_dict({"url": article.url}),
                    "url": article.url,
                    "title": article.title,
                    "text": article.text,
                    "source": article.source.id,
                }
                for article in articles
            ],
            "max_items": max_items,
        }
        try:
            response = self._call_tool("dedupe_rank", "cluster", payload)
        except (utils.ToolUnavailableError, utils.ToolExecutionError) as exc:
            self.log.debug("Dedupe service unavailable, falling back: %s", exc)
            response = None
        clusters = response.get("clusters") if isinstance(response, Mapping) else None
        if clusters:
            ranked: List[EnrichedArticle] = []
            for cluster in clusters:
                article_id = cluster.get("representative")
                score = float(cluster.get("score", 0.0))
                target = next((a for a in articles if utils.hash_dict({"url": a.url}) == article_id), None)
                if not target:
                    continue
                target.cluster_id = cluster.get("id")
                target.diversity = float(cluster.get("diversity", 0.5))
                target.score = score
                ranked.append(target)
            ranked.sort(key=lambda a: a.score, reverse=True)
            return ranked[:max_items]
        # fallback scoring
        weights = self.resources.get("weights", {})
        for article in articles:
            article.diversity = 0.8  # optimistic default
            article.score = utils.compute_score(
                source_trust=article.source.trust_score,
                novelty=article.novelty,
                sector_impact=article.source.sector_impact,
                tr_relevance=article.tr_relevance,
                diversity=article.diversity,
                weights=weights,
            )
        articles.sort(key=lambda a: a.score, reverse=True)
        return articles[:max_items]

    # ------------------------------------------------------------------
    # Step: Policy guard
    # ------------------------------------------------------------------
    def apply_policy_guard(self, articles: Iterable[EnrichedArticle]) -> List[EnrichedArticle]:
        result: List[EnrichedArticle] = []
        for article in articles:
            try:
                response = self._call_tool(
                    "policy_guard",
                    "check",
                    {"url": article.url, "text": article.text[:5000]},
                )
                if isinstance(response, Mapping):
                    article.policy = {
                        "risk_tags": response.get("risk_tags", []),
                        "reg_refs": response.get("reg_refs", []),
                    }
            except (utils.ToolUnavailableError, utils.ToolExecutionError) as exc:
                self.log.debug("Policy guard unavailable for %s: %s", article.url, exc)
            result.append(article)
        return result

    # ------------------------------------------------------------------
    # Step: Render
    # ------------------------------------------------------------------
    def render(self, articles: Iterable[EnrichedArticle], date: dt.date) -> str:
        from . import render as renderer

        articles = list(articles)
        rendered_items = [self._to_render_item(article) for article in articles]
        sections = renderer.bucket_sections(rendered_items)
        return renderer.render_markdown(date, sections, self.prompts)

    # ------------------------------------------------------------------
    # Step: Publish
    # ------------------------------------------------------------------
    def publish(self, markdown: str) -> None:
        try:
            publisher_cfg = self.toolkit._configs.get("publisher")
        except AttributeError:
            publisher_cfg = None
        if not publisher_cfg:
            self.log.info("Publisher not configured; skipping")
            return
        try:
            self._call_tool("publisher", "broadcast", {"content": markdown})
            self.log.info("Publish job triggered")
        except (utils.ToolUnavailableError, utils.ToolExecutionError) as exc:
            self.log.error("Publisher call failed: %s", exc)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _load_sources(self) -> List[Source]:
        sources: List[Source] = []
        for group_name in ("turkish_sources", "global_sources"):
            group = self.resources.get(group_name, [])
            for data in group:
                sources.append(
                    Source(
                        id=data.get("id"),
                        name=data.get("name"),
                        rss=data.get("rss"),
                        sitemap=data.get("sitemap"),
                        homepage=data.get("homepage"),
                        trust_score=float(data.get("trust_score", 0.5)),
                        sector_impact=float(data.get("sector_impact", 0.5)),
                        group="turkish" if group_name == "turkish_sources" else "global",
                    )
                )
        return sources

    def _extract_urls(self, response: Any) -> List[str]:
        urls: List[str] = []
        if isinstance(response, Mapping):
            items = response.get("items") or response.get("entries") or []
        elif isinstance(response, list):
            items = response
        else:
            items = []
        for item in items:
            if isinstance(item, str):
                urls.append(item)
            elif isinstance(item, Mapping):
                if item.get("url"):
                    urls.append(item["url"])
                elif item.get("link"):
                    urls.append(item["link"])
        return list(dict.fromkeys(urls))

    def _urls_from_sitemap(self, data: Any) -> List[str]:
        if isinstance(data, Mapping):
            urls = data.get("urls") or data.get("items") or []
        elif isinstance(data, list):
            urls = data
        else:
            return []
        result = []
        for item in urls:
            if isinstance(item, Mapping):
                url = item.get("loc") or item.get("url")
            else:
                url = item
            if isinstance(url, str):
                result.append(url)
        return result

    def _call_tool(self, tool: str, method: str, payload: Mapping[str, Any]) -> Any:
        return self.toolkit.call(tool, method, payload)

    def _call_cached(self, tool: str, method: str, payload: Mapping[str, Any]) -> Any:
        key = utils.hash_dict({"tool": tool, "method": method, "payload": payload})
        cache_file = utils.cache_path(utils.RAW_CACHE, key)
        cached = utils.load_cached_json(cache_file)
        if cached is not None:
            return cached
        try:
            response = self._call_tool(tool, method, payload)
        except utils.ToolUnavailableError:
            raise
        except utils.ToolExecutionError as exc:
            self.log.debug("Tool call failed (non-cached) %s.%s: %s", tool, method, exc)
            raise
        utils.save_cached_json(cache_file, response)
        return response

    def _to_normalized(self, item: RawItem, response: Mapping[str, Any]) -> Optional[NormalizedArticle]:
        published = utils.parse_date(response.get("date") or response.get("published_at"))
        if not published:
            return None
        text = response.get("text") or response.get("content")
        if not text:
            return None
        title = response.get("title") or response.get("headline") or item.url
        byline = response.get("byline") or response.get("author")
        summary = response.get("summary")
        return NormalizedArticle(
            source=item.source,
            url=item.url,
            title=title.strip(),
            text=text.strip(),
            byline=byline.strip() if isinstance(byline, str) else None,
            published_at=published,
            summary=summary.strip() if isinstance(summary, str) else None,
            extras={k: v for k, v in response.items() if k not in {"title", "text", "content", "byline", "author", "date", "published_at", "summary"}},
        )

    def _save_normalized(self, article: NormalizedArticle) -> None:
        key = utils.hash_dict({"url": article.url})
        path = utils.cache_path(utils.NORMALIZED_CACHE, key)
        data = {
            "source": {
                "id": article.source.id,
                "name": article.source.name,
                "rss": article.source.rss,
                "sitemap": article.source.sitemap,
                "homepage": article.source.homepage,
                "trust_score": article.source.trust_score,
                "sector_impact": article.source.sector_impact,
                "group": article.source.group,
            },
            "url": article.url,
            "title": article.title,
            "text": article.text,
            "byline": article.byline,
            "published_at": utils.isoformat(article.published_at),
            "summary": article.summary,
            "extras": article.extras,
        }
        utils.save_cached_json(path, data)

    def _load_normalized(self, url: str) -> Optional[NormalizedArticle]:
        key = utils.hash_dict({"url": url})
        path = utils.cache_path(utils.NORMALIZED_CACHE, key)
        data = utils.load_cached_json(path)
        if not data:
            return None
        source_data = data.get("source", {})
        source = Source(
            id=source_data.get("id"),
            name=source_data.get("name"),
            rss=source_data.get("rss"),
            sitemap=source_data.get("sitemap"),
            homepage=source_data.get("homepage"),
            trust_score=float(source_data.get("trust_score", 0.5)),
            sector_impact=float(source_data.get("sector_impact", 0.5)),
            group=source_data.get("group", "global"),
        )
        published = utils.parse_date(data.get("published_at"))
        if not published:
            return None
        return NormalizedArticle(
            source=source,
            url=data.get("url"),
            title=data.get("title"),
            text=data.get("text"),
            byline=data.get("byline"),
            published_at=published,
            summary=data.get("summary"),
            extras=data.get("extras", {}),
        )

    def _to_render_item(self, article: EnrichedArticle) -> "renderer.RenderedItem":
        from . import render as renderer

        context_text = article.summary or utils.summarize_text(article.text, sentences=3)
        bullets = utils.bullets_from_text(article.text, count=3)
        risk_tags = article.policy.get("risk_tags", []) if article.policy else []
        reg_refs = article.policy.get("reg_refs", []) if article.policy else []
        return renderer.RenderedItem(
            headline=article.title,
            context=context_text,
            bullets=bullets,
            url=article.url,
            source_name=article.source.name,
            published_at=article.published_at,
            group=article.source.group,
            risk_tags=risk_tags,
            reg_refs=reg_refs,
            entities=article.entities,
            themes=article.themes,
            score=article.score,
            content=article.text,
        )


__all__ = ["NewsletterPipeline"]
