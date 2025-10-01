"""Rendering utilities for newsletter assembly."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Dict, List, Mapping


@dataclass
class RenderedItem:
    headline: str
    context: str
    bullets: List[str]
    url: str
    source_name: str
    published_at: dt.datetime
    group: str
    risk_tags: List[str]
    reg_refs: List[str]
    entities: List[str]
    themes: List[str]
    score: float
    content: str


SectionMap = Dict[str, List[RenderedItem]]


def bucket_sections(items: List[RenderedItem]) -> Dict[str, List[RenderedItem]]:
    sections: Dict[str, List[RenderedItem]] = {
        "tldr": [],
        "turkey": [],
        "global": [],
        "deep_dives": [],
        "regulation": [],
        "infrastructure": [],
    }
    # Sort by score once for consistent ordering
    sorted_items = sorted(items, key=lambda itm: itm.score, reverse=True)
    sections["tldr"] = sorted_items[:5]
    for item in sorted_items:
        target_bucket = "turkey" if item.group == "turkish" else "global"
        sections[target_bucket].append(item)
        if item.risk_tags or item.reg_refs:
            sections["regulation"].append(item)
        if any(theme.lower() in {"infrastructure", "platform", "ops"} for theme in item.themes):
            sections["infrastructure"].append(item)
        elif any(entity.lower() in {"agent", "agentic ai"} for entity in item.entities):
            sections["infrastructure"].append(item)
        if len(item.content) > 1200 or "analysis" in item.themes:
            sections["deep_dives"].append(item)
    # Fallbacks to ensure sections have content
    if not sections["deep_dives"]:
        sections["deep_dives"] = sorted_items[:2]
    if not sections["regulation"]:
        sections["regulation"] = sorted_items[:2]
    if not sections["infrastructure"]:
        sections["infrastructure"] = sorted_items[:3]
    return sections


def render_markdown(date: dt.date, sections: Dict[str, List[RenderedItem]], prompts: Mapping[str, Mapping]) -> str:
    summary_template = prompts.get("summary_template", "{headline}\n{context}")
    header = prompts.get("render", {}).get("header", "").format(date=date)
    footer = prompts.get("render", {}).get("footer", "")
    parts: List[str] = [header.strip(), ""]
    sections_config = prompts.get("sections", {})

    def format_item(item: RenderedItem) -> str:
        body = summary_template.format(
            headline=f"[{item.headline}]({item.url}) — {item.source_name}",
            context=item.context,
            bullet1=item.bullets[0] if item.bullets else "",
            bullet2=item.bullets[1] if len(item.bullets) > 1 else "",
            bullet3=item.bullets[2] if len(item.bullets) > 2 else "",
        )
        if item.risk_tags:
            body += f"\n_Risk tags: {', '.join(item.risk_tags)}_"
        if item.reg_refs:
            body += f"\n_Regülasyon referansları: {', '.join(item.reg_refs)}_"
        body += f"\n_Published: {item.published_at.date()}_"
        return body

    def section_block(key: str, items_list: List[RenderedItem]) -> str:
        config = sections_config.get(key, {})
        title = config.get("title", key.title())
        instructions = config.get("instructions", "")
        block: List[str] = [f"## {title}"]
        if instructions:
            block.append(f"<!-- {instructions.strip()} -->")
        if key == "tldr":
            bullets = [f"- [{itm.headline}]({itm.url}) — {itm.context.split('. ')[0]}" for itm in items_list]
            block.extend(bullets)
        else:
            block.extend(format_item(itm) for itm in items_list)
        return "\n".join(block)

    for key in ("tldr", "turkey", "global", "deep_dives", "regulation", "infrastructure"):
        items_list = sections.get(key, [])
        if not items_list:
            continue
        parts.append(section_block(key, items_list))
        parts.append("")
    parts.append(footer.strip())
    return "\n".join(part for part in parts if part)


__all__ = ["RenderedItem", "bucket_sections", "render_markdown"]
