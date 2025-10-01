"""CLI entry point for the AI × Banking newsletter builder."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .pipeline import NewsletterPipeline
from .utils import MCPToolkit, load_yaml, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compile AI × Banking weekly newsletter")
    parser.add_argument("--days", type=int, default=7, help="Look-back window in days")
    parser.add_argument("--max-items", type=int, default=15, help="Maximum number of final stories")
    parser.add_argument("--out", type=Path, default=Path("out"), help="Output directory for markdown")
    parser.add_argument("--turkish-weight", type=float, default=0.6, help="Weight for Türkiye sources in scoring")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--dry-run", action="store_true", help="Skip publisher step")
    parser.add_argument("--resources", type=Path, default=Path("config/resources.yaml"), help="Path to resource configuration")
    parser.add_argument("--prompts", type=Path, default=Path("config/prompts.yaml"), help="Path to prompt configuration")
    parser.add_argument("--mcp-config", type=Path, default=Path("mcp/mcp.config.json"), help="Path to MCP tool configuration")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.debug)
    log = logging.getLogger("runner")
    log.info("Loading configuration")
    resources = load_yaml(args.resources)
    prompts = load_yaml(args.prompts)
    toolkit = MCPToolkit.from_file(args.mcp_config)
    pipeline = NewsletterPipeline(toolkit=toolkit, resources=resources, prompts=prompts)
    pipeline.run(
        days=args.days,
        max_items=args.max_items,
        output_dir=args.out,
        turkish_weight=args.turkish_weight,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
