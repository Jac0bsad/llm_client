from __future__ import annotations

import argparse
import sys

from .clients.openai_client import OpenAIClient


def cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="llm-client", description="LLM Client quick CLI"
    )
    parser.add_argument(
        "prompt",
        nargs="*",
        help="Prompt to send to the model (join with spaces)",
    )
    parser.add_argument(
        "--model",
        default="deepseek",
        help="Model config name defined in conf/openai_llms.yaml",
    )
    args = parser.parse_args(argv)

    text = " ".join(args.prompt).strip() if args.prompt else None
    if not text:
        parser.print_help()
        return 0

    client = OpenAIClient()
    messages = [{"role": "user", "content": text}]
    result = client.send_messages(messages, model=args.model)
    print(result)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(cli())
