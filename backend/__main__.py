"""Serve the policies over HTTP.

    python -m backend --adapter run1=artifacts/rl/run1/adapter-final \\
                      --adapter run2=artifacts/rl/run2/adapter-final

The baseline is always served, as `baseline`. Binds to localhost by default:
this is a single-user local demo and the middle layer is its only client.
"""

import argparse
import sys
from pathlib import Path

from baseline_agent import DEFAULT_MODEL
from tools import build_registry, code_exec

from .app import create_app
from .engine import ServedPolicies, check_adapters


def parse_adapter(text: str) -> tuple[str, Path]:
    name, sep, path = text.partition("=")
    if not sep or not name or not path:
        raise argparse.ArgumentTypeError(f"expected NAME=PATH, got {text!r}")
    return name, Path(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m backend", description=__doc__.split("\n")[0])
    parser.add_argument(
        "--adapter", type=parse_adapter, action="append", default=[], metavar="NAME=PATH"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    arguments = parser.parse_args(argv)

    adapters = dict(arguments.adapter)
    if len(adapters) != len(arguments.adapter):
        print("two adapters share a name", file=sys.stderr)
        return 1
    try:
        check_adapters(adapters)
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1

    import uvicorn

    # The same rule `baseline_agent` applies, so the served prompt advertises
    # the tools the evaluated prompt did. Nothing here runs them.
    registry = build_registry(include_code=code_exec.sandbox_available())
    engine = ServedPolicies(adapters, model_name=arguments.model)
    uvicorn.run(create_app(engine, registry), host=arguments.host, port=arguments.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
