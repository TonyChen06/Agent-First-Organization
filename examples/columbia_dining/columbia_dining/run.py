"""Chat with the Columbia Dining agent in the terminal."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

from dotenv import load_dotenv

if TYPE_CHECKING:
    from arklex.orchestrator.orchestrator import AgentOrg

ROOT = Path(__file__).resolve().parent


def make_agent() -> AgentOrg:
    load_dotenv(ROOT.parent / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("Set OPENAI_API_KEY in the project's .env file.")
    os.environ.setdefault("DATA_DIR", str(ROOT / "data"))
    if not (Path(os.environ["DATA_DIR"]) / "chunked_documents.pkl").is_file():
        raise FileNotFoundError(
            "Reference passages not found in DATA_DIR. Run "
            "`.venv/bin/python -m columbia_dining.prepare_docs` from the project root."
        )

    logging.getLogger("arklex").setLevel(logging.WARNING)
    # Optional integrations are imported eagerly; required workers are checked below.
    logging.getLogger("arklex.resources.resource_map").addFilter(
        lambda record: not record.getMessage().startswith("Failed to import ")
    )
    from arklex.models.llm_config import LLMConfig
    from arklex.orchestrator.executor.executor import Executor
    from arklex.orchestrator.orchestrator import AgentOrg
    from columbia_dining.workers.register import register_workers

    graph = json.loads((ROOT / "taskgraph.json").read_text())
    register_workers()
    executor = Executor(
        tools=graph["tools"],
        workers=graph["workers"],
        nodes=graph["nodes"],
        llm_config=LLMConfig.model_validate(graph["llm_config"]),
    )
    missing = {worker["id"] for worker in graph["workers"]} - executor.workers.keys()
    if missing:
        raise ValueError(
            f"Required workers could not load: {', '.join(sorted(missing))}"
        )
    return AgentOrg(config=graph, executor=executor)


async def chat(agent: AgentOrg) -> None:
    history, parameters = [], {}
    text = "<start>"
    print("Columbia Dining · type /reset to start over or /quit to exit.\n")
    while True:
        result = await agent.get_response(
            {
                "text": text,
                "chat_history": history,
                "parameters": parameters,
            }
        )
        parameters = result["parameters"]
        history.extend(
            [
                {"role": "user", "content": text},
                {"role": "assistant", "content": result["answer"]},
            ]
        )
        print(f"Dining: {result['answer']}\n")
        text = input("You: ").strip()
        while not text:
            text = input("You: ").strip()
        if text.lower() in {"/quit", "quit", "exit"}:
            return
        if text.lower() in {"/reset", "reset"}:
            history, parameters, text = [], {}, "<start>"


def main() -> None:
    try:
        asyncio.run(chat(make_agent()))
    except (KeyboardInterrupt, EOFError):
        print("\nGoodbye!")
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
