"""Run five simulated conversations and evaluate them with ArkSim."""

import asyncio
import json
from uuid import uuid4

from arksim.config import AgentConfig
from arksim.evaluator import EvaluationInput, run_evaluation
from arksim.simulation_engine import (
    AgentResponse,
    Simulation,
    SimulationInput,
    ToolCall,
    run_simulation,
)
from arksim.simulation_engine.agent.base import BaseAgent
from dotenv import load_dotenv

from columbia_dining.run import ROOT, make_agent


class DiningAgent(BaseAgent):
    def __init__(self, agent_config: AgentConfig) -> None:
        super().__init__(agent_config)
        self.agent = make_agent()
        self.history = []
        self.parameters = {}
        self.chat_id = str(uuid4())

    async def get_chat_id(self) -> str:
        return self.chat_id

    async def _turn(self, text: str) -> str:
        result = await self.agent.get_response(
            {
                "text": text,
                "chat_history": self.history,
                "parameters": self.parameters,
            }
        )
        self.parameters = result["parameters"]
        self.history.extend(
            [
                {"role": "user", "content": text},
                {"role": "assistant", "content": result["answer"]},
            ]
        )
        return result["answer"]

    async def execute(self, user_query: str, **kwargs: object) -> AgentResponse:
        if not self.history:
            await self._turn("<start>")
        answer = await self._turn(user_query)
        trajectory = self.parameters.get("memory", {}).get("trajectory", [])
        calls = []
        # Keep worker evidence alongside each answer for review.
        for record in trajectory[-1] if trajectory else []:
            if record.get("output"):
                calls.append(
                    ToolCall(
                        id=str(uuid4()),
                        name=record["info"]["resource"]["name"],
                        arguments={"query": user_query},
                        result=record["output"],
                    )
                )
        return AgentResponse(content=answer, tool_calls=calls)


def score(simulation: Simulation) -> None:
    """Give the judge reference text and recorded menu data as evidence."""
    output = ROOT / "data" / "evaluation"
    evaluation_input = simulation.model_copy(deep=True)
    references = [
        p.read_text() for p in sorted((ROOT / "docs/reference").glob("*.txt"))
    ]
    for conversation in evaluation_input.conversations:
        evidence = list(references)
        for message in conversation.conversation_history:
            for call in message.tool_calls or []:
                if call.name == "DiningMenuWorker" and call.result.startswith("{"):
                    evidence.append(
                        f"Turn {message.turn_id} menu evidence: {call.result}"
                    )
        conversation.simulated_user_prompt.variables["scenario.knowledge"] = evidence
    (output / "evaluation_input.json").write_text(
        evaluation_input.model_dump_json(indent=2)
    )
    run_evaluation(
        EvaluationInput(
            model="gpt-6-luna",
            provider="openai",
            num_workers=4,
            scenario_file_path=str(ROOT / "scenarios.json"),
            output_dir=str(output),
            generate_html_report=True,
            metrics_to_run=[
                "faithfulness",
                "helpfulness",
                "coherence",
                "relevance",
                "goal_completion",
                "agent_behavior_failure",
            ],
        ),
        simulation=evaluation_input,
    )


def main() -> None:
    load_dotenv(ROOT.parent / ".env")
    output = ROOT / "data" / "evaluation"
    common = {
        "model": "gpt-6-luna",
        "provider": "openai",
        "num_workers": 1,
        "scenario_file_path": str(ROOT / "scenarios.json"),
    }
    simulation = asyncio.run(
        run_simulation(
            SimulationInput(
                **common,
                agent_config={
                    "agent_type": "custom",
                    "agent_name": "columbia-dining",
                    "custom_config": {"agent_class": DiningAgent},
                },
                num_conversations_per_scenario=1,
                max_turns=5,
                output_file_path=str(output / "simulation.json"),
            )
        )
    )
    expected = len(json.loads((ROOT / "scenarios.json").read_text())["scenarios"])
    if len(simulation.conversations) != expected:
        raise RuntimeError("Some simulated conversations failed; see the run log.")
    score(simulation)


if __name__ == "__main__":
    main()
