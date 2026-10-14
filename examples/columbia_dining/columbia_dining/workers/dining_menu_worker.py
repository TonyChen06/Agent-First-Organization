"""Arklex worker for Columbia's live menus and dining hall hours."""

import json
from datetime import datetime
from typing import Any

from arklex.models.model_service import ModelService
from arklex.orchestrator.entities.orchestrator_state_entities import (
    OrchestratorState,
    StatusEnum,
)
from arklex.resources.tools.utils import ToolGenerator
from arklex.resources.workers.base.base_worker import BaseWorker
from arklex.resources.workers.base.entities import WorkerOutput
from columbia_dining.dining import lookup_menu
from columbia_dining.menu_filters import NY, DiningRequest
from columbia_dining.menu_source import MenuFetchError, fetch_menus

REQUEST_PROMPT = """Extract the student's Columbia Dining request from the conversation.
Use only the student's statements as preferences. Keep unknown fields null and
preserve earlier preferences unless explicitly changed. Empty lists mean explicit
none or no preference. Do not infer absence of allergies from silence.
Use Columbia labels for diets: Vegetarian, Vegan, Halal, Gluten Free. Preserve
unsupported restrictions instead of dropping them. Normalize hall abbreviations:
JJ/JJs -> JJ's, Ferris -> Ferris, John Jay -> John Jay.
Resolve relative dates using the supplied New York time. If the date is unspecified,
leave it null (today). Use intent=hours for opening hours, menu for menu questions,
recommend for meal choices, and out_of_scope for other questions.
For 'what is open now/currently', use intent=hours and meal=now. For general
opening-hours questions, leave meal null so the full day's hours are returned.
For cravings use food words. Keep dislikes in avoid. Treat 'surprise me' as no food
preference, without removing any stated restrictions. Never invent nutrition values.
Set offer_ideas=true when the student is unsure what they want, asks for suggestions,
or declines to choose a meal or preference (including 'neither' after a choice).
This does not mean they have no allergies or dietary restrictions: preserve those
already stated and keep unknown restrictions null. A specific meal is optional.
Do not infer 'now' from the current time or an earlier opening-hours question.
If they say they are not eating now or decline the offered meals, clear that meal
filter to null and offer ideas across the requested day's menus.
"""

PREFERENCE_QUESTION = (
    "Any food preferences or dietary restrictions? You can also say no preference."
)


class DiningMenuOutput(WorkerOutput):
    response: str
    status: StatusEnum


class DiningMenuWorker(BaseWorker):
    description: str = (
        "Answer meal and hours questions using Columbia's live dining data."
    )

    def init_worker_data(
        self, orch_state: OrchestratorState, node_specific_data: dict[str, Any]
    ) -> None:
        self.orch_state = orch_state

    def _request(self, now: datetime) -> DiningRequest:
        """Extract preferences using Arklex's configured model and history."""
        message = self.orch_state.user_message
        model = ModelService(self.orch_state.bot_config.llm_config).model
        prompt = REQUEST_PROMPT + f"\nCurrent New York time: {now.isoformat()}"
        return model.with_structured_output(
            DiningRequest, method="function_calling"
        ).invoke(
            [
                ("system", prompt),
                ("human", f"{message.history}\nLatest user message: {message.message}"),
            ]
        )

    def _output(self, response: str = "", evidence: str = "") -> DiningMenuOutput:
        self.orch_state.message_flow = evidence or response
        # Arklex's link validator reads the worker's recorded output.
        if self.orch_state.trajectory and self.orch_state.trajectory[-1]:
            self.orch_state.trajectory[-1][-1].output = evidence or response
        if evidence:
            response = ToolGenerator.context_generate(self.orch_state)
        return DiningMenuOutput(response=response, status=StatusEnum.COMPLETE)

    def _execute(self) -> DiningMenuOutput:
        now = datetime.now(NY)
        try:
            request = self._request(now)
        except Exception:
            return self._output(
                "I couldn't read your request. Please check the model connection "
                "and try again."
            )

        if request.intent == "out_of_scope":
            return self._output(
                "I can help with Columbia Dining menus, hours, and dining questions."
            )
        if (
            request.intent == "recommend"
            and request.diets is None
            and request.allergies is None
            and request.cravings is None
            and not request.avoid
            and not request.high_protein
            and not request.offer_ideas
            and PREFERENCE_QUESTION not in self.orch_state.user_message.history
        ):
            return self._output(PREFERENCE_QUESTION)
        try:
            evidence = lookup_menu(fetch_menus(), request, now)
        except (MenuFetchError, ValueError, KeyError, TypeError, IndexError):
            return self._output(
                "I couldn't read Columbia's current menus and hours. "
                "Please try again or check https://dining.columbia.edu/."
            )
        return self._output(evidence=json.dumps(evidence, ensure_ascii=False))
