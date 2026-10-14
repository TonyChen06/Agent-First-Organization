"""New York serving windows and source-label filters for dining menus."""

import html
import re
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from typing import Literal
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field

NY = ZoneInfo("America/New_York")


def clean(value: str) -> str:
    return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", value)).split())


def utc_to_ny(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(NY)


def local_hour(day: date, value: str) -> datetime:
    hour, minute = divmod(int(value), 100)
    if hour == 24 and minute == 0:
        return datetime.combine(day + timedelta(days=1), time.min, NY)
    return datetime.combine(day, time(hour, minute), NY)


@dataclass(frozen=True)
class Dish:
    id: str
    name: str
    description: str
    hall_id: str
    hall: str
    station: str
    meal: str
    starts: datetime
    ends: datetime
    dietary_labels: tuple[str, ...]
    allergens: tuple[str, ...]
    allergen_field_present: bool
    source_url: str


@dataclass
class MenuSnapshot:
    fetched_at: datetime
    dishes: list[Dish]
    locations: dict[str, dict]

    def opening_windows(
        self, hall_id: str, day: date
    ) -> list[tuple[datetime, datetime]]:
        """Include overnight service that began on the preceding local date."""
        windows = []
        for anchor in (day - timedelta(days=1), day):
            for schedule in self.locations[hall_id].get("open_hours_fields", []):
                if anchor.isoformat() in schedule.get("excluded", []):
                    continue
                valid_from = utc_to_ny(schedule["date_from"])
                valid_to = utc_to_ny(schedule["date_to"])
                if not valid_from.date() <= anchor <= valid_to.date():
                    continue
                for weekdays in schedule.get("days", []):
                    if not isinstance(weekdays, dict):
                        continue
                    for hours in weekdays.get(
                        "days_" + anchor.strftime("%A").lower(), []
                    ):
                        start = local_hour(anchor, hours["hours_from"])
                        end = local_hour(anchor, hours["hours_to"])
                        if end <= start:
                            end += timedelta(days=1)
                        start, end = max(start, valid_from), min(end, valid_to)
                        # Exclusions apply to the calendar date as well as the anchor.
                        if day.isoformat() in schedule.get("excluded", []):
                            continue
                        if start < end and end > datetime.combine(day, time.min, NY):
                            windows.append((start, end))
        return sorted(set(windows))


class DiningRequest(BaseModel):
    meal: (
        Literal["breakfast", "brunch", "lunch", "dinner", "late night", "now"] | None
    ) = Field(
        default=None,
        description="Null means no specific meal; search the requested day's menus.",
    )
    day: date | None = None
    diets: list[str] | None = Field(
        default=None,
        description="Null means not asked; [] means explicitly no restrictions.",
    )
    allergies: list[str] | None = Field(
        default=None, description="Null means unknown; [] means explicitly none."
    )
    cravings: list[str] | None = Field(
        default=None, description="Food keywords; [] means anything/surprise me."
    )
    avoid: list[str] = Field(default_factory=list)
    hall: str | None = None
    high_protein: bool = False
    offer_ideas: bool = Field(
        default=False,
        description="The user is undecided, asks for ideas, or declines to choose preferences.",
    )
    intent: Literal["recommend", "menu", "hours", "out_of_scope"] = "recommend"


def normalized(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


ALLERGEN_ALIASES = {
    "milk": "dairy",
    "lactose": "dairy",
    "egg": "eggs",
    "peanut": "peanuts",
    "tree nut": "tree nuts",
    "nut": "nuts",
    "soybean": "soy",
    "soybeans": "soy",
    "gluten": "wheat",
    "sesame seed": "sesame",
    "sesame seeds": "sesame",
}
KNOWN_ALLERGENS = {
    "dairy",
    "eggs",
    "peanuts",
    "tree nuts",
    "nuts",
    "soy",
    "wheat",
    "sesame",
    "fish",
    "shellfish",
    "coconut",
    "sunflower oil",
    "pork",
}


def allergen(value: str) -> str:
    value = normalized(value)
    return ALLERGEN_ALIASES.get(value, value)


@dataclass(frozen=True)
class Candidate:
    dish: Dish
    windows: tuple[tuple[datetime, datetime], ...]


def meal_matches(dish_meal: str, requested: str) -> bool:
    value = normalized(dish_meal)
    return (
        requested == "now"
        or value == requested
        or value == "daily"
        or value == "lunch dinner"
        and requested in {"lunch", "dinner"}
        or value == "brunch"
        and requested in {"breakfast", "lunch"}
    )


def select(
    snapshot: MenuSnapshot, request: DiningRequest, now: datetime
) -> list[Candidate]:
    day = request.day or now.astimezone(NY).date()
    start_day = datetime.combine(day, time.min, NY)
    end_day = start_day + timedelta(days=1)
    avoids = [normalized(x) for x in request.avoid if normalized(x)]
    allergens = {allergen(x) for x in request.allergies or []}
    if allergens - KNOWN_ALLERGENS:
        return []
    candidates = []
    for dish in snapshot.dishes:
        if not (dish.starts < end_day and dish.ends > start_day):
            continue
        if not meal_matches(dish.meal, request.meal or "now"):
            continue
        hall = normalized(request.hall or "")
        hall = {"jj": "jj s", "jjs": "jj s"}.get(hall, hall)
        if hall and hall not in normalized(dish.hall):
            continue
        windows = tuple(
            (max(dish.starts, a, start_day), min(dish.ends, b, end_day))
            for a, b in snapshot.opening_windows(dish.hall_id, day)
            if max(dish.starts, a, start_day) < min(dish.ends, b, end_day)
        )
        if request.intent == "recommend":
            windows = tuple((a, b) for a, b in windows if b > now)
        if not windows or (
            request.meal == "now" and not any(a <= now < b for a, b in windows)
        ):
            continue
        labels = {normalized(x) for x in dish.dietary_labels}
        required = {normalized(x) for x in request.diets or []}
        if "vegan" in labels:
            labels.add("vegetarian")
        if not required <= labels:
            continue
        # Wheat labels alone do not cover all gluten-containing ingredients.
        if (
            any(normalized(x) == "gluten" for x in request.allergies or [])
            and "gluten free" not in labels
        ):
            continue
        listed = {allergen(x) for x in dish.allergens}
        expanded = set(listed)
        if listed & {"peanuts", "tree nuts", "nuts"}:
            expanded.add("nuts")
        if "nuts" in listed:
            expanded.update({"peanuts", "tree nuts"})
        if allergens and (not dish.allergen_field_present or allergens & expanded):
            continue
        food = normalized(dish.name + " " + dish.description)
        if any(re.search(r"\b" + re.escape(x) + r"\b", food) for x in avoids):
            continue
        candidates.append(Candidate(dish, windows))
    return candidates


def clock_label(value: datetime) -> str:
    return value.strftime("%-I:%M %p")


def show_hours(snapshot: MenuSnapshot, request: DiningRequest, now: datetime) -> str:
    now = now.astimezone(NY)
    day = request.day or now.date()
    lines = [
        f"Halls scheduled open at {now.isoformat()} (New York time):"
        if request.meal == "now"
        else f"Published hall hours for {day.isoformat()} (New York time):"
    ]
    for hall_id, location in snapshot.locations.items():
        if request.hall and normalized(request.hall) not in normalized(
            location["title"]
        ):
            continue
        windows = snapshot.opening_windows(hall_id, day)
        if request.meal == "now":
            windows = [(a, b) for a, b in windows if a <= now < b]
        if windows:

            def label(value: datetime) -> str:
                suffix = f" ({value.strftime('%b %-d')})" if value.date() != day else ""
                return clock_label(value) + suffix

            hours = ", ".join(f"{label(a)}–{label(b)}" for a, b in windows)
            lines.append(f"- {clean(location['title'])}: {hours}")
    if len(lines) == 1:
        lines.append("No matching opening hours were found.")
    lines.append(
        "Source: https://dining.columbia.edu/ — individual stations may have shorter service hours."
    )
    return "\n".join(lines)
