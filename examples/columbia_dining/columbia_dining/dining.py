"""Normalize published menu records and return filtered dining evidence."""

from dataclasses import asdict
from datetime import datetime
from typing import Any

from columbia_dining.menu_filters import (
    KNOWN_ALLERGENS,
    DiningRequest,
    Dish,
    MenuSnapshot,
    allergen,
    clean,
    select,
    show_hours,
    utc_to_ny,
)
from columbia_dining.menu_source import SOURCE_URL


def menu_snapshot(data: dict[str, Any]) -> MenuSnapshot:
    """Normalize menu records for date, hours, and dietary filtering."""
    locations = {
        str(loc["nid"]): {**loc, "title": clean(loc["title"])}
        for loc in data["locations"]
    }
    terms = data["terms"]
    dishes = []
    for hall_id, menus in data["menus_by_hall"].items():
        location = locations[hall_id]
        for menu in menus:
            for period_index, period in enumerate(menu.get("date_range_fields", [])):
                for meal_id in period.get("menu_type", []):
                    meal = clean(terms["types"][str(meal_id)]["name"])
                    for station_index, station in enumerate(period.get("stations", [])):
                        station_id = str(station["station"][0])
                        station_name = clean(terms["stations"][station_id]["name"])
                        for item_index, item in enumerate(
                            station.get("meals_paragraph", [])
                        ):
                            if not clean(item.get("title", "")):
                                continue
                            dishes.append(
                                Dish(
                                    id=(
                                        f"{hall_id}:{menu['nid']}:{period_index}:"
                                        f"{meal_id}:{station_index}:{item_index}"
                                    ),
                                    name=clean(item["title"]),
                                    description=clean(item.get("meal_text", "")),
                                    hall_id=hall_id,
                                    hall=clean(location["title"]),
                                    station=station_name,
                                    meal=meal,
                                    starts=utc_to_ny(period["date_from"]),
                                    ends=utc_to_ny(period["date_to"]),
                                    dietary_labels=tuple(item.get("prefs", [])),
                                    allergens=tuple(item.get("allergens", [])),
                                    allergen_field_present="allergens" in item,
                                    source_url=(
                                        SOURCE_URL.rstrip("/")
                                        + location.get("path", "/")
                                    ),
                                )
                            )
    return MenuSnapshot(
        fetched_at=datetime.fromisoformat(data["fetched_at"]),
        dishes=dishes,
        locations=locations,
    )


def lookup_menu(
    data: dict[str, Any], request: DiningRequest, now: datetime
) -> dict[str, Any]:
    """Return dated menu evidence for Arklex's response worker."""
    result = {
        "source_url": data["source_url"],
        "fetched_at": data["fetched_at"],
        "request": request.model_dump(mode="json"),
    }
    day = request.day or now.date()
    if request.meal == "now" and day != now.date():
        return {**result, "message": "Choose a meal for that date; now means today."}

    snapshot = menu_snapshot(data)
    if request.intent == "hours":
        return {**result, "hours": show_hours(snapshot, request, now)}

    unsupported = {allergen(x) for x in request.allergies or []} - KNOWN_ALLERGENS
    if unsupported:
        return {
            **result,
            "items": [],
            "message": "The menu labels cannot verify these allergies: "
            + ", ".join(sorted(unsupported))
            + ". Ask dining staff for ingredient information.",
        }

    # Return matching evidence for the response worker to explain.
    candidates = select(snapshot, request, now)
    items = []
    seen = set()
    for candidate in sorted(
        candidates, key=lambda c: (c.dish.hall, c.dish.station, c.dish.name)
    ):
        item = asdict(candidate.dish)
        item.pop("id")
        item["starts"] = item["starts"].isoformat()
        item["ends"] = item["ends"].isoformat()
        item["menu_windows_within_hall_hours"] = [
            [a.isoformat(), b.isoformat()] for a, b in candidate.windows
        ]
        key = (
            candidate.dish.hall,
            candidate.dish.station,
            candidate.dish.name,
            candidate.windows,
        )
        if key not in seen:
            items.append(item)
            seen.add(key)
    result["items"] = items
    result["message"] = (
        "Published menu windows may exceed individual station service hours."
        if items
        else "No published dishes match the date, meal, hall hours, and stated filters."
    )
    if not items and (request.intent == "recommend" or request.meal == "now"):
        published = select(
            snapshot,
            request.model_copy(
                update={
                    "intent": "menu",
                    "meal": None if request.meal == "now" else request.meal,
                }
            ),
            now,
        )
        if published:
            last_end = max(
                end for candidate in published for _, end in candidate.windows
            )
            result["message"] = (
                f"Matching dishes are published for {day.isoformat()}, but their "
                f"scheduled service ended at {last_end.isoformat()}."
                if last_end <= now
                else "Matching dishes are published, but none are scheduled to be served right now."
            )
    if request.meal is None:
        result["message"] += (
            f" No specific meal was selected; these are ideas from {day.isoformat()}'s "
            "published menus, not a list of dishes available right now. "
            "Use their serving windows when suggesting them."
        )
    if request.offer_ideas:
        result["message"] += (
            " The student wants help choosing. Offer varied ideas from the matching "
            "dishes before asking an optional follow-up; do not require them to pick "
            "a meal or craving first. Unknown restrictions are still unknown."
        )
    if request.allergies:
        result["allergy_note"] = (
            "Dishes with missing allergen fields or listed conflicts were excluded. "
            "Confirm ingredients and cross-contact with dining staff."
        )
    return result
