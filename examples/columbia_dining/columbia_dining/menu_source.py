"""Fetch structured menu data from Columbia Dining's homepage."""

import json
import re
from datetime import datetime, timezone
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

SOURCE_URL = "https://dining.columbia.edu/"


class MenuFetchError(RuntimeError):
    """The dining page could not be downloaded or its menu data read."""


def _decode_template(value: str) -> object:
    """Decode the outer JavaScript string, then the JSON inside it."""
    value = value.replace("\\'", "'").replace("\\`", "`").replace("\\$", "$")
    return json.loads(json.loads('"' + value + '"'))


def parse_menu_page(page: str) -> dict[str, Any]:
    """Extract locations, terminology, and menus without changing source fields."""
    try:
        embedded: dict[str, Any] = {}
        for name in ("dining_nodes", "dining_terms"):
            match = re.search(
                r"var\s+" + name + r"\s*=\s*`((?:\\.|[^`])*)`", page, re.DOTALL
            )
            if match is None:
                raise MenuFetchError(f"The dining page is missing {name}.")
            embedded[name] = _decode_template(match[1])

        matches = list(
            re.finditer(
                r"hall_menus_data\[(\d+)\]\s*=\s*JSON.parse\(`((?:\\.|[^`])*)`\)",
                page,
                re.DOTALL,
            )
        )
        if not matches:
            raise MenuFetchError("The dining page contains no readable hall menu data.")

        locations = embedded["dining_nodes"]["locations"]
        terms = embedded["dining_terms"]
        menus = {match[1]: _decode_template(match[2]) for match in matches}
        if not isinstance(locations, list) or not all(
            isinstance(loc, dict) for loc in locations
        ):
            raise ValueError("Expected location records")
        if not isinstance(terms, dict) or not all(
            isinstance(terms.get(key), dict) for key in ("types", "stations")
        ):
            raise ValueError("Expected meal and station definitions")
        if not all(
            isinstance(entries, list)
            and all(isinstance(entry, dict) for entry in entries)
            for entries in menus.values()
        ):
            raise ValueError("Expected menu records grouped by hall")
        return {"locations": locations, "terms": terms, "menus_by_hall": menus}
    except (ValueError, KeyError, TypeError) as exc:
        raise MenuFetchError(
            "The dining page's menu data has an unexpected format."
        ) from exc


def fetch_menus() -> dict[str, Any]:
    """Download Columbia's current page and return its published menu data."""
    # The site rejected urllib's default User-Agent during development.
    request = Request(SOURCE_URL, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urlopen(request, timeout=30) as response:
            page = response.read().decode("utf-8")
    except (URLError, OSError, UnicodeError) as exc:
        raise MenuFetchError(f"Could not read {SOURCE_URL}: {exc}") from exc

    data = parse_menu_page(page)
    return {
        "source_url": SOURCE_URL,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        **data,
    }
