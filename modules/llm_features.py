"""Optional LLM-extracted wine attribute features."""

from __future__ import annotations

import json
from typing import Any, Iterable, Optional

WINE_ATTRIBUTE_KEYS = [
    "fruitiness",
    "acidity",
    "tannin",
    "body",
    "sweetness",
    "oakiness",
    "earthiness",
    "complexity",
    "finish_length",
    "overall_sentiment",
]


def _mock_attributes() -> dict[str, float]:
    return {key: 0.0 for key in WINE_ATTRIBUTE_KEYS}


def _attribute_prompt(description: str) -> str:
    keys = ", ".join(WINE_ATTRIBUTE_KEYS)
    return (
        "Extract structured wine-review attributes from the description. "
        "Return only JSON with numeric scores from 0 to 1 for these keys: "
        f"{keys}.\n\nDescription:\n{description}"
    )


def extract_wine_attributes(
    description: str,
    client: Optional[Any] = None,
    model: Optional[str] = None,
    mock: bool = False,
) -> dict[str, Any]:
    """Extract structured attributes from one review.

    No external call is made unless ``client`` is explicitly provided.
    """
    if mock or client is None:
        return _mock_attributes()

    model = model or "gpt-4.1-mini"
    response = client.responses.create(
        model=model,
        input=_attribute_prompt(description),
    )
    text = response.output_text
    parsed = json.loads(text)
    return {key: parsed.get(key) for key in WINE_ATTRIBUTE_KEYS}


def batch_extract_wine_attributes(
    descriptions: Iterable[str],
    client: Optional[Any] = None,
    model: Optional[str] = None,
    mock: bool = False,
) -> list[dict[str, Any]]:
    """Extract attributes for a batch of descriptions."""
    return [
        extract_wine_attributes(
            description,
            client=client,
            model=model,
            mock=mock,
        )
        for description in descriptions
    ]

