"""Optional few-shot LLM classification and regression."""

from __future__ import annotations

import re
from typing import Any, Optional


def _format_examples(examples: Optional[list[dict[str, Any]]]) -> str:
    if not examples:
        return ""
    lines = []
    for example in examples:
        lines.append(f"Review: {example['description']}\nAnswer: {example['label']}")
    return "\n\n".join(lines)


def _call_llm(prompt: str, client: Any, model: Optional[str]) -> str:
    response = client.responses.create(model=model or "gpt-4.1-mini", input=prompt)
    return response.output_text.strip()


def few_shot_classify_review(
    description: str,
    class_labels: list[str],
    examples: Optional[list[dict[str, Any]]] = None,
    client: Optional[Any] = None,
    model: Optional[str] = None,
) -> str:
    """Classify one review with a few-shot LLM prompt.

    No external call is made unless ``client`` is provided.
    """
    if client is None:
        raise ValueError("Pass an LLM client explicitly to run few-shot classification.")

    labels = ", ".join(class_labels)
    prompt = (
        "Predict the wine rating class from the review. "
        f"Allowed labels: {labels}. Return exactly one label.\n\n"
        f"{_format_examples(examples)}\n\nReview: {description}\nAnswer:"
    )
    output = _call_llm(prompt, client, model)
    for label in class_labels:
        if output.strip().lower() == label.lower():
            return label
    raise ValueError(f"LLM returned invalid class label: {output}")


def few_shot_predict_points(
    description: str,
    examples: Optional[list[dict[str, Any]]] = None,
    client: Optional[Any] = None,
    model: Optional[str] = None,
    min_points: float = 80.0,
    max_points: float = 100.0,
) -> float:
    """Predict numeric points with a few-shot LLM prompt."""
    if client is None:
        raise ValueError("Pass an LLM client explicitly to run few-shot regression.")

    prompt = (
        "Predict the numeric Wine Enthusiast points score for the review. "
        "Return only one number.\n\n"
        f"{_format_examples(examples)}\n\nReview: {description}\nPoints:"
    )
    output = _call_llm(prompt, client, model)
    match = re.search(r"\d+(?:\.\d+)?", output)
    if not match:
        raise ValueError(f"Could not parse numeric prediction from: {output}")
    prediction = float(match.group(0))
    if not min_points <= prediction <= max_points:
        raise ValueError(f"Prediction {prediction} outside expected range.")
    return prediction

