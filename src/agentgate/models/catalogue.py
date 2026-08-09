"""``make models`` -- list the model identifiers this key can reach.

This exists to answer one question: what do I put in ``AGENTGATE_CLOUD_CHEAP_MODEL``? The
answer is not something this repository can know, and guessing a name that has since been
retired fails at request time rather than at startup.

**It cannot tell you what anything costs.** The models endpoint returns identifiers, ownership,
and creation timestamps -- no pricing, at all. Ordering by cost would require the price table
this command exists to help you populate, which is circular. So it lists what is reachable and
emits a ready-to-paste price line with zeroed values for you to fill in from the provider's
pricing page.

Nothing here infers a price, a size, or a capability from a model's name. A name is a string.

Deliberately does not go through cloud-lane validation. That path requires a populated price
table, and needing this command is the state you are in *before* you have one.
"""

from __future__ import annotations

import json
import sys
from typing import Any, Final

import httpx

from agentgate.config import ConfigurationError, get_settings

EXIT_OK: Final = 0
EXIT_NO_KEY: Final = 2
EXIT_REQUEST_FAILED: Final = 3

DEFAULT_API_BASE: Final = "https://api.openai.com/v1"
PRICE_VARIABLE: Final = "AGENTGATE_MODEL_PRICES_USD_PER_MILLION"


def fetch_models(api_key: str, base_url: str, *, timeout: float) -> list[dict[str, Any]]:
    """Return the raw model records the key can reach.

    Raises:
        httpx.HTTPError: on any transport or status failure.
    """
    response = httpx.get(
        f"{base_url.rstrip('/')}/models",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=timeout,
    )
    response.raise_for_status()
    payload: dict[str, Any] = response.json()
    records: list[dict[str, Any]] = payload.get("data", [])
    return sorted(records, key=lambda record: str(record.get("id", "")))


def price_line(model_ids: list[str]) -> str:
    """A paste-ready assignment with every identifier zeroed out.

    Zeros rather than plausible numbers: an unpriced model refuses to start, which is a
    prompt to fill this in. A plausible-looking wrong number would start happily and quietly
    mis-account every run.
    """
    table = {model_id: {"input": 0.0, "output": 0.0} for model_id in model_ids}
    return f"{PRICE_VARIABLE}={json.dumps(table, separators=(',', ':'), sort_keys=True)}"


def render(records: list[dict[str, Any]], *, only: list[str] | None = None) -> str:
    """Format the listing and the follow-up instructions.

    Args:
        records: Reachable models, as returned by the provider.
        only: Restrict the emitted price line to these identifiers. The full listing is
            still printed -- filtering the *line* keeps it pasteable without hiding what
            is reachable, which is the question the listing answers.
    """
    if not records:
        return "This key can reach no models. Check the key and the base URL."

    ids = [str(record.get("id", "")) for record in records]
    width = max(len(model_id) for model_id in ids)

    lines = [f"{len(ids)} model(s) reachable with this key:", ""]
    lines.extend(
        f"  {model_id:<{width}}  owned by {record.get('owned_by', 'unknown')}"
        for model_id, record in zip(ids, records, strict=True)
    )
    lines.extend(
        [
            "",
            "Pricing is not available from this API. The models endpoint returns identifiers,",
            "ownership, and timestamps -- nothing about cost -- so this list is not ordered by",
            "price and no price has been inferred from any name.",
            "",
            "Next steps:",
            f"  1. Choose a small, cheap model and set it as BOTH {'AGENTGATE_CLOUD_CHEAP_MODEL'}",
            "     and AGENTGATE_CLOUD_CAPABLE_MODEL. Point the capable tier at something",
            "     costlier only as a deliberate edit.",
            "  2. Read the current per-million input and output prices from the provider's",
            "     pricing page for the models you actually intend to use.",
            "  3. Paste the line below into .env, keeping only those models, and replace the",
            "     zeros. Record the date you checked -- prices change and a stale number that",
            "     looks authoritative is worse than none.",
            "",
            "A model with no price will not start: unknown cost is never read as zero.",
            "",
            price_line(only if only else ids),
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """List reachable models and emit a price-table skeleton.

    Args:
        argv: Optional model identifiers. When given, the price line covers only those,
            which keeps it pasteable when a key can reach a hundred models. The listing
            itself is always complete.

    Returns:
        ``0`` on success, ``2`` if no key is configured, ``3`` if the request failed or an
        identifier was not reachable.
    """
    requested = list(argv or [])

    try:
        settings = get_settings()
    except ConfigurationError as error:
        print(str(error), file=sys.stderr)
        return EXIT_NO_KEY

    if settings.openai_api_key is None:
        print(
            "No API key configured. Set OPENAI_API_KEY in .env, then run this again.",
            file=sys.stderr,
        )
        return EXIT_NO_KEY

    base_url = settings.openai_base_url or DEFAULT_API_BASE
    try:
        records = fetch_models(
            settings.openai_api_key.get_secret_value(),
            base_url,
            timeout=settings.request_timeout_seconds,
        )
    except httpx.HTTPError as error:
        # str(error) on an httpx error includes the URL but never the Authorization header.
        print(f"Could not list models from {base_url}: {error}", file=sys.stderr)
        return EXIT_REQUEST_FAILED

    reachable = {str(record.get("id", "")) for record in records}
    unknown = sorted(set(requested) - reachable)
    if unknown:
        print(
            f"Not reachable with this key: {', '.join(unknown)}. "
            "Run without arguments to see the full list.",
            file=sys.stderr,
        )
        return EXIT_REQUEST_FAILED

    print(render(records, only=requested or None))
    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    sys.exit(main(sys.argv[1:]))
