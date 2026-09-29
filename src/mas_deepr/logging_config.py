"""Centralized logging setup for scripts and interactive use.

Deliberately separate from ``telemetry/tracker.py``: telemetry is a
structured, permanent JSONL cost/latency ledger read back with polars; this
is human-readable progress/debug output for watching a run live. Same
privacy rule as telemetry -- never log prompt or response text here, only
metadata (URLs, cache keys, error types, counts, role/pass identifiers).
"""

import logging
import sys

_CONFIGURED = False


def configure_logging(level: int = logging.INFO) -> None:
    """Attach one stderr handler to the ``mas_deepr`` logger tree.

    Idempotent -- safe to call from every script's ``main()`` (or import
    time) without duplicating handlers on repeated calls.
    """
    global _CONFIGURED
    logger = logging.getLogger("mas_deepr")
    if _CONFIGURED:
        logger.setLevel(level)
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    )
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    _CONFIGURED = True
