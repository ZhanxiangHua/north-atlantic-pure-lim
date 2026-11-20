"""Durable experiment events with UTC times and one locked append per event."""
import fcntl
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from tc_emulator.common import ROOT

WORK = Path(
    os.environ.get("TC_REFINEMENT_WORK", ROOT / "runs/annual_refinement_20261120")
).resolve()


def record(event, **details):
    path = WORK / "experiment_records" / "events.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "event": event,
        **details,
    }
    with path.open("a") as out:
        fcntl.flock(out, fcntl.LOCK_EX)
        out.write(json.dumps(row, allow_nan=False, sort_keys=True, default=str) + "\n")
        out.flush()
        os.fsync(out.fileno())
    return row
