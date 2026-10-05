#!/usr/bin/env python3
"""Print the last immutable panel-generation event and process status."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cache_dir", type=Path)
    args = parser.parse_args()
    events = sorted((args.cache_dir/"events").glob("g*.json"))
    manifests = sorted(args.cache_dir.glob("chunked-*.json"))
    result = {"event_count": len(events)}
    if manifests:
        manifest = json.loads(manifests[0].read_text())
        result.update({name: manifest.get(name) for name in (
            "status", "accepted_points", "peak_force_n", "peak_confirmed",
            "post_peak_observed", "checkpoint_file", "checkpoint_sha256",
        )})
    if events:
        event = json.loads(events[-1].read_text())
        result["last_event"] = {name: event.get(name) for name in (
            "generation_requested", "generation_committed", "status", "error",
            "elapsed_seconds", "event_sha256",
        )}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

