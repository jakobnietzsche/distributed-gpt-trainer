import json
from dataclasses import asdict
from pathlib import Path

from dgpt.event import Event


class EventLogger:
    """
    Logs events during training and evaluation.
    Stores events for metrics and debugging.
    """

    def __init__(self, run_directory: str, run_id: str):
        run_directory = Path(run_directory) / run_id
        run_directory.mkdir(parents=True, exist_ok=True)

        log_path = run_directory / "events.jsonl"
        self.log_file = log_path.open(mode="w", encoding="utf-8")

    def log_event(self, event: Event) -> None:
        event_data = asdict(event)
        event_data["timestamp"] = event.timestamp.isoformat()

        self.log_file.write(json.dumps(event_data) + "\n")
        self.log_file.flush()

    def close(self) -> None:
        self.log_file.close()