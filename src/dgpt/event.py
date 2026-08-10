import datetime
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Event:
    """
    Represents a structured event recorded during an experiment.

    Events are serialized into JSONL for logging and analysis.
    """
    run_id: str
    rank: int
    step: int
    config_hash: str
    metrics: dict[str, int | float]
    event_type: str
    code_revision: str
    timestamp: datetime.datetime = field(default_factory=lambda: datetime.datetime.now(tz=datetime.UTC))
