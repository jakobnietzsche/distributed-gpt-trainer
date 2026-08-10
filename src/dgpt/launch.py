import secrets
from datetime import UTC, datetime
from pathlib import Path

from dgpt.config import ExperimentConfig, generate_config_hash
from dgpt.event import Event
from dgpt.event_logger import EventLogger


def launch():
    """
    Launches the distributed GPT training process.
    """
    experiment_config = ExperimentConfig()
    
    run_id = generate_run_id()
    event_logger = EventLogger(run_directory="runs/", run_id=run_id)

    event_logger.log_event(
        Event(
            run_id=run_id,
            rank=0,
            step=0,
            config_hash=generate_config_hash(experiment_config),
            metrics={},
            event_type="benchmark_completed",
            code_revision="abc123",  # Placeholder for actual code revision
        )
    )

    event_logger.close()

def generate_run_id() -> str:
    """
    Generates a unique run ID based on the current timestamp.
    """
    now = datetime.now(tz=UTC)
    adjectives_path = Path(__file__).parent / "resources/adjectives.txt"
    nouns_path = Path(__file__).parent / "resources/nouns.txt"

    adjectives = adjectives_path.read_text(encoding="utf-8").splitlines()
    nouns = nouns_path.read_text(encoding="utf-8").splitlines()

    adjective = secrets.choice(adjectives)
    noun = secrets.choice(nouns)

    return f"{adjective}-{noun}-{now.strftime('%Y%m%d%H%M%S')}"


if __name__ == "__main__":
    launch()