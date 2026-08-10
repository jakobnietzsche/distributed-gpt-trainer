import hashlib
import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ExperimentConfig:
    """
    Configuration for an experiment.
    Serves as a baseline for all experiments.
    Useful background on the machine learning term “experiment”:
        https://developers.google.com/machine-learning/managing-ml-projects/experiments
    """

    learning_rate: float = 0.001
    batch_size: int = 24
    epochs: int = 10
    seed: int = 50

def generate_config_hash(config: ExperimentConfig) -> str:
    """
    Generates a hash of the current experiment configuration.
    Useful for reproducibility and tracking experiments.
    """
    config_str = json.dumps(asdict(config), sort_keys=True)
    return hashlib.sha256(config_str.encode("utf-8")).hexdigest()