import datetime
import platform

import torch


class EnvironmentReport:
    """
    Generates a report of current environment.
    Used for debugging and reproducibility.
    """
    
    def __init__(self):
        self.os_name: str = platform.system()
        self.os_version: str = platform.release()
        self.machine: str = platform.machine()
        self.python_version: str = platform.python_version()
        self.pytorch_version: str = torch.__version__
        self.device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.timestamp: datetime.datetime = datetime.datetime.now(tz=datetime.UTC)

    def __str__(self) -> str:
        return (
            "Environment Report\n"
            f"  Operating system: {self.os_name} {self.os_version} ({self.machine})\n"
            f"  Python:           {self.python_version}\n"
            f"  PyTorch:          {self.pytorch_version}\n"
            f"  Device:           {str(self.device).upper()}\n"
            f"  Created at:       {self.timestamp:%Y-%m-%d %H:%M:%S UTC}\n"
        )
    