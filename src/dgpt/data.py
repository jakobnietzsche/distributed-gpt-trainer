import torch


class SlidingWindowDataset:
    """
    Creates input/target windows for next-token prediction.

    Example:
        data = [1, 2, 3, 4, 5], window_size = 3

        input  = [1, 2, 3]
        target = [2, 3, 4]
    """
    def __init__(self, data: list[int], window_size: int):
        if window_size <= 0:
            raise ValueError("Window size must be greater than zero")
        if len(data) <= window_size:
            raise ValueError("Window size must be less than the length of the data")
        self.data = data
        self.window_size = window_size

    def __len__(self) -> int:
        return len(self.data) - self.window_size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of bounds")
        input_tensor = torch.tensor(self.data[idx : idx + self.window_size], dtype=torch.long)
        target_tensor = torch.tensor(self.data[idx + 1: idx + self.window_size + 1], dtype=torch.long)
        return input_tensor, target_tensor


def split_into_train_and_validation(data: list[int], validation_ratio: float = 0.1) -> tuple[list[int], list[int]]:
    """
    Splits the data into training and validation sets based on the specified ratio.
    """
    if not 0 < validation_ratio < 1:
        raise ValueError("Validation ratio must be between 0 and 1")

    split_index = int(len(data) * (1 - validation_ratio))
    return data[:split_index], data[split_index:]