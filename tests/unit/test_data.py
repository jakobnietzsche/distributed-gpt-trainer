import torch

from dgpt.data import SlidingWindowDataset, split_into_train_and_validation


def test_sliding_window_dataset():
    data = [
        73, 102, 32, 73, 32, 104, 97, 118, 101, 32, 115, 101, 101, 110, 32,
        102, 117, 114, 116, 104, 101, 114, 32, 105, 116, 32, 105, 115, 32,
        98, 121, 32, 115, 116, 97, 110, 100, 105, 110, 103, 32, 111, 110,
        32, 116, 104, 101, 32, 115, 104, 111, 117, 108, 100, 101, 114, 115,
        32, 111, 102, 32, 71, 105, 97, 110, 116, 115, 46,
    ]
    dataset = SlidingWindowDataset(data, window_size=5)

    expected_length = len(data) - 5
    expected_first_item = [torch.tensor([73, 102, 32, 73, 32], dtype=torch.long), torch.tensor([102, 32, 73, 32, 104], dtype=torch.long)]
    expected_fifth_item = [torch.tensor([32, 104, 97, 118, 101], dtype=torch.long), torch.tensor([104, 97, 118, 101, 32], dtype=torch.long)]

    assert len(dataset) == expected_length
    assert torch.equal(dataset[0][0], expected_first_item[0])
    assert torch.equal(dataset[0][1], expected_first_item[1])
    assert torch.equal(dataset[4][0], expected_fifth_item[0])
    assert torch.equal(dataset[4][1], expected_fifth_item[1])


def test_split_into_train_and_validation():
    data = [
        73, 102, 32, 73, 32, 104, 97, 118, 101, 32, 115, 101, 101, 110, 32,
        102, 117, 114, 116, 104, 101, 114, 32, 105, 116, 32, 105, 115, 32,
        98, 121, 32, 115, 116, 97, 110, 100, 105, 110, 103, 32, 111, 110,
        32, 116, 104, 101, 32, 115, 104, 111, 117, 108, 100, 101, 114, 115,
        32, 111, 102, 32, 71, 105, 97, 110, 116, 115, 46,
    ]

    train_data, validation_data = split_into_train_and_validation(data, validation_ratio=0.2)

    assert len(train_data) == int(len(data) * (1 - 0.2))
    assert len(validation_data) == int(len(data) - len(train_data))
