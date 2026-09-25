import torch


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean cross-entropy for logits [B, T, V] and targets [B, T]."""
    log_probs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    correct_log_probs = log_probs.gather(
        dim=-1, index=targets.unsqueeze(-1)
    ).squeeze(-1)
    return -correct_log_probs.mean()
