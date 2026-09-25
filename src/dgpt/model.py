import torch
from torch import nn


class Embeddings(nn.Module):
    """
    Converts token IDs into learned vector representations for the model.

    Combines token embeddings, which represent what each token is, with
    positional embeddings, which represent where each token occurs in the sequence.
    """

    def __init__(self, vocab_size: int, embedding_dim: int, context_size: int) -> None:
        super().__init__()
        self.token_embeddings = nn.Embedding(
            embedding_dim=embedding_dim, num_embeddings=vocab_size
        )
        self.position_embeddings = nn.Embedding(
            embedding_dim=embedding_dim, num_embeddings=context_size
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Map token IDs [batch, sequence] to embeddings [batch, sequence, embedding_dim]."""
        position_ids = torch.arange(
            token_ids.size(1), device=token_ids.device
        ).unsqueeze(0)
        token_embeddings = self.token_embeddings(token_ids)
        position_embeddings = self.position_embeddings(position_ids)

        return token_embeddings + position_embeddings


class AttentionHead(nn.Module):
    """
    A single attention head that computes attention scores and applies them to the input embeddings.
    """

    causal_mask: torch.Tensor

    def __init__(self, embedding_dim: int, head_dim: int, context_size: int) -> None:
        super().__init__()
        self.head_dim = head_dim

        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(context_size, context_size), diagonal=1).bool(),
        )

        # the linear layer's shape is (head_dim, embedding_dim). this head learns a projection from dim size embedding_dim to head_dim
        self.W_q = nn.Linear(
            in_features=embedding_dim, out_features=head_dim, bias=False
        )
        self.W_k = nn.Linear(
            in_features=embedding_dim, out_features=head_dim, bias=False
        )
        self.W_v = nn.Linear(
            in_features=embedding_dim, out_features=head_dim, bias=False
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Map [batch, sequence, embedding_dim] to [batch, sequence, head_dim]."""
        Q, K, V = self.W_q(X), self.W_k(X), self.W_v(X)

        attention_scores = (Q @ K.transpose(-2, -1)) / (self.head_dim**0.5)

        num_tokens = X.size(1)
        mask = self.causal_mask[:num_tokens, :num_tokens]

        attention_scores = attention_scores.masked_fill(mask, float("-inf"))

        attention_weights = torch.softmax(attention_scores, dim=-1)
        output = attention_weights @ V

        return output


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism that allows the model to focus on different parts of the input sequence.
    """

    def __init__(self, embedding_dim: int, num_heads: int, context_size: int) -> None:
        super().__init__()

        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")

        head_dim = embedding_dim // num_heads

        self.heads = nn.ModuleList(
            [
                AttentionHead(
                    embedding_dim=embedding_dim,
                    head_dim=head_dim,
                    context_size=context_size,
                )
                for _ in range(num_heads)
            ]
        )

        self.W_o = nn.Linear(
            in_features=embedding_dim, out_features=embedding_dim, bias=False
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Apply causal attention, preserving [batch, sequence, embedding_dim]."""
        output = torch.cat([head(X) for head in self.heads], dim=-1)
        return self.W_o(output)


class MultiLayerPerceptron(nn.Module):
    """
    A multilayer perceptron feedforward neural network.
    Used as the feedforward component within a transformer block.
    """

    def __init__(self, embedding_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features=embedding_dim, out_features=hidden_dim)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(in_features=hidden_dim, out_features=embedding_dim)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Transform each token, preserving [batch, sequence, embedding_dim]."""
        return self.fc2(self.activation(self.fc1(X)))


class LayerNorm(nn.Module):
    """
    A normalization layer to keep activations well-behaved;
    i.e., to keep activations on a stable scale and distribution.
    """

    def __init__(self, embedding_dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(embedding_dim))
        self.bias = nn.Parameter(torch.zeros(embedding_dim))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Normalize the last dimension of [..., embedding_dim], preserving shape."""
        mean = X.mean(dim=-1, keepdim=True)
        variance = X.var(dim=-1, correction=0, keepdim=True)
        normalized_input = (X - mean) * torch.rsqrt(variance + self.eps)

        return self.weight * normalized_input + self.bias


class TransformerBlock(nn.Module):
    """
    A pre-norm transformer block with causal attention, an MLP, and residual connections.
    """

    def __init__(
        self,
        embedding_dim: int,
        context_size: int,
        num_attention_heads: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.attention = MultiHeadAttention(
            embedding_dim=embedding_dim,
            num_heads=num_attention_heads,
            context_size=context_size,
        )
        self.norm_1 = LayerNorm(embedding_dim=embedding_dim)
        self.mlp = MultiLayerPerceptron(
            embedding_dim=embedding_dim, hidden_dim=hidden_dim
        )
        self.norm_2 = LayerNorm(embedding_dim=embedding_dim)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Apply attention and MLP residuals, preserving [batch, sequence, embedding_dim]."""
        X = X + self.attention(self.norm_1(X))
        X = X + self.mlp(self.norm_2(X))
        return X


class GPT(nn.Module):
    """
    A decoder-only transformer that maps token IDs to next-token vocabulary logits.
    """

    def __init__(
        self,
        embedding_dim: int,
        vocab_size: int,
        context_size: int,
        num_layers: int,
        num_attention_heads: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.embeddings = Embeddings(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            context_size=context_size,
        )
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embedding_dim=embedding_dim,
                    context_size=context_size,
                    num_attention_heads=num_attention_heads,
                    hidden_dim=hidden_dim,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = LayerNorm(embedding_dim=embedding_dim)
        self.lm_head = nn.Linear(
            in_features=embedding_dim, out_features=vocab_size, bias=False
        )

        self.lm_head.weight = self.embeddings.token_embeddings.weight # We do this to reduce the total number of parameters


    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Map token IDs [batch, sequence] to vocabulary logits [batch, sequence, vocab_size]."""
        X = self.embeddings(token_ids)
        for layer in self.layers:
            X = layer(X)

        X = self.final_norm(X)
        return self.lm_head(X)
