import pytest
import torch

from dgpt.loss import cross_entropy
from dgpt.model import (
    GPT,
    AttentionHead,
    Embeddings,
    LayerNorm,
    MultiHeadAttention,
    MultiLayerPerceptron,
    TransformerBlock,
)


def test_embedding_forward_shape():
    vocab_size, embedding_dim, context_size, batch_size = 256, 256, 128, 1
    embeddings = Embeddings(vocab_size=vocab_size, embedding_dim=embedding_dim, context_size=context_size)
    text = "The reality of which we can speak is never reality in itself"
    token_ids = torch.tensor([ord(c) for c in text], dtype=torch.long).unsqueeze(0)

    sequence_len = token_ids.shape[1]
    output = embeddings(token_ids)

    assert output.shape[0] == batch_size
    assert output.shape[1] == sequence_len
    assert output.shape[2] == embedding_dim


def test_embedding_forward_result():
    vocab_size, embedding_dim, context_size = 256, 256, 128
    embeddings = Embeddings(vocab_size=vocab_size, embedding_dim=embedding_dim, context_size=context_size)
    text = "The reality of which we can speak is never reality in itself"
    token_ids = torch.tensor([ord(c) for c in text], dtype=torch.long).unsqueeze(0)

    token_embeddings = embeddings.token_embeddings(token_ids)
    position_ids = torch.arange(token_ids.size(1), device=token_ids.device).unsqueeze(0)
    position_embeddings = embeddings.position_embeddings(position_ids)

    expected = token_embeddings + position_embeddings
    actual = embeddings(token_ids)

    assert torch.equal(expected, actual)


def test_attention_head_forward_shape():
    embedding_dim, head_dim, context_size, batch_size = 256, 64, 128, 1
    attention_head = AttentionHead(embedding_dim=embedding_dim, head_dim=head_dim, context_size=context_size)
    text = "The reality of which we can speak is never reality in itself"
    token_ids = torch.tensor([ord(c) for c in text], dtype=torch.long).unsqueeze(0)

    embeddings = Embeddings(vocab_size=256, embedding_dim=embedding_dim, context_size=context_size)
    input_embeddings = embeddings(token_ids)

    output = attention_head(input_embeddings)

    assert output.shape[0] == batch_size
    assert output.shape[1] == token_ids.shape[1]
    assert output.shape[2] == head_dim


def test_attention_head_causality():
    embedding_dim, head_dim, context_size = 256, 64, 128
    attention_head = AttentionHead(embedding_dim=embedding_dim, head_dim=head_dim, context_size=context_size)
    text_1 = "The reality of which we can speak is never reality in itself"
    text_2 = "The reality of which we can speak is always reality as far as we can tell"
    common_prefix = "The reality of which we can speak is "

    token_ids_1 = torch.tensor([ord(c) for c in text_1], dtype=torch.long).unsqueeze(0)
    token_ids_2 = torch.tensor([ord(c) for c in text_2], dtype=torch.long).unsqueeze(0)
    prefix_token_ids = torch.tensor([ord(c) for c in common_prefix], dtype=torch.long).unsqueeze(0)

    prefix_length = prefix_token_ids.shape[1]

    embeddings = Embeddings(vocab_size=256, embedding_dim=embedding_dim, context_size=context_size)

    input_embeddings_1 = embeddings(token_ids_1)
    input_embeddings_2 = embeddings(token_ids_2)

    output_1 = attention_head(input_embeddings_1)
    output_2 = attention_head(input_embeddings_2)

    assert torch.allclose(output_1[:, :prefix_length, :], output_2[:, :prefix_length, :], atol=1e-6) # The outputs for the common prefix should be identical


def test_multihead_attention_forward_shape():
    embedding_dim, num_heads, context_size = 256, 4, 128
    multihead_attention = MultiHeadAttention(embedding_dim=embedding_dim, num_heads=num_heads, context_size=context_size)
    text = "The occipital lobe, located in the back of the brain, decodes visual signals."
    token_ids = torch.tensor([ord(c) for c in text], dtype=torch.long).unsqueeze(0)

    embeddings = Embeddings(vocab_size=256, embedding_dim=embedding_dim, context_size=context_size)
    input_embeddings = embeddings(token_ids)

    output = multihead_attention(input_embeddings)

    assert output.shape[0] == 1
    assert output.shape[1] == token_ids.shape[1]
    assert output.shape[2] == embedding_dim


def test_multihead_attention_causality():
    embedding_dim, num_heads, context_size = 256, 4, 128
    multihead_attention = MultiHeadAttention(embedding_dim=embedding_dim, num_heads=num_heads, context_size=context_size)
    text1 = "The occipital lobe, located in the back of the brain, decodes visual signals."
    text2 = "The occipital lobe, located in the back of the brain, does nothing of importance."
    common_prefix = "The occipital lobe, located in the back of the brain, "

    token_ids_1 = torch.tensor([ord(c) for c in text1], dtype=torch.long).unsqueeze(0)
    token_ids_2 = torch.tensor([ord(c) for c in text2], dtype=torch.long).unsqueeze(0)
    prefix_token_ids = torch.tensor([ord(c) for c in common_prefix], dtype=torch.long).unsqueeze(0)

    prefix_length = prefix_token_ids.shape[1]

    embeddings = Embeddings(vocab_size=256, embedding_dim=embedding_dim, context_size=context_size)

    input_embeddings_1 = embeddings(token_ids_1)
    input_embeddings_2 = embeddings(token_ids_2)

    output_1 = multihead_attention(input_embeddings_1)
    output_2 = multihead_attention(input_embeddings_2)

    assert torch.allclose(output_1[:, :prefix_length, :], output_2[:, :prefix_length, :], atol=1e-6) # The outputs for the common prefix should be identical


def test_multihead_attention_invalid_num_heads():
    embedding_dim, num_heads, context_size = 256, 5, 128
    try:
        MultiHeadAttention(embedding_dim=embedding_dim, num_heads=num_heads, context_size=context_size)
        assert False, "Expected ValueError due to embedding_dim not being divisible by num_heads"
    except ValueError as e:
        assert str(e) == "embedding_dim must be divisible by num_heads"


def test_multi_layer_perceptron_forward_shape():
    embedding_dim, hidden_dim, context_size = 256, 64, 128

    multi_layer_perceptron = MultiLayerPerceptron(embedding_dim=embedding_dim, hidden_dim=hidden_dim)

    text = "Two vectors are orthogonal if their dot product is zero"
    token_ids = torch.tensor([ord(c) for c in text], dtype=torch.long).unsqueeze(0)

    embeddings = Embeddings(vocab_size=256, embedding_dim=embedding_dim, context_size=context_size)
    input_embeddings = embeddings(token_ids)

    output = multi_layer_perceptron(input_embeddings)

    assert output.shape == input_embeddings.shape


def test_multi_layer_perceptron_backward_result():
    batch_size, sequence_length, embedding_dim, hidden_dim = 2, 4, 16, 64

    multi_layer_perceptron = MultiLayerPerceptron(embedding_dim=embedding_dim, hidden_dim=hidden_dim)
    X = torch.randn(batch_size, sequence_length, embedding_dim, requires_grad=True)
    output = multi_layer_perceptron(X)
    loss = torch.mean(output)

    loss.backward()

    assert X.grad is not None
    assert torch.isfinite(X.grad).all()

    for param in multi_layer_perceptron.parameters():
        assert param.grad is not None
        assert torch.isfinite(param.grad).all()
        assert param.grad.shape == param.shape


@pytest.mark.parametrize("constant_input", [False, True])
def test_layer_norm_forward(constant_input):
    torch.manual_seed(0)
    embedding_dim = 16
    layer = LayerNorm(embedding_dim).double()
    reference = torch.nn.LayerNorm(embedding_dim).double()

    with torch.no_grad():
        layer.weight.normal_()
        layer.bias.normal_()
    reference.load_state_dict(layer.state_dict())

    X = torch.randn(2, 5, embedding_dim, dtype=torch.float64)
    if constant_input:
        X.fill_(3.0)

    actual = layer(X)
    expected = reference(X)

    assert actual.shape == X.shape
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-9)


@pytest.mark.parametrize("constant_input", [False, True])
def test_layer_norm_backward(constant_input):
    torch.manual_seed(0)
    embedding_dim = 16
    layer = LayerNorm(embedding_dim).double()
    reference = torch.nn.LayerNorm(embedding_dim).double()

    with torch.no_grad():
        layer.weight.normal_()
        layer.bias.normal_()
    reference.load_state_dict(layer.state_dict())

    X = torch.randn(2, 5, embedding_dim, dtype=torch.float64)
    if constant_input:
        X.fill_(3.0)

    X.requires_grad_()
    reference_X = X.detach().clone().requires_grad_()

    actual = layer(X)
    expected = reference(reference_X)

    upstream_gradient = torch.randn_like(actual)
    actual.backward(upstream_gradient)
    expected.backward(upstream_gradient)

    gradient_pairs = [
        (X.grad, reference_X.grad),
        (layer.weight.grad, reference.weight.grad),
        (layer.bias.grad, reference.bias.grad),
    ]

    for actual_grad, expected_grad in gradient_pairs:
        assert actual_grad is not None
        assert expected_grad is not None
        assert torch.isfinite(actual_grad).all()
        torch.testing.assert_close(
            actual_grad, expected_grad, atol=1e-10, rtol=1e-9
        )


def test_transformer_block_forward_shape():
    embedding_dim, num_heads, hidden_dim, context_size = 256, 4, 512, 128
    transformer_block = TransformerBlock(
        embedding_dim=embedding_dim,
        num_attention_heads=num_heads,
        hidden_dim=hidden_dim,
        context_size=context_size,
    )

    text = "The Articles of Confederation was the first US constitution."
    token_ids = torch.tensor([ord(c) for c in text], dtype=torch.long).unsqueeze(0)

    embeddings = Embeddings(vocab_size=256, embedding_dim=embedding_dim, context_size=context_size)
    input_embeddings = embeddings(token_ids)

    output = transformer_block(input_embeddings)

    assert output.shape[0] == 1
    assert output.shape[1] == token_ids.shape[1]
    assert output.shape[2] == embedding_dim


def test_transformer_block_residual_identity():
    transformer_block = TransformerBlock(
        embedding_dim=16,
        context_size=8,
        num_attention_heads=4,
        hidden_dim=64,
    )

    with torch.no_grad():
        for param in transformer_block.attention.parameters():
            param.zero_()
        for param in transformer_block.mlp.parameters():
            param.zero_()

    X = torch.randn(2, 5, 16)

    torch.testing.assert_close(
        transformer_block(X), X, atol=0, rtol=0
    )


def test_gpt_forward_shape():
    batch_size, embedding_dim, vocab_size = 2, 128, 256
    model = GPT(
        embedding_dim=embedding_dim,
        vocab_size=vocab_size,
        context_size=128,
        num_layers=2,
        num_attention_heads=1,
        hidden_dim=10
        )

    text = "The nervous system can be thought of as your body's electrical wiring"
    token_ids = torch.tensor(
        [ord(c) for c in text],
        dtype=torch.long,
        ).unsqueeze(0).repeat(batch_size, 1)

    sequence_len = token_ids.shape[1]

    output = model(token_ids)

    assert output.shape[0] == batch_size
    assert output.shape[1] == sequence_len
    assert output.shape[2] == vocab_size


def test_gpt_backward():
    torch.manual_seed(0)
    vocab_size = 256
    model = GPT(
        embedding_dim=16,
        vocab_size=vocab_size,
        context_size=128,
        num_layers=2,
        num_attention_heads=4,
        hidden_dim=64,
    )

    texts = [
        "Over the river",
        "Through the woods",
        "To grandmother's house we go",
    ]
    max_length = max(len(text) for text in texts)

    token_ids = torch.tensor(
        [
            [ord(c) for c in text.ljust(max_length)]
            for text in texts
        ],
        dtype=torch.long,
    )

    inputs = token_ids[:, :-1]
    targets = token_ids[:, 1:]

    loss = cross_entropy(model(inputs), targets)

    assert torch.isfinite(loss)

    loss.backward()

    for name, parameter in model.named_parameters():
        assert parameter.grad is not None
        assert parameter.grad.shape == parameter.shape
        assert torch.isfinite(parameter.grad).all()


def test_embedding_and_lm_head_weight_tying():
        model = GPT(
            embedding_dim=16,
            vocab_size=256,
            context_size=128,
            num_layers=2,
            num_attention_heads=4,
            hidden_dim=64,
        )

        assert model.embeddings.token_embeddings.weight is model.lm_head.weight