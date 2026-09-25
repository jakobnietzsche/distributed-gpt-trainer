from dgpt.tokenizer import ByteTokenizer


def test_encode_ascii():
    tokenizer = ByteTokenizer()
    text = "If I have seen further it is by standing on the shoulders of Giants."
    
    expected_bytes = [ord(c) for c in text]

    assert tokenizer.encode(text) == expected_bytes


def test_back_and_forth_tokenization():
    tokenizer = ByteTokenizer()
    text = "Passion🙂is the genesis of 🔴genius."

    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)

    assert decoded == text