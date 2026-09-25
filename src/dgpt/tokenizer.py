class ByteTokenizer:
    def encode(self, text: str) -> list[int]:
        """
        Encodes a string into a list of byte values (0-255).
        """
        return list(text.encode("utf-8"))

    def decode(self, byte_values: list[int]) -> str:
        """
        Decodes a list of byte values (0-255) back into a string.
        """
        return bytes(byte_values).decode("utf-8")