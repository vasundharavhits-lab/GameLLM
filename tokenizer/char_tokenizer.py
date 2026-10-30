
class CharTokenizer:
    """A simple tokenizer that maps each character to an integer."""

    def __init__(self, text: str):
        self.chars = sorted(set(text))
        self.stoi = {char: index for index, char in enumerate(self.chars)}
        self.itos = {index: char for char, index in self.stoi.items()}

    @property
    def vocab_size(self):
        return len(self.chars)

    def encode(self, text: str) -> list[int]:
        unknown = sorted(set(text) - set(self.stoi))
        if unknown:
            raise ValueError(f"Unknown characters: {unknown!r}")
        return [self.stoi[char] for char in text]

    def decode(self, token_ids: list[int]) -> str:
        return "".join(self.itos[index] for index in token_ids)
