from __future__ import annotations

import json
from pathlib import Path


class CharTokenizer:
    """Character-level tokenizer; the vocabulary is the sorted set of characters."""

    def __init__(self, chars: list[str]) -> None:
        if len(set(chars)) != len(chars):
            raise ValueError("duplicate characters in vocabulary")
        self.chars = list(chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        return cls(sorted(set(text)))

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, s: str) -> list[int]:
        unknown = sorted({c for c in s if c not in self.stoi})
        if unknown:
            raise ValueError(f"characters not in vocabulary: {unknown!r}")
        return [self.stoi[c] for c in s]

    def decode(self, ids) -> str:
        return "".join(self.itos[int(i)] for i in ids)

    def token_str(self, i: int) -> str:
        """Printable form of one token (escapes whitespace so it shows up in plots)."""
        return repr(self.itos[int(i)])[1:-1]

    def to_dict(self) -> dict:
        return {"type": "char", "chars": self.chars}

    @classmethod
    def from_dict(cls, d: dict) -> "CharTokenizer":
        if d.get("type") != "char":
            raise ValueError(f"unsupported tokenizer type: {d.get('type')!r}")
        return cls(d["chars"])

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
