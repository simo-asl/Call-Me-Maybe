"""Token encoding helpers used by the constrained decoder."""

from typing import Any
from pydantic import BaseModel, PrivateAttr


class Encoder(BaseModel):
    """Encode and decode text using the model vocabulary."""

    _trie: dict[str, Any] = PrivateAttr()
    _vocab: list[str | None] = PrivateAttr()

    def __init__(self, tokens: dict[str, int]):
        """Build a trie and reverse vocabulary from token mappings."""
        vocab: list[str | None] = [None] * len(tokens)
        trie: dict[str, Any] = {}
        print('Encoder: Building trie and vocab...')
        for word, token in tokens.items():
            vocab[token] = word
            node = trie
            for char in word:
                node = node.setdefault(char, {})
            node['token'] = token
        super().__init__()
        self._trie = trie
        self._vocab = vocab
        print('Encoder created.')

    def encode(self, text: str) -> list[int]:
        """Translate text to token ids using longest vocabulary matches."""
        text = standard_to_special(text)
        ids: list[int] = []
        index = 0
        while index < len(text):
            node = self._trie
            match_id = None
            match_len = -1
            cursor = index
            while cursor < len(text) and text[cursor] in node:
                node = node[text[cursor]]
                cursor += 1
                if 'token' in node:
                    match_id = node['token']
                    match_len = cursor - index
            if match_id is not None:
                ids.append(match_id)
                index += match_len
            else:
                index += 1
        return ids

    def decode(self, tokens: list[int] | int) -> str:
        if isinstance(tokens, int):
            text = self._vocab[tokens] or ''
        else:
            text = ''.join(
                self._vocab[token] or ''
                for token in tokens
            )

        return special_to_standard(text)


def special_to_standard(text: str) -> str:
    """Convert tokenizer-specific whitespace markers to normal text."""
    return text.replace('Ġ', ' ').replace('Ċ', '\n').replace('ĉ', '\t')


def standard_to_special(text: str) -> str:
    """Convert normal whitespace to tokenizer-specific markers."""
    return text.replace(' ', 'Ġ').replace('\n', 'Ċ').replace('\t', 'ĉ')
