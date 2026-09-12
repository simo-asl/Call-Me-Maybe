"""Token encoding helpers used by the constrained decoder."""

import re
from typing import Any

from pydantic import BaseModel, PrivateAttr


WORD_PATTERN = re.compile(r'''
    "(?:\\.|[^"])*"   |
    '(?:\\.|[^'])*'   |
    \S+
''', re.VERBOSE)


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

    def encode_words(self, text: str) -> set[int]:
        """Return all token ids found in prompt words."""
        ids: set[int] = set()
        words = WORD_PATTERN.findall(text)
        for word in words:
            word = word.strip('.,!?').strip('"\'')
            if not word:
                continue
            ids.update(self.encode(word))
            ids.add(self.encode(' ' + word)[0])
        return ids

    def encode_words_separated(self, text: str) -> list[list[int]]:
        """Return separately tokenized candidate values from a prompt."""
        ids: list[list[int]] = []
        colon_match = re.search(r':\s*(.+)$', text)
        if colon_match:
            ids.append(self.encode(colon_match.group(1).strip()))

        unescaped = text.replace('\\"', '"')
        for part in WORD_PATTERN.findall(unescaped):
            part = part.strip('".,!?:;\\').strip("'")
            if part:
                ids.append(self.encode(part))
        return ids

    def decode(self, tokens: list[int] | int) -> str:
        """Translate token ids back to readable text."""
        if isinstance(tokens, int):
            return self._vocab[tokens] or ''
        return special_to_standard(
            ''.join(self._vocab[token] or '' for token in tokens)
        )


def special_to_standard(text: str) -> str:
    """Convert tokenizer-specific whitespace markers to normal text."""
    return text.replace('Ġ', ' ').replace('Ċ', '\n').replace('ĉ', '\t')


def standard_to_special(text: str) -> str:
    """Convert normal whitespace to tokenizer-specific markers."""
    return text.replace(' ', 'Ġ').replace('\n', 'Ċ').replace('\t', 'ĉ')
