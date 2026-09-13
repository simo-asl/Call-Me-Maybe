"""Small wrapper that applies token masks to raw LLM logits."""

import numpy as np
from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]
from pydantic import BaseModel, PrivateAttr

from src.token_encoder import Encoder


class LLM(BaseModel):
    """Expose constrained next-token and option selection over the SDK."""

    _llm: Small_LLM_Model = PrivateAttr()
    _encoder: Encoder = PrivateAttr()
    _instruction_tokens: list[int] | None = PrivateAttr()

    def __init__(self, llm: Small_LLM_Model, encoder: Encoder):
        """Store the SDK model and encoder used for constrained decoding."""
        super().__init__()
        self._llm = llm
        self._encoder = encoder
        self._instruction_tokens = None
        print('LLM created.')

    def next_token(
        self,
        tokens: list[int],
        mask: set[int] | None = None,
    ) -> int:
        """Return the highest-logit token among the allowed tokens."""
        logits = self.get_logits(tokens, mask)
        return int(np.argmax(logits))

    def next_option(
        self,
        tokens: list[int],
        options: list[list[int]],
        terminator: list[int] | None = None,
    ) -> list[int]:
        """Constrain generation to one complete token sequence in options."""
        result: list[int] = []
        context = tokens.copy()
        remaining = [
            option + (terminator or [])
            for option in options
        ]

        while remaining:
            allowed = {option[0] for option in remaining}
            token = self.next_token(context, allowed)
            result.append(token)
            context.append(token)
            remaining = [
                option[1:]
                for option in remaining
                if option[0] == token and len(option) > 1
            ]

        if terminator:
            return result[:-len(terminator)]
        return result

    def set_instruction(self, instruction: list[int] | str) -> None:
        """Set the system/tool instruction prepended to model context."""
        if isinstance(instruction, str):
            instruction = self._encoder.encode(instruction)
        self._instruction_tokens = instruction

    def get_logits(
        self,
        tokens: list[int],
        mask: set[int] | None = None,
    ) -> list[float]:
        """Get next-token logits and optionally mask disallowed token ids."""
        instruction = self._instruction_tokens or []
        logits = self._llm.get_logits_from_input_ids(instruction + tokens)
        if mask is not None:
            logits = self._apply_mask(mask, logits)
        return [float(value) for value in logits]

    @staticmethod
    def _apply_mask(
        mask: set[int] | list[int],
        logits: list[float],
    ) -> list[float]:
        """Replace logits of disallowed tokens with negative infinity."""
        masked = np.full_like(logits, -float('inf'))
        for token_id in mask:
            masked[token_id] = logits[token_id]
        return list(masked)

    @property
    def encoder(self) -> Encoder:
        """Return the encoder associated with this model wrapper."""
        return self._encoder
