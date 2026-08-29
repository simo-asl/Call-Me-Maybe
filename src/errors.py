"""Project-specific exceptions shown as clear command-line errors."""


class CallMeMaybeError(Exception):
    """Base class for expected application errors."""


class InputError(CallMeMaybeError):
    """Raised when an input file cannot be read or does not match its schema."""


class GenerationError(CallMeMaybeError):
    """Raised when constrained decoding cannot produce a valid call."""
