"""Colored terminal output for generation diagnostics."""


_VISUAL_ENABLED = False
_DEBUG_TOKENS_ENABLED = False


class Colors:
    """ANSI escape sequences used by terminal diagnostics.

    Attributes:
        RESET: Escape sequence that clears active styles.
        BOLD: Escape sequence for bold text.
        GREEN: Escape sequence for green text.
        YELLOW: Escape sequence for yellow text.
        CYAN: Escape sequence for cyan text.
        MAGENTA: Escape sequence for magenta text.
    """

    RESET = "\033[0m"
    BOLD = "\033[1m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    MAGENTA = "\033[95m"


def configure_output(*, visual: bool, debug_tokens: bool) -> None:
    """Configure the optional terminal diagnostic mode.

    Args:
        visual: Whether to show constrained-generation steps.
        debug_tokens: Whether to show token-selection diagnostics.
    """

    global _VISUAL_ENABLED, _DEBUG_TOKENS_ENABLED
    _VISUAL_ENABLED = visual
    _DEBUG_TOKENS_ENABLED = debug_tokens


def show_token_debug(
    context_size: int,
    allowed: int,
    vocab_size: int,
    token_id: int,
    token: str,
) -> None:
    """Display one compact token-selection diagnostic block.

    Args:
        context_size: Number of token IDs in the active model context.
        allowed: Number of vocabulary IDs permitted by the mask.
        vocab_size: Total number of logits returned by the model.
        token_id: Vocabulary ID selected after masking the logits.
        token: Vocabulary text associated with the selected ID.
    """

    if not _DEBUG_TOKENS_ENABLED:
        return

    print(
        f"\n{Colors.BOLD}{Colors.MAGENTA}Token Debug{Colors.RESET}"
    )
    print(
        f"{Colors.YELLOW}  Context tokens :{Colors.RESET} {context_size}"
    )
    print(
        f"{Colors.GREEN}  Allowed tokens :{Colors.RESET} "
        f"{allowed}/{vocab_size}"
    )
    print(
        f"{Colors.CYAN}  Selected ID    :{Colors.RESET} {token_id}"
    )
    print(
        f"{Colors.CYAN}  Token text     :{Colors.RESET} {token!r}"
    )


class GenerationVisualizer:
    """Display constrained-generation steps when visualization is enabled."""

    def show_step(
        self,
        step: int,
        token: str,
        allowed: int,
        vocab_size: int,
        current: str,
    ) -> None:
        """Print details about one selected generation token.

        Args:
            step: One-based generation step number.
            token: Text fragment selected at this step.
            allowed: Number of vocabulary tokens permitted by the mask.
            vocab_size: Total number of vocabulary tokens.
            current: Function-call JSON generated after selecting the token.
        """

        if not _VISUAL_ENABLED:
            return

        safe_token = repr(token)

        print(
            f"\n{Colors.BOLD}{Colors.YELLOW}Step {step}{Colors.RESET}"
        )
        print(
            f"{Colors.GREEN}Allowed tokens:"
            f" {allowed}/{vocab_size}"
            f"{Colors.RESET}"
        )
        print(
            f"{Colors.CYAN}Selected:"
            f" {safe_token}"
            f"{Colors.RESET}"
        )
        print(f"{Colors.CYAN}{current}{Colors.RESET}")
