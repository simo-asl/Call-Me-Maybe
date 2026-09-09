"""Simple colored terminal visualization."""


class Colors:
    """ANSI escape sequences used by the generation visualizer.

    Attributes:
        RESET: Escape sequence that clears active styles.
        GREEN: Escape sequence for green text.
        YELLOW: Escape sequence for yellow text.
        CYAN: Escape sequence for cyan text.
    """
    RESET = "\033[0m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"


class GenerationVisualizer:
    """Display each constrained-generation step in the terminal."""

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
        safe_token = repr(token)

        print(f"\n{Colors.YELLOW}Step {step}{Colors.RESET}")

        print(
            f"{Colors.GREEN}Allowed tokens:"
            f" {allowed}/{vocab_size}"
            f"{Colors.RESET}")

        print(
            f"{Colors.CYAN}Selected:"
            f" {safe_token}"
            f"{Colors.RESET}")

        print(f"{Colors.CYAN}{current}{Colors.RESET}")
