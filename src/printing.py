"""Simple colored terminal visualization."""


class Colors:
    RESET = "\033[0m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"


class GenerationVisualizer:
    """Display generation information in the terminal."""

    def show_step(
        self,
        step: int,
        token: str,
        allowed: int,
        vocab_size: int,
        current: str,
    ) -> None:
        """Print one generation step."""
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
