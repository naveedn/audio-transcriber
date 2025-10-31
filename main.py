"""Lightweight entrypoint for local debugging."""

from src.main import cli


def main() -> None:
    """Run the CLI entrypoint when invoked as a script."""
    cli.main(prog_name="transcribe")


if __name__ == "__main__":
    main()
