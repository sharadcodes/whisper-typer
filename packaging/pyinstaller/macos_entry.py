"""
PyInstaller entrypoint for macOS app bundle.

We keep this in a dedicated file so packaging is stable even if the
project's CLI entrypoints change.
"""

from whisper_typer.client_ui import main


if __name__ == "__main__":
    main()

