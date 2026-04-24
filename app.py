"""Root entrypoint for serving the MEVerse OpenEnv app."""

from meverse.server.app import app, main

__all__ = ["app", "main"]


if __name__ == "__main__":
    main()
