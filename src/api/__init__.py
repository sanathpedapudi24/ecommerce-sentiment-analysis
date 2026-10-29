"""Flask REST API package."""

from .app import create_app, get_wsgi_app

__all__ = ["create_app", "get_wsgi_app"]
