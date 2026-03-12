"""Projection Observatory Dashboard — interactive Panel application."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Callable


def create_app() -> object:
    """Lazy import and create the Panel application.

    The actual ``app`` module is imported at call time so that the
    dashboard sub-package can be imported without requiring Panel or
    the ``app`` module to exist yet (it is part of a later build step).
    """
    from .app import create_app as _create_app

    return _create_app()


__all__ = ["create_app"]
