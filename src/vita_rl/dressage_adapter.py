"""Boundary for a future Dressage/slime integration.

Dressage and slime remain external dependencies. This module deliberately does
not import or modify either project yet.
"""


def create_adapter(*args, **kwargs):
    """Fail explicitly until the first RL integration is designed."""

    del args, kwargs
    raise NotImplementedError("Dressage integration is not implemented yet")
