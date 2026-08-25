"""Public contract package.

The backend domain module is the implementation source for this initial P0
slice; this package provides the stable shared import surface used by future
web/type-generation tooling without exposing infrastructure dependencies.
"""

from backend.app.domain.contracts import *  # noqa: F401,F403

