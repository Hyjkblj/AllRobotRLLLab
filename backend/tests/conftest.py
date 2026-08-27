"""Keep process-wide API fixtures isolated from a developer runtime tree."""

from __future__ import annotations

import os
import tempfile


# The application is imported at module collection time.  Give the default
# Local File Mode a throw-away root for the test process while preserving an
# explicitly configured runtime for integration tests.
if "ROBOTLAB_RUNTIME_DIR" not in os.environ:
    os.environ["ROBOTLAB_RUNTIME_DIR"] = tempfile.mkdtemp(prefix="allrobotrl-tests-")
