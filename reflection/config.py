"""Configuration constants for the reflection workflow."""

import os


# Constants
MAX_REVISIONS = int(os.environ.get("MAX_REVISIONS", 3))
DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "gpt-5")
DEFAULT_TEMPERATURE = float(os.environ.get("DEFAULT_TEMPERATURE", 0.9))
MAX_REVISIONS_MESSAGE = os.environ.get(
    "MAX_REVISIONS_MESSAGE", 
    "MAX REVISIONS REACHED - APPROVING CURRENT VERSION"
)
