"""Plugin Architecture for TrainingEngine v2."""

from typing import Any


class BasePlugin:
    """Base interface for TrainingEngine v2 Plugins."""

    def name(self) -> str:
        return self.__class__.__name__

    def register(self, engine: Any) -> None:
        """Register event handlers, metrics, or hooks with the engine."""
        raise NotImplementedError
