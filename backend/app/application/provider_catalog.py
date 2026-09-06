"""Registries for pluggable training and sim2sim providers.

The application layer selects providers by stable task/robot metadata.  The
registries deliberately store opaque provider objects so importing Isaac,
MuJoCo, or a vendor SDK remains a runtime concern of the adapter package.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


class ProviderRegistryError(ValueError):
    """Raised when a provider cannot be registered or resolved."""


class _ProviderRegistry:
    provider_kind = "provider"

    def __init__(self, providers: Iterable[Any] = ()) -> None:
        self._providers: dict[str, Any] = {}
        for provider in providers:
            self.register(provider)

    def register(self, provider: Any) -> Any:
        name = str(getattr(provider, "name", "")).strip()
        if not name:
            raise ProviderRegistryError(f"{self.provider_kind} must expose a non-empty name")
        if name in self._providers:
            raise ProviderRegistryError(f"duplicate {self.provider_kind}: {name}")
        self._providers[name] = provider
        return provider

    def get(self, name: str | None) -> Any | None:
        if name is None:
            return None
        return self._providers.get(str(name).strip())

    def require(self, name: str) -> Any:
        provider = self.get(name)
        if provider is None:
            raise ProviderRegistryError(f"unknown {self.provider_kind}: {name}")
        return provider

    def contains(self, name: str) -> bool:
        return str(name).strip() in self._providers

    def list(self) -> list[Any]:
        return list(self._providers.values())

    def names(self) -> tuple[str, ...]:
        return tuple(self._providers)


class TrainingProviderRegistry(_ProviderRegistry):
    """Resolve a training/play/export implementation by provider name."""

    provider_kind = "training provider"


class Sim2SimRegistry(_ProviderRegistry):
    """Resolve a simulator adapter by the RobotSpec adapter identifier."""

    provider_kind = "sim2sim adapter"


__all__ = ["ProviderRegistryError", "Sim2SimRegistry", "TrainingProviderRegistry"]
