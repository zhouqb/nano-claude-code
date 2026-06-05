"""Dataset adapter registry.

Add a new benchmark by implementing ``DatasetAdapter`` and registering it here.
Adapters are constructed lazily so importing this package never pulls in heavy
optional dependencies (``datasets``, ``swebench``) until a dataset is used.
"""

from __future__ import annotations

from collections.abc import Callable

from evals.datasets.base import DatasetAdapter

_FACTORIES: dict[str, Callable[[], DatasetAdapter]] = {}


def register(name: str, factory: Callable[[], DatasetAdapter]) -> None:
    _FACTORIES[name] = factory


def available() -> list[str]:
    return sorted(_FACTORIES)


def get_adapter(name: str) -> DatasetAdapter:
    try:
        return _FACTORIES[name]()
    except KeyError:
        raise KeyError(f"Unknown dataset {name!r}. Available: {', '.join(available())}") from None


def _swe_bench_lite() -> DatasetAdapter:
    from evals.datasets.swe_bench_lite import SweBenchLiteAdapter

    return SweBenchLiteAdapter()


def _swe_bench_verified() -> DatasetAdapter:
    from evals.datasets.swe_bench_lite import SweBenchVerifiedAdapter

    return SweBenchVerifiedAdapter()


register("swe-bench-lite", _swe_bench_lite)
register("swe-bench-verified", _swe_bench_verified)
