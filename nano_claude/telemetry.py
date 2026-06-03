"""OpenTelemetry wiring: opt-in traces + logs for the agent loop.

The OpenTelemetry *API* (``opentelemetry-api``) is a core dependency, so the
instrumentation sprinkled through the loop (``tracer.start_as_current_span``,
``log.info``) is always importable. Until :func:`init_telemetry` installs a real
provider, ``tracer`` is a no-op proxy and ``log`` records go nowhere — so the
instrumentation is effectively free when telemetry is off.

The heavyweight *SDK* and OTLP exporter live in the optional ``otel`` extra and
are imported lazily by :func:`init_telemetry`, which only installs a provider
when ``NANO_CLAUDE_TELEMETRY`` is set::

    NANO_CLAUDE_TELEMETRY=1 nano-claude                       # traces via OTLP/HTTP
    NANO_CLAUDE_TELEMETRY=1 NANO_CLAUDE_TELEMETRY_TRACES=off nano-claude  # logs only, no backend

Signals are split by destination, because the REPL owns the console:

* **Traces** → ``NANO_CLAUDE_TELEMETRY_TRACES``: ``otlp`` (default; point at
  Jaeger or a collector), ``console`` (debugging only), or ``off`` (no exporter,
  so nothing to run a backend for — handy for logs-only). ``console`` can also
  be selected with the legacy ``NANO_CLAUDE_TELEMETRY_CONSOLE=1``.
* **Logs** → a per-session file ``<session-id>.log.jsonl`` beside the session
  JSONL, so log output never mixes into the chat UI. Set
  ``NANO_CLAUDE_TELEMETRY_OTLP_LOGS=1`` to ship them to a collector instead.

Standard ``OTEL_*`` environment variables (``OTEL_EXPORTER_OTLP_ENDPOINT``,
``OTEL_EXPORTER_OTLP_HEADERS``, ``OTEL_SERVICE_NAME``, …) are honoured by the SDK
and exporters directly, so the collector endpoint is configured the usual way.
"""

from __future__ import annotations

import logging
import os
import warnings
from pathlib import Path
from typing import Any

from opentelemetry import trace

from nano_claude import __version__

# Always-available API handles. With no provider installed these are no-ops:
# span creation is cheap and log records are dropped.
tracer = trace.get_tracer("nano_claude")
log = logging.getLogger("nano_claude")

_TRUTHY = {"1", "true", "yes", "on"}
_initialized = False
_logger_provider: Any = None
# The swappable file exporter for per-session logs, when logs go to a file
# (the default). None when telemetry is off or logs are routed over OTLP.
_session_log_exporter: Any = None


def _enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUTHY


def init_telemetry() -> bool:
    """Install a real OTel provider when telemetry is enabled. Idempotent.

    Returns ``True`` if telemetry is now active, ``False`` if it is disabled by
    env var or the SDK (the ``otel`` extra) is not installed.
    """
    global _initialized
    if _initialized:
        return True
    if not _enabled("NANO_CLAUDE_TELEMETRY"):
        return False
    try:
        configure()
    except ImportError:
        logging.getLogger(__name__).warning(
            "NANO_CLAUDE_TELEMETRY is set but the OpenTelemetry SDK is not installed. "
            "Install it with: pip install 'nano-claude[otel]'"
        )
        return False
    _initialized = True
    return True


def _resource():
    from opentelemetry.sdk.resources import Resource

    # Let OTEL_SERVICE_NAME / OTEL_RESOURCE_ATTRIBUTES win; only supply defaults
    # the env hasn't set.
    defaults: dict[str, str] = {"service.version": __version__}
    if not os.environ.get("OTEL_SERVICE_NAME"):
        defaults["service.name"] = "nano-claude-code"
    return Resource.create(defaults)


def _trace_mode() -> str:
    """Where spans go: ``otlp`` (default), ``console``, or ``off`` (no exporter,
    so there's no backend to run). ``NANO_CLAUDE_TELEMETRY_CONSOLE=1`` is a
    legacy alias for ``console``.
    """
    mode = os.environ.get("NANO_CLAUDE_TELEMETRY_TRACES", "").strip().lower()
    if mode in ("otlp", "console", "off"):
        return mode
    if _enabled("NANO_CLAUDE_TELEMETRY_CONSOLE"):
        return "console"
    return "otlp"


def configure(*, span_exporter=None, log_exporter=None) -> None:
    """Wire trace + log providers. Traces follow ``NANO_CLAUDE_TELEMETRY_TRACES``
    (OTLP/HTTP by default); logs go to a per-session file. Tests pass in-memory
    exporters directly.

    Raises ``ImportError`` if the SDK / OTLP exporter packages are missing.
    """
    resource = _resource()
    _configure_traces(resource, span_exporter)
    _configure_logs(resource, log_exporter)


def _configure_traces(resource, span_exporter) -> None:
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        ConsoleSpanExporter,
        SimpleSpanProcessor,
    )

    mode = _trace_mode()
    if span_exporter is None and mode == "off":
        return  # leave the no-op proxy provider in place — no backend needed

    # A TracerProvider can only be set once per process; reuse ours (or any
    # already installed) and just attach another processor on reconfigure.
    provider = trace.get_tracer_provider()
    if not isinstance(provider, TracerProvider):
        provider = TracerProvider(resource=resource)
        trace.set_tracer_provider(provider)

    if span_exporter is not None:
        provider.add_span_processor(SimpleSpanProcessor(span_exporter))
    elif mode == "console":
        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    else:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))


def _build_session_file_exporter():
    """A LogExporter that writes JSON-line records to a swappable session file.

    Defined lazily so importing this module never requires the SDK. The target
    file is set via :func:`set_session_log_file` and re-pointed per session, so
    each conversation gets its own ``<session-id>.log.jsonl``.
    """
    import threading

    from opentelemetry.sdk._logs.export import LogExportResult, LogRecordExporter

    class _SessionFileLogExporter(LogRecordExporter):
        def __init__(self) -> None:
            self._file = None
            self._lock = threading.Lock()

        def set_session_file(self, path: Path) -> None:
            with self._lock:
                if self._file is not None:
                    self._file.close()
                path.parent.mkdir(parents=True, exist_ok=True)
                self._file = path.open("a", encoding="utf-8")

        def export(self, batch):  # noqa: ANN001 - SDK-defined signature
            # `batch` holds ReadableLogRecord objects, each with a one-line
            # `to_json(indent=None)` — written as JSONL to the session file.
            with self._lock:
                if self._file is None:
                    return LogExportResult.SUCCESS  # no session bound yet; drop
                for record in batch:
                    self._file.write(record.to_json(indent=None))
                    self._file.write("\n")
                self._file.flush()
            return LogExportResult.SUCCESS

        def shutdown(self) -> None:
            with self._lock:
                if self._file is not None:
                    self._file.close()
                    self._file = None

    return _SessionFileLogExporter()


def _configure_logs(resource, log_exporter) -> None:
    from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
    from opentelemetry.sdk._logs.export import (
        BatchLogRecordProcessor,
        SimpleLogRecordProcessor,
    )

    provider = LoggerProvider(resource=resource)

    global _session_log_exporter
    _session_log_exporter = None
    if log_exporter is not None:  # tests inject an in-memory exporter
        provider.add_log_record_processor(SimpleLogRecordProcessor(log_exporter))
    elif _enabled("NANO_CLAUDE_TELEMETRY_OTLP_LOGS"):
        # Opt-in: ship logs to an OTLP collector instead of a file.
        from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter

        provider.add_log_record_processor(BatchLogRecordProcessor(OTLPLogExporter()))
    else:
        # Default: per-session log file (a REPL can't share its console with
        # log output). SimpleLogRecordProcessor so records land in the current
        # session's file before /clear swaps to the next one.
        _session_log_exporter = _build_session_file_exporter()
        provider.add_log_record_processor(SimpleLogRecordProcessor(_session_log_exporter))

    # Route the `nano_claude` logger to OTel without also spilling to stderr.
    # The logs signal is still pre-stable; the SDK handler warns about its own
    # eventual move to opentelemetry-instrumentation-logging — silence that here
    # rather than take on an extra dependency for an unstable signal.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        handler = LoggingHandler(level=logging.INFO, logger_provider=provider)
    log.setLevel(logging.INFO)
    log.addHandler(handler)
    log.propagate = False

    global _logger_provider
    _logger_provider = provider


def set_session_log_file(path: str | Path) -> None:
    """Point per-session file logging at ``path`` (called when a session starts
    or /clear begins a new one). No-op when telemetry is off or logs go to OTLP.
    """
    if _session_log_exporter is not None:
        _session_log_exporter.set_session_file(Path(path))


def shutdown_telemetry() -> None:
    """Flush and shut down providers so buffered spans/logs survive exit."""
    provider = trace.get_tracer_provider()
    shutdown = getattr(provider, "shutdown", None)
    if callable(shutdown):
        try:
            shutdown()
        except Exception:  # noqa: BLE001 - never fail teardown over telemetry
            pass
    if _logger_provider is not None:
        try:
            _logger_provider.shutdown()
        except Exception:  # noqa: BLE001
            pass


def reset_for_testing() -> None:
    """Detach our log handler and clear init state (tests only).

    The global TracerProvider cannot be unset once installed, so tests reuse it
    and clear their in-memory exporter between cases instead.
    """
    global _initialized, _logger_provider, _session_log_exporter
    _initialized = False
    for h in list(log.handlers):
        log.removeHandler(h)
    log.propagate = True
    _logger_provider = None
    _session_log_exporter = None
