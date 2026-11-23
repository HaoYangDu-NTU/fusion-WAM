# Read-only review and resolutions

Independent reviewer checked raw official schema/Gym mapping, allowlisting,
horizon/split selection, averaging and new implementation. Initial six tests passed.

Two issues were reproduced by failing regression tests and fixed:

1. Dataset construction could accept truncated episode metadata or missing later
   payloads. It now checks download receipt identity, metadata counts, unique
   episodes, and all declared parquet/video files before yielding frames.
2. The future opt-in training update validated loss after optimizer mutation. It
   now rejects nonfinite/nonscalar loss before backward/step. No training was run.

Recheck: both issues resolved, no other Important findings. An accidentally
repeated checkpoint definition was removed; CPU-only checkpoint scope is explicit.

Nonblocking operational limit: interruption between extracted-root rename and
receipt creation refuses to overwrite an unreceipted destination. This is
fail-closed. Inspect/remove or move aside that task's unreceipted directory before
retrying; do not manufacture a completion receipt. Per-episode eval JSONL survives
an interruption, but automatic resume/aggregation of a partial run is not provided.

Final delta review: default OSMesa, backend provenance, smoke-only RGB validation, source-path bootstrap and rollout metrics have no Important findings. Two pip-freeze entries referenced conda build directories (packaging/pip); removed them from pip lock because conda-explicit-linux-64.txt already pins these packages. Validate lock with pip --dry-run and keep conda-managed versions in the explicit lock.
