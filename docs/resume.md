# Resume a frozen native experiment

`tools/resume_experiment.py CHECKPOINT OUTPUT_DIRECTORY` continues a saved
prefix of a decomposition experiment. Set `PYTHONPATH` to the original,
unchanged harness. The output directory must be new. The original journal
and native executable remain intact.

The checkpoint contains the original arguments and family, worker executable
and build record, complete journal digest, quality digest, optional preview
digests, and file fingerprints. `required_files(paths, binary)` enumerates the
minimum fingerprint set: all input assets, the harness package sources, native
worker source and executable, runtime library, and resolved ELF dependencies.
Capture these fingerprints from the original run; the adapter refuses an
incomplete or changed set.

The adapter repeats validation outside timing, verifies the exact shuffled
schedule and pixel/API counters, and replays completed rows without invoking
native timing. It requires a byte-identical prefix before executing the
remaining jobs. `--stop-after N` or an output-directory `stop-requested` file
stops at a completed batch boundary with exit75 and a new `checkpoint.json`.
A complete run writes `result.json`, preserving the usual raw-trial schema.
Only one-handle bundle cases are supported.

`tools/checkpoint_gate.py` supports a separately audited live handoff. Its
configuration names a ready-receipt path and an authorization path. When
installed atomically at a temporary worker pathname, an already executing
inode finishes normally; the next invocation records its arguments and waits
without executing native work. The controller must verify that invocation
against the next scheduled job, save and fsync the exact completed prefix and
provenance, then authorize the matching gate PID to exit. Keep the original
benchmark lock held throughout this handoff.

Resume under the same target lock, CPU affinity, assets, arguments and native
binary. Segment provenance records scheduling gaps; host temperature and OS
cache state can change during a gap. Report those gaps with the results.

The native fixtures exercise two resume boundaries, all ten repetitions,
complete-prefix replay, source/input omission and corruption, and an atomic
gate handoff while a prior native process is still executing.
