"""Which proposed benchmark stages this run actually measured.

A consumer acknowledgement is the terminal accepting a graphics query or
unlinking a shared-memory frame. It is not GPU time, compositor time, or
scanout. A protocol fixture cannot promote itself into any of those.
"""


def catalog(font_trials, draw_trials=False):
    raster = "measured" if font_trials else "not-run"
    compose = "measured" if draw_trials else "not-run"
    return [
        {
            "id": "cpu-glyph-raster",
            "status": raster,
            "measures": "Native mask generation for the shared cmap intersection.",
            "does_not_measure": "Shaping, GPU upload, presentation, and scanout.",
        },
        {
            "id": "mask-cache-compose",
            "status": compose,
            "measures": "The shared mask cache and CPU compositor on a draw case.",
            "does_not_measure": "FreeType's FTC cache and Neuronte's internal draw cache.",
        },
        {
            "id": "application-protocol",
            "status": "separate-suite",
            "measures": "The text and render suites time application and graphics-SDK work.",
            "does_not_measure": "Their consumer is a byte counter. It does not acknowledge a frame.",
        },
        {
            "id": "consumer-graphics-query",
            "status": "not-run",
            "measures": "A Kitty graphics query answered by a terminal process.",
            "does_not_measure": "The reply includes session setup. It is not display latency.",
        },
        {
            "id": "consumer-shm-read",
            "status": "not-run",
            "measures": (
                "Enqueue-to-observed-unlink: elapsed time from immediately before kittyfb_present "
                "until a poll sees the shared-memory name gone. frame_written_ns uses that same start "
                "until a poll sees frames_encoded advance."
            ),
            "does_not_measure": (
                "Neither milestone is the interval from a completed write to unlink. "
                "Polling can be late by one sample. Subtracting the milestones is not exact "
                "consumer latency. Unlink is a read acknowledgement, not scanout."
            ),
        },
        {
            "id": "gpu-upload",
            "status": "unavailable",
            "reason": "No GPU upload counter is read. Software GL forced for a virtual probe is still not a GPU measurement.",
        },
        {
            "id": "gpu-execution",
            "status": "unavailable",
            "reason": "No GPU timestamp or profiler is attached.",
        },
        {
            "id": "sdf-atlas",
            "status": "unavailable",
            "reason": "The public coverage API returns a cell mask, not a raw distance field. No atlas is built.",
        },
        {
            "id": "compositor-present",
            "status": "unavailable",
            "reason": "No compositor present fence is available to this suite.",
        },
        {
            "id": "scanout",
            "status": "unavailable",
            "reason": "No scanout or photon timestamp is available. A virtual display is not a panel.",
        },
    ]


def apply_probe(stages, probe):
    """Fold a probe record into the stage list. Reject display claims."""
    if probe is None:
        return stages
    for key in ("display_latency", "scanout", "gpu_latency"):
        if probe.get(key) != "unavailable":
            raise ValueError("presentation probe claimed " + key)
    if probe.get("similarity_winner") or probe.get("quality_rank"):
        raise ValueError("presentation probe ranked visual quality")
    peer = probe.get("peer")
    query = probe.get("graphics_query")
    stages = [dict(row) for row in stages]
    by_id = {row["id"]: row for row in stages}
    if peer == "protocol-fixture":
        if probe.get("shm_consumed_ns") is not None:
            raise ValueError("a protocol fixture must not report shared-memory consumption")
        by_id["consumer-graphics-query"].update(
            status="fixture-only",
            reason="The peer is an in-process reply, not a terminal. Its delay is not an acknowledgement measurement.")
        by_id["consumer-shm-read"].update(
            status="fixture-only",
            reason="The fixture does not read shared memory.")
    elif peer == "kilix-kitty" and query == "answered":
        by_id["consumer-graphics-query"].update(
            status="measured",
            session_start_ns=probe.get("session_start_ns"),
            reason="Round trip is kitty-framebuffer startup including the graphics query, termios, and window-size ioctl. Not scanout.")
        if probe.get("shm_consumed") == "observed" and probe.get("shm_consumed_ns") is not None:
            by_id["consumer-shm-read"].update(
                status="measured",
                clock="enqueue-to-observed-unlink",
                milestone_start="immediately-before-kittyfb-present",
                frame_enqueue_ns=probe.get("frame_enqueue_ns"),
                frame_written_ns=probe.get("frame_written_ns"),
                shm_consumed_ns=probe.get("shm_consumed_ns"),
                poll=probe.get("poll"),
                reason=(
                    "shm_consumed_ns and frame_written_ns share a start immediately before kittyfb_present. "
                    "The first is enqueue-to-observed-unlink and the second is enqueue-to-observed-frames-encoded. "
                    "Both include queue, write, and poll delay, and a sample can be late by one poll. "
                    "Subtracting them is not exact consumer latency. Unlink is a read acknowledgement, not display latency."
                ))
        else:
            by_id["consumer-shm-read"].update(
                status="unavailable",
                reason=probe.get("shm_consumed") or "shared-memory consumption was not observed")
    else:
        by_id["consumer-graphics-query"].update(
            status="unavailable", reason=probe.get("reason") or "no graphics terminal answered")
        by_id["consumer-shm-read"].update(
            status="unavailable", reason="no consumed shared-memory frame")
    by_id["scanout"]["status"] = "unavailable"
    by_id["gpu-execution"]["status"] = "unavailable"
    by_id["gpu-upload"]["status"] = "unavailable"
    by_id["compositor-present"]["status"] = "unavailable"
    return stages
