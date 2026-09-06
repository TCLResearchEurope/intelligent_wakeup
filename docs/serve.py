#!/usr/bin/env python3
"""
This code was developed by TCL Research Europe.

Serve the demo page locally the way GitLab Pages serves it.

Python's http.server ignores Range requests, so audio plays from the start but
the wake-up timeline cannot seek — clicking a turn appears to do nothing. This
server answers Range requests with 206 responses, which is what the deployed
page gets, so seeking behaves the same locally as in production.

Usage:
    python3 docs/serve.py                 # http://localhost:8000
    python3 docs/serve.py --port 9000
    python3 docs/serve.py --check         # verify the page, then exit
"""

import argparse
import functools
import http.server
import json
import os
import re
import socketserver
import sys
import urllib.request
from pathlib import Path

DOCS = Path(__file__).resolve().parent


class RangeHandler(http.server.SimpleHTTPRequestHandler):
    """Static handler that honours single-range byte requests."""

    def send_head(self):
        """Serve a 206 for a Range request, otherwise defer to the base class."""
        path = self.translate_path(self.path)
        if os.path.isdir(path) or not os.path.isfile(path):
            return super().send_head()

        size = os.path.getsize(path)
        match = re.match(r"bytes=(\d*)-(\d*)\s*$", self.headers.get("Range", ""))
        if not match or (not match.group(1) and not match.group(2)):
            self.send_response(200)
            self.send_header("Content-Type", self.guess_type(path))
            self.send_header("Content-Length", str(size))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            return open(path, "rb")

        if match.group(1):
            start = int(match.group(1))
            end = int(match.group(2)) if match.group(2) else size - 1
        else:
            # "bytes=-500" means the final 500 bytes.
            start = max(0, size - int(match.group(2)))
            end = size - 1
        end = min(end, size - 1)

        if start > end:
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.end_headers()
            return None

        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        handle = open(path, "rb")
        handle.seek(start)
        return _Window(handle, end - start + 1)

    def log_message(self, fmt, *args):
        """Quieten the per-request log; keep failures visible."""
        status = args[1] if len(args) > 1 else ""
        if not str(status).startswith("2"):
            super().log_message(fmt, *args)

    def handle_one_request(self):
        """Serve one request, treating a client hang-up as normal.

        A browser abandons an audio response as soon as it has the bytes it
        needs, which the base class reports as an unhandled traceback. That is
        expected here, not a fault worth printing.
        """
        try:
            super().handle_one_request()
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True


class _Window:
    """A read-only view of the first `remaining` bytes of an open file."""

    def __init__(self, handle, remaining):
        self.handle = handle
        self.remaining = remaining

    def read(self, size=-1):
        """Read within the window, stopping at its end."""
        if self.remaining <= 0:
            return b""
        want = self.remaining if size < 0 else min(size, self.remaining)
        chunk = self.handle.read(want)
        self.remaining -= len(chunk)
        return chunk

    def close(self):
        """Close the underlying file."""
        self.handle.close()


def check(port: int) -> int:
    """Fetch the page and its samples, reporting anything broken."""
    base = f"http://127.0.0.1:{port}"
    problems = []

    def get(path, headers=None):
        request = urllib.request.Request(f"{base}/{path}", headers=headers or {})
        return urllib.request.urlopen(request, timeout=10)

    try:
        page = get("index.html").read().decode("utf-8")
    except Exception as exc:  # noqa: BLE001 - report any failure to the user
        print(f"FAIL  index.html: {exc}")
        return 1
    print(f"ok    index.html ({len(page)} bytes)")

    # Every id the page's script looks up has to exist in the markup. A
    # container that gets renamed or deleted leaves its section empty, which is
    # invisible from the server side but obvious here.
    # Ids reach the DOM two ways: directly, and through the section() helper
    # that renders each part of the page.
    wanted = set(re.findall(r'getElementById\("([^"]+)"\)', page))
    wanted |= set(re.findall(r'section\("([^"]+)"', page))
    present = set(re.findall(r'id="([^"]+)"', page))
    absent = sorted(wanted - present)
    if absent:
        problems.append("index.html has no element for: " + ", ".join(absent))
    else:
        print(f"ok    all {len(wanted)} script containers present in the markup")

    try:
        data = json.loads(get("data/demo.json").read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL  data/demo.json: {exc}")
        return 1
    print(f"ok    data/demo.json ({len(data['voices'])} voices, "
          f"{len(data['featured'])} featured, {len(data['naturalness'])} pairs)")

    audio = [scene["audio"] for scene in data["featured"]]
    audio += [data["pair"]["with"]["audio"], data["pair"]["without"]["audio"]]
    audio += [voice["audio"] for voice in data["voices"]]
    for row in data["naturalness"]:
        audio += [row["ours"]["audio"], row["real"]["audio"]]

    total = 0
    for path in audio:
        try:
            response = get(path)
            # Read the body: a served header proves nothing about the file
            # behind it, and draining it also keeps the server log quiet.
            body = response.read()
            total += len(body)
            if response.status != 200:
                problems.append(f"{path}: HTTP {response.status}")
            elif not body:
                problems.append(f"{path}: empty")
        except Exception as exc:  # noqa: BLE001
            problems.append(f"{path}: {exc}")
    if problems:
        print(f"FAIL  {len(problems)} of {len(audio)} audio files failed")
    else:
        print(f"ok    {len(audio)} audio files served ({total / 1e6:.1f} MB)")

    # Seeking depends on this: without a 206 the timeline cannot jump.
    try:
        response = get(audio[0], {"Range": "bytes=1000-1999"})
        chunk = response.read()
        if response.status == 206 and len(chunk) == 1000:
            print("ok    range requests answered with 206 (timeline can seek)")
        else:
            problems.append(
                f"range request returned {response.status} with {len(chunk)} bytes, "
                "expected 206 with 1000"
            )
    except Exception as exc:  # noqa: BLE001
        problems.append(f"range request failed: {exc}")

    # Every turn needs a start time inside the audio, in order. A turn timed
    # past the end of its scene cannot be reached by clicking, which is exactly
    # what the corpus's own planned timings did.
    for scene in data["featured"] + [data["pair"]["with"], data["pair"]["without"]]:
        name = scene.get("id", "scene")
        previous = -1.0
        for i, turn in enumerate(scene["turns"]):
            start = turn.get("t")
            if start is None:
                problems.append(f"{name} turn {i} has no time")
                continue
            if start > scene["duration"]:
                problems.append(
                    f"{name} turn {i} starts at {start:.0f}s, past the "
                    f"{scene['duration']:.0f}s audio"
                )
            if start < previous:
                problems.append(f"{name} turn {i} starts before the turn before it")
            previous = start
    if data["pair"]["without"]["assistantTurns"]:
        problems.append("the no-assistant scene contains assistant turns")
    if data["pair"]["with"]["wakeIndex"] is None:
        problems.append("the with-assistant scene has no detected wake turn")

    for problem in problems:
        print(f"FAIL  {problem}")
    print("\n" + ("all checks passed" if not problems else f"{len(problems)} problem(s)"))
    return 1 if problems else 0


def main() -> None:
    """Serve docs/, optionally running the checks and exiting."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000, help="port to listen on")
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the page and every sample loads, then exit",
    )
    args = parser.parse_args()

    if not (DOCS / "data" / "demo.json").exists():
        sys.exit("data/demo.json is missing — run: python3 docs/build_assets.py")

    socketserver.TCPServer.allow_reuse_address = True
    handler = functools.partial(RangeHandler, directory=str(DOCS))
    # --check only needs a port for the duration of the checks, so let the OS
    # pick a free one. Binding a fixed port would fail the CI job whenever
    # something else on the runner happens to hold it.
    port = 0 if args.check and args.port == parser.get_default("port") else args.port
    with socketserver.ThreadingTCPServer(("", port), handler) as server:
        if args.check:
            import threading

            bound = server.server_address[1]
            threading.Thread(target=server.serve_forever, daemon=True).start()
            code = check(bound)
            server.shutdown()
            sys.exit(code)

        print(f"Serving the demo page on http://localhost:{args.port}")
        print("Range requests are supported, so the timeline can seek. Ctrl+C to stop.")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    main()
