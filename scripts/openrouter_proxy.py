#!/usr/bin/env python3
"""Minimal local OpenRouter forwarding proxy for secure VitaBench runs.

VitaBench records its model configuration in simulation metadata. The proxy
keeps the real API key in its process environment and injects it on outbound
requests, so the benchmark only sees a harmless placeholder header.
"""

import argparse
import os
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


TARGET = os.environ.get(
    "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1/chat/completions"
)


class ProxyHandler(BaseHTTPRequestHandler):
    """Forward completion requests without logging headers or request bodies."""

    def do_POST(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            self.send_error(500, "OPENROUTER_API_KEY is unavailable")
            return

        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        request = urllib.request.Request(
            TARGET,
            data=body,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": self.headers.get(
                    "Content-Type", "application/json"
                ),
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                payload = response.read()
                self.send_response(response.status)
                self.send_header(
                    "Content-Type", response.headers.get("Content-Type", "application/json")
                )
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
        except urllib.error.HTTPError as error:
            payload = error.read()
            self.send_response(error.code)
            self.send_header(
                "Content-Type", error.headers.get("Content-Type", "application/json")
            )
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        except (OSError, urllib.error.URLError):
            self.send_error(502, "OpenRouter request failed")

    def log_message(self, format: str, *args: object) -> None:
        del format, args


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port-file", type=Path, required=True)
    args = parser.parse_args()

    server = ThreadingHTTPServer(("127.0.0.1", 0), ProxyHandler)
    args.port_file.write_text(str(server.server_port))
    server.serve_forever()


if __name__ == "__main__":
    main()
