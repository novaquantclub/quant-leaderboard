"""Docker-free worker service and authenticated client. No market data is stored here."""

import argparse
import hashlib
import hmac
import http.client
import ipaddress
import json
import math
import os
import queue
import re
import secrets
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import suppress
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import metadata as packages
from pathlib import Path
from urllib.parse import urlsplit

MAX_SOURCE = 128_000
MAX_REPLY = 262_144
MAX_STATE = 16_000_000
SYMBOL = re.compile(r"[A-Z0-9.^=_-]{1,32}\Z")


class StrategyError(RuntimeError):
    pass


class WorkerError(StrategyError):
    """Infrastructure failure requiring a retry of the frozen evaluation."""


class WorkerBusyError(WorkerError):
    pass


def decode_json(data):
    def invalid(value):
        raise ValueError(f"Non-finite JSON number: {value}")

    return json.loads(data, parse_constant=invalid)


def validate_metadata(value):
    if not isinstance(value, dict):
        raise StrategyError("Invalid strategy metadata.")
    for field in ("name", "author"):
        text = value.get(field)
        if not isinstance(text, str) or not text.strip() or len(text) > 120:
            raise StrategyError(
                f"Trader.{field} must be a non-empty string of at most 120 characters."
            )
    tickers = value.get("tickers")
    if not isinstance(tickers, list) or not 1 <= len(tickers) <= 20:
        raise StrategyError("Trader.tickers must list between 1 and 20 symbols.")
    if any(not isinstance(t, str) or not SYMBOL.fullmatch(t) for t in tickers) or len(
        set(tickers)
    ) != len(tickers):
        raise StrategyError("Use unique uppercase ticker symbols.")
    return {
        "name": value["name"].strip(),
        "author": value["author"].strip(),
        "tickers": tickers,
    }


def validate_orders(value, tickers):
    if not isinstance(value, dict) or "error" in value:
        raise StrategyError(
            str(value.get("error", "Invalid response."))[:600]
            if isinstance(value, dict)
            else "Invalid response."
        )
    orders, trader_data = value.get("orders"), value.get("trader_data")
    if not isinstance(orders, dict) or len(orders) > 20:
        raise StrategyError("Strategy must return an order dictionary.")
    if not isinstance(trader_data, str) or len(trader_data.encode()) > 100_000:
        raise StrategyError("trader_data must be a string of at most 100 KB.")
    for symbol, quantity in orders.items():
        if (
            symbol not in tickers
            or isinstance(quantity, bool)
            or not isinstance(quantity, (float, int))
            or not math.isfinite(quantity)
        ):
            raise StrategyError(
                "Orders must contain declared symbols and finite numeric quantities."
            )
    return orders, trader_data


class SandboxProcess:
    """Only launcher. There is deliberately no unsandboxed fallback."""

    def __init__(self, source):
        bwrap = shutil.which("bwrap")
        if sys.platform != "linux" or not bwrap or os.geteuid() == 0:
            raise WorkerError(
                "Worker requires Linux, bubblewrap, and an unprivileged service user."
            )
        if not isinstance(source, str) or len(source.encode()) > MAX_SOURCE:
            raise StrategyError("Strategy source exceeds 128 KB.")
        self.directory = tempfile.TemporaryDirectory(prefix="nqc-strategy-")
        self.process = None
        self.started = self.touched = time.monotonic()
        self.lock = threading.Lock()
        self.messages = queue.Queue(maxsize=1)
        try:
            path = Path(self.directory.name) / "strategy.py"
            path.write_text(source, encoding="utf-8")
            root = Path(__file__).resolve().parent
            self.process = subprocess.Popen(
                [
                    sys.executable,
                    "-I",
                    "-B",
                    str(root / "sandbox_launch.py"),
                    bwrap,
                    str(path),
                    str(root / "strategy_bootstrap.py"),
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"},
                close_fds=True,
                start_new_session=True,
            )
            threading.Thread(target=self._reader, daemon=True).start()
            self.metadata = validate_metadata(self._receive(15))
        except BaseException:
            self.close()
            raise

    def _reader(self):
        try:
            while True:
                line = self.process.stdout.readline(MAX_REPLY + 1)
                if not line or len(line) > MAX_REPLY:
                    self.process.kill()
                    self.messages.put_nowait(
                        StrategyError("Strategy exited or exceeded the response limit.")
                    )
                    return
                self.messages.put_nowait(line)
        except (OSError, ValueError, queue.Full):
            if self.process.poll() is None:
                self.process.kill()

    def _receive(self, timeout):
        remaining = 45 - (time.monotonic() - self.started)
        if remaining <= 0:
            raise StrategyError("Strategy exceeded the total time limit.")
        try:
            value = self.messages.get(timeout=min(timeout, remaining))
        except queue.Empty:
            self.close()
            raise StrategyError("Strategy timed out.") from None
        if isinstance(value, Exception):
            raise value
        try:
            value = decode_json(value)
        except (ValueError, UnicodeError):
            raise StrategyError("Strategy returned invalid JSON.") from None
        if isinstance(value, dict) and "error" in value:
            raise StrategyError(str(value["error"])[:600])
        return value

    def run(self, state):
        with self.lock:
            self.touched = time.monotonic()
            encoded = (
                json.dumps(state, allow_nan=False, separators=(",", ":")).encode()
                + b"\n"
            )
            if len(encoded) > MAX_STATE:
                raise StrategyError("State exceeds the worker input limit.")

            # A writer thread also bounds a strategy that stops reading stdin.
            def send():
                try:
                    self.process.stdin.write(encoded)
                    self.process.stdin.flush()
                except (OSError, ValueError):
                    pass

            sender = threading.Thread(target=send, daemon=True)
            sender.start()
            result = self._receive(3)
            sender.join(timeout=0.1)
            validate_orders(result, self.metadata["tickers"])
            return result

    def close(self):
        if self.process is not None:
            if self.process.poll() is None:
                self.process.kill()
            self.process.wait(timeout=5)
            for stream in (self.process.stdin, self.process.stdout):
                if stream:
                    stream.close()
        self.directory.cleanup()


def verify_sandbox():
    """Fail service startup when the real launcher cannot enforce its basic policy."""
    with tempfile.TemporaryDirectory(prefix="nqc-health-") as directory:
        canary = Path(directory) / "host-only"
        canary.write_text(secrets.token_urlsafe(32), encoding="utf-8")
        source = (
            "HOST_CANARY = "
            + repr(str(canary))
            + "\n"
            + """
import json, os, resource, socket
class Trader:
    name = 'Worker isolation check'
    author = 'Infrastructure'
    tickers = ['HEALTH']
    def run(self, state):
        checks = {}
        try:
            with open(HOST_CANARY) as file:
                file.read()
            checks['host_file_blocked'] = False
        except OSError:
            checks['host_file_blocked'] = True
        try:
            with open('/work/write-check', 'w') as file:
                file.write('probe')
            checks['writes_blocked'] = False
        except OSError:
            checks['writes_blocked'] = True
        try:
            connection = socket.socket()
            connection.close()
            checks['sockets_blocked'] = False
        except OSError:
            checks['sockets_blocked'] = True
        checks['worker_secret_absent'] = 'NQC_WORKER_TOKEN' not in os.environ
        checks['memory_limited'] = 0 < resource.getrlimit(resource.RLIMIT_AS)[1] <= 1024**3
        checks['cpu_limited'] = 0 < resource.getrlimit(resource.RLIMIT_CPU)[1] <= 30
        checks['namespaces'] = {name: os.readlink('/proc/self/ns/' + name) for name in ('mnt', 'pid', 'net')}
        return {}, json.dumps(checks)
"""
        )
        session = None
        try:
            session = SandboxProcess(source)
            response = session.run(
                {
                    "step": 0,
                    "history_start": 0,
                    "history": {"HEALTH": [[100, 100, 100, 100, 1]]},
                    "positions": {"HEALTH": 0},
                    "cash": 1,
                    "portfolio_value": 1,
                    "trader_data": "",
                }
            )
            checks = json.loads(response["trader_data"])
            namespaces = checks.pop("namespaces")
            if not all(checks.values()) or any(
                namespaces[name] == os.readlink(f"/proc/self/ns/{name}")
                for name in namespaces
            ):
                raise WorkerError("Native sandbox policy verification failed.")
        except (StrategyError, ValueError, OSError, KeyError) as exc:
            raise WorkerError(
                "Native sandbox verification failed. Check the worker host configuration."
            ) from exc
        finally:
            if session is not None:
                session.close()


def runtime_fingerprint():
    code_root = Path(__file__).resolve().parent
    fingerprint = {
        "policy": "native-bwrap-seccomp-v1",
        "python": sys.version,
        "code": {
            name: hashlib.sha256((code_root / name).read_bytes()).hexdigest()
            for name in (
                "strategy_runner.py",
                "strategy_bootstrap.py",
                "sandbox_launch.py",
            )
        },
        "packages": {},
    }
    for name in (
        "numpy",
        "pandas",
        "scipy",
        "statsmodels",
        "scikit-learn",
        "numba",
    ):
        try:
            fingerprint["packages"][name] = packages.version(name)
        except packages.PackageNotFoundError:
            fingerprint["packages"][name] = None
    return hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()


class WorkerServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self, address, token, *, session_factory=SandboxProcess, max_sessions=2
    ):
        if len(token) < 32:
            raise ValueError("Worker API token must contain at least 32 characters.")
        self.sandbox_verified = session_factory is SandboxProcess
        if self.sandbox_verified:
            verify_sandbox()
        self.token, self.factory, self.max_sessions = (
            token,
            session_factory,
            max_sessions,
        )
        self.sessions, self.sessions_lock = {}, threading.Lock()
        self.request_slots = threading.BoundedSemaphore(16)
        self.runtime_id = runtime_fingerprint()
        super().__init__(address, WorkerHandler)
        self.stopped = threading.Event()
        threading.Thread(target=self._reaper, daemon=True).start()

    def verify_runtime(self):
        if runtime_fingerprint() != self.runtime_id:
            raise WorkerError("Worker files or packages changed. Restart the service.")
        return self.runtime_id

    def process_request(self, request, address):
        if not self.request_slots.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, address)
        except BaseException:
            self.request_slots.release()
            raise

    def process_request_thread(self, request, address):
        try:
            super().process_request_thread(request, address)
        finally:
            self.request_slots.release()

    def _reaper(self):
        while not self.stopped.wait(1):
            with self.sessions_lock:
                expired = [
                    key
                    for key, session in self.sessions.items()
                    if time.monotonic() - session.started > 45
                ]
                for key in expired:
                    self.sessions.pop(key).close()

    def server_close(self):
        self.stopped.set()
        with self.sessions_lock:
            for session in self.sessions.values():
                session.close()
            self.sessions.clear()
        super().server_close()


class WorkerHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def setup(self):
        super().setup()
        # Bound idle/header reads too, before authentication or body parsing.
        self.connection.settimeout(20)
        self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def log_message(self, *args):
        pass  # Do not log source code, state, URLs with tokens, or authorization headers.

    def _reply(self, status, value):
        encoded = json.dumps(value, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_POST(self):
        if not hmac.compare_digest(
            self.headers.get("Authorization", "").encode(),
            ("Bearer " + self.server.token).encode(),
        ):
            self.close_connection = True
            self._reply(401, {"error": "Worker authentication failed."})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= MAX_STATE:
                raise StrategyError("Invalid request size.")
            body = decode_json(self.rfile.read(length))
            if not isinstance(body, dict):
                raise StrategyError("Request must be an object.")
            if self.path == "/info":
                self._reply(
                    200,
                    {
                        "runtime_id": self.server.verify_runtime(),
                        "sandbox_verified": self.server.sandbox_verified,
                    },
                )
                return
            if self.path == "/sessions":
                purpose = body.get("purpose", "practice")
                if purpose not in ("practice", "final"):
                    raise WorkerError("Unknown worker session purpose.")
                runtime_id = self.server.verify_runtime()
                if body.get("expected_runtime_id", runtime_id) != runtime_id:
                    raise WorkerError("Worker runtime differs from the frozen round.")
                with self.server.sessions_lock:
                    practice_count = sum(
                        getattr(session, "purpose", "practice") == "practice"
                        for session in self.server.sessions.values()
                    )
                    if len(self.server.sessions) >= self.server.max_sessions or (
                        purpose == "practice"
                        and practice_count >= max(1, self.server.max_sessions - 1)
                    ):
                        self._reply(
                            429, {"error": "Worker is busy. Try again shortly."}
                        )
                        return
                    session = self.server.factory(body.get("source"))
                    session.purpose = purpose
                    key = secrets.token_urlsafe(24)
                    self.server.sessions[key] = session
                self._reply(
                    200,
                    {
                        "session": key,
                        "metadata": session.metadata,
                        "history_updates": True,
                        "runtime_id": runtime_id,
                    },
                )
                return
            match = re.fullmatch(
                r"/sessions/([A-Za-z0-9_-]{32})/(step|close)", self.path
            )
            if not match:
                raise StrategyError("Unknown worker endpoint.")
            key, operation = match.groups()
            with self.server.sessions_lock:
                session = self.server.sessions.get(key)
                if operation == "close" and session:
                    self.server.sessions.pop(key).close()
            if session is None:
                raise StrategyError("Worker session expired.")
            result = {"closed": True} if operation == "close" else session.run(body)
            self._reply(200, result)
        except (WorkerError, OSError) as exc:
            self.close_connection = True
            self._reply(503, {"error": str(exc)[:600], "error_type": "worker"})
        except (StrategyError, ValueError, AttributeError, KeyError) as exc:
            self.close_connection = True
            self._reply(400, {"error": str(exc)[:600]})


class RemoteTrader:
    def __init__(
        self,
        source=None,
        *,
        endpoint=None,
        token=None,
        purpose="practice",
        expected_runtime_id=None,
    ):
        endpoint = endpoint if endpoint is not None else os.getenv("NQC_WORKER_URL", "")
        token = token if token is not None else os.getenv("NQC_WORKER_TOKEN", "")
        url = urlsplit(endpoint)
        if (
            not url.hostname
            or url.username
            or url.password
            or url.query
            or url.fragment
            or url.path not in ("", "/")
        ):
            raise WorkerError(
                "Configure a worker URL without credentials, a path, or query parameters."
            )
        if url.scheme == "http":
            if url.hostname != "localhost":
                try:
                    local = ipaddress.ip_address(url.hostname).is_loopback
                except ValueError:
                    local = False
                if not local:
                    raise WorkerError("Use HTTPS for a worker outside localhost.")
        elif url.scheme != "https":
            raise WorkerError(
                "Configure an HTTPS worker URL (HTTP is allowed only on localhost)."
            )
        if len(token) < 32:
            raise WorkerError("Configure the worker API token.")
        if purpose not in ("practice", "final"):
            raise WorkerError("Unknown worker session purpose.")
        self.token, self.session = token, None
        self.history_updates = False
        self.history_start = self.history_step = None
        self.history_fingerprints = {}
        connection = (
            http.client.HTTPSConnection
            if url.scheme == "https"
            else http.client.HTTPConnection
        )
        self.connection = connection(url.hostname, url.port, timeout=20)
        if source is None:
            return
        try:
            request = {"source": source, "purpose": purpose}
            if expected_runtime_id is not None:
                request["expected_runtime_id"] = expected_runtime_id
            result = self._request("/sessions", request)
            self.session = result["session"]
            if (
                expected_runtime_id is not None
                and result.get("runtime_id") != expected_runtime_id
            ):
                raise WorkerError("Worker runtime differs from the frozen round.")
            self.history_updates = result.get("history_updates") is True
            self.metadata = validate_metadata(result["metadata"])
            self.name, self.author, self.tickers = (
                self.metadata[key] for key in ("name", "author", "tickers")
            )
        except BaseException:
            self.close()
            raise

    def _request(self, path, body):
        try:
            data = json.dumps(body, allow_nan=False, separators=(",", ":")).encode()
            if len(data) > MAX_STATE:
                raise StrategyError("Request exceeds the size limit.")
            self.connection.request(
                "POST",
                path,
                body=data,
                headers={
                    "Authorization": "Bearer " + self.token,
                    "Content-Type": "application/json",
                },
            )
            response = self.connection.getresponse()
            raw = response.read(MAX_REPLY + 1)
            if len(raw) > MAX_REPLY:
                raise WorkerError("Worker response exceeds the size limit.")
            value = decode_json(raw)
            if not isinstance(value, dict):
                raise WorkerError("Worker returned an invalid response object.")
            if response.status != 200:
                message = str(value.get("error", "Worker rejected the request."))[:600]
                if response.status == 429:
                    raise WorkerBusyError(message)
                if (
                    response.status in (401, 403)
                    or response.status >= 500
                    or value.get("error_type") == "worker"
                ):
                    raise WorkerError(message)
                raise StrategyError(message)
            return value
        except (OSError, http.client.HTTPException, ValueError, KeyError) as exc:
            raise WorkerError("Cannot communicate with the strategy worker.") from exc

    @staticmethod
    def _fingerprints(groups, length=None):
        return {
            (group, name): hashlib.sha256(values[:length].tobytes()).digest()
            for group, arrays in groups.items()
            for name, values in arrays.items()
        }

    def run(self, state):
        first = int(next(iter(state.history.values())).index[0])
        arrays = {
            "history": {
                name: frame.to_numpy(dtype=float)
                for name, frame in state.history.items()
            },
            "observations": {
                name: series.to_numpy(dtype=float)
                for name, series in getattr(state, "observations", {}).items()
            },
        }
        incremental = (
            self.history_updates
            and self.history_start == first
            and self.history_step is not None
            and state.step > self.history_step
            and self._fingerprints(arrays, self.history_step - first + 1)
            == self.history_fingerprints
        )
        offset = self.history_step - first + 1 if incremental else 0
        payload = {
            "step": state.step,
            "history_start": first,
            "history": {
                name: values[offset:].tolist()
                for name, values in arrays["history"].items()
            },
            "positions": state.positions,
            "cash": state.cash,
            "portfolio_value": state.portfolio_value,
            "trader_data": state.trader_data,
            "observations": {
                name: values[offset:].tolist()
                for name, values in arrays["observations"].items()
            },
        }
        if incremental:
            payload["history_update_start"] = self.history_step + 1
        result = validate_orders(
            self._request(f"/sessions/{self.session}/step", payload), self.tickers
        )
        self.history_start, self.history_step = first, state.step
        self.history_fingerprints = self._fingerprints(arrays)
        return result

    def close(self):
        try:
            with suppress(StrategyError):
                if self.session:
                    self._request(f"/sessions/{self.session}/close", {"close": True})
        finally:
            self.connection.close()
            self.session = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def main():
    parser = argparse.ArgumentParser(
        description="Native Linux strategy worker; no Docker required."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8788)
    args = parser.parse_args()
    if sys.platform != "linux" or os.geteuid() == 0 or not shutil.which("bwrap"):
        parser.error(
            "Install bubblewrap and run as a dedicated, unprivileged Linux user."
        )
    server = WorkerServer((args.host, args.port), os.getenv("NQC_WORKER_TOKEN", ""))
    try:
        print(f"Strategy worker listening on {args.host}:{args.port}", flush=True)
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def worker_runtime_id():
    with RemoteTrader() as client:
        info = client._request("/info", {"info": True})
        if info.get("sandbox_verified") is not True:
            raise WorkerError(
                "The worker must verify native isolation before a final round can be prepared or scored."
            )
        value = info.get("runtime_id", "")
        if not isinstance(value, str) or not re.fullmatch("[0-9a-f]{64}", value):
            raise WorkerError("Worker returned an invalid runtime fingerprint.")
        return value


if __name__ == "__main__":
    main()
