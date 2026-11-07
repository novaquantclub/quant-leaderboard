import json
import os
import shutil
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from test_synthetic_market import training_fixture

from backtester import TradingState
from strategy_runner import (
    RemoteTrader,
    SandboxProcess,
    StrategyError,
    WorkerServer,
    validate_metadata,
    validate_orders,
)

SOURCE = """
class Trader:
    name = 'Test strategy'
    author = 'Untrusted author'
    tickers = ['SPY']
    def run(self, state):
        return {'SPY': 1}, str(len(state.history['SPY']))
"""


def wire_state():
    return {
        "step": 0,
        "history_start": -100,
        "history": {"SPY": training_fixture(120)["SPY"].iloc[:101].to_numpy().tolist()},
        "positions": {"SPY": 0},
        "cash": 100_000,
        "portfolio_value": 100_000,
        "trader_data": "",
    }


class FakeSession:
    def __init__(self, source):
        self.started = time.monotonic()
        self.metadata = {"name": "Test", "author": "Test", "tickers": ["SPY"]}

    def run(self, state):
        return {"orders": {"SPY": 1}, "trader_data": "ok"}

    def close(self):
        pass


class WorkerProtocolTests(unittest.TestCase):
    def setUp(self):
        self.token = "test-only-token-with-thirty-two-characters"
        self.server = WorkerServer(
            ("127.0.0.1", 0), self.token, session_factory=FakeSession
        )
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.endpoint = f"http://127.0.0.1:{self.server.server_port}"
        self.addCleanup(self.close_server)

    def close_server(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)

    def test_authenticated_stream_and_session_cleanup(self):
        with RemoteTrader(SOURCE, endpoint=self.endpoint, token=self.token) as trader:
            frame = training_fixture(120)["SPY"].iloc[:101].copy()
            frame.index = range(-100, 1)
            state = TradingState(0, {"SPY": frame}, {"SPY": 0}, 100_000, 100_000)
            self.assertEqual(trader.run(state), ({"SPY": 1}, "ok"))
            self.assertEqual(len(self.server.sessions), 1)
        self.assertEqual(self.server.sessions, {})

    def test_wrong_token_and_non_local_plaintext_transport_are_rejected(self):
        with self.assertRaises(StrategyError):
            RemoteTrader(SOURCE, endpoint=self.endpoint, token="wrong" * 10)
        with self.assertRaisesRegex(StrategyError, "HTTPS"):
            RemoteTrader(SOURCE, endpoint="http://example.com", token=self.token)

    def test_capacity_is_bounded(self):
        with (
            RemoteTrader(SOURCE, endpoint=self.endpoint, token=self.token),
            RemoteTrader(SOURCE, endpoint=self.endpoint, token=self.token),
            self.assertRaisesRegex(StrategyError, "busy"),
        ):
            RemoteTrader(SOURCE, endpoint=self.endpoint, token=self.token)

    def test_non_finite_and_oversized_results_are_rejected(self):
        for quantity in (True, float("nan"), float("inf"), "1"):
            with self.subTest(quantity=quantity), self.assertRaises(StrategyError):
                validate_orders(
                    {"orders": {"SPY": quantity}, "trader_data": ""}, ["SPY"]
                )
        with self.assertRaises(StrategyError):
            validate_orders({"orders": {}, "trader_data": "x" * 100001}, ["SPY"])
        with self.assertRaises(StrategyError):
            validate_metadata({"name": "", "author": "a", "tickers": ["SPY"]})

    @unittest.skipIf(
        sys.platform == "linux" and shutil.which("bwrap") and os.geteuid() != 0,
        "A sandbox is available",
    )
    def test_missing_sandbox_fails_closed(self):
        with self.assertRaises(StrategyError):
            SandboxProcess(SOURCE)


@unittest.skipUnless(
    sys.platform == "linux" and shutil.which("bwrap") and os.geteuid() != 0,
    "Requires an unprivileged Linux user and bubblewrap",
)
class NativeSandboxTests(unittest.TestCase):
    def sandbox(self, source):
        process = SandboxProcess(source)
        self.addCleanup(process.close)
        return process

    def test_numpy_pandas_strategy_executes_with_only_current_history(self):
        process = self.sandbox(SOURCE)
        result = process.run(wire_state())
        self.assertEqual(result["orders"], {"SPY": 1})
        self.assertEqual(result["trader_data"], "101")

    def test_interest_rate_history_arrives_as_observed_percentages(self):
        source = (
            SOURCE[: SOURCE.index("    def run")]
            + """    def run(self, state):
        rate = state.observations['interest rate']
        assert rate.index[-1] == state.step
        return {}, repr((len(rate), float(rate.iloc[-1])))
"""
        )
        state = wire_state()
        state["observations"] = {"interest rate": [5.25] * 101}
        result = self.sandbox(source).run(state)
        self.assertEqual(result["trader_data"], "(101, 5.25)")

    def test_host_files_environment_network_writes_and_fork_are_blocked(self):
        with tempfile.TemporaryDirectory() as directory:
            secret = Path(directory) / "private-marker"
            secret.write_text("private value")
            source = SOURCE[: SOURCE.index("    def run")]
            source += """    def run(self, state):
        import os, socket, json, inspect
        checks = {}
        try:
            open(HOST_FILE).read()
            checks['host_file_blocked'] = False
        except OSError:
            checks['host_file_blocked'] = True
        checks['secret_env_blocked'] = 'NQC_CANARY' not in os.environ
        try:
            socket.socket()
            checks['network_blocked'] = False
        except OSError:
            checks['network_blocked'] = True
        try:
            open('/tmp/marker', 'w').write('x')
            checks['writes_blocked'] = False
        except OSError:
            checks['writes_blocked'] = True
        try:
            pid = os.fork()
            if pid == 0: os._exit(0)
            os.waitpid(pid, 0)
            checks['fork_blocked'] = False
        except OSError:
            checks['fork_blocked'] = True
        frame = inspect.currentframe().f_back
        checks['no_evaluator_frame'] = all(key not in frame.f_locals for key in ('raw_data', 'evaluation_dates', 'seed'))
        checks['only_current_bars'] = len(state.history['SPY']) == 101
        return {}, json.dumps(checks)
"""
            source = "HOST_FILE = " + repr(str(secret)) + "\n" + source
            with patch.dict(os.environ, {"NQC_CANARY": "must not reach strategy"}):
                process = self.sandbox(source)
                checks = json.loads(process.run(wire_state())["trader_data"])
            self.assertTrue(checks and all(checks.values()), checks)

    def test_runaway_strategy_is_killed(self):
        source = (
            SOURCE[: SOURCE.index("    def run")]
            + "    def run(self, state):\n        while True: pass\n"
        )
        process = self.sandbox(source)
        started = time.monotonic()
        with self.assertRaisesRegex(StrategyError, "timed out"):
            process.run(wire_state())
        self.assertLess(time.monotonic() - started, 8)
        self.assertIsNotNone(process.process.poll())

    def test_output_flood_is_bounded(self):
        with self.assertRaises(StrategyError):
            self.sandbox("import os\nos.write(1,b'x' * 300000)\n" + SOURCE)

    def test_memory_mapping_is_limited(self):
        source = (
            SOURCE[: SOURCE.index("    def run")]
            + """    def run(self, state):
        import mmap
        try:
            allocation = mmap.mmap(-1, 2 * 1024**3)
            return {}, 'unlimited'
        except (OSError, MemoryError):
            return {}, 'limited'
"""
        )
        self.assertEqual(
            self.sandbox(source).run(wire_state())["trader_data"], "limited"
        )

    def test_common_strategy_randomness_reproduces_across_fresh_workers(self):
        source = (
            SOURCE[: SOURCE.index("    def run")]
            + """    def run(self, state):
        import random, numpy as np
        return {}, repr((random.random(), np.random.random()))
"""
        )
        first = self.sandbox(source).run(wire_state())["trader_data"]
        second = self.sandbox(source).run(wire_state())["trader_data"]
        self.assertEqual(first, second)

    def test_real_worker_http_round_trip_and_frozen_final_scoring(self):
        import pandas as pd

        from backtester import EngineConfig
        from competition import Competition
        from interest_rate import INTEREST_RATE, INTEREST_RATE_FILE
        from synthetic_market import SyntheticConfig

        token = "integration-test-only-worker-token-12345678"
        source = SOURCE.replace(
            "return {'SPY': 1}, str(len(state.history['SPY']))",
            "assert len(state.observations['interest rate']) == 101 + state.step\n"
            "        return {'SPY': 1}, str(len(state.history['SPY']))",
        )
        server = WorkerServer(("127.0.0.1", 0), token)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                training = root / "training"
                training.mkdir()
                for symbol, frame in training_fixture(130).items():
                    frame.to_csv(training / f"{symbol}.csv", index_label="Date")
                pd.Series([5.25] * 130, index=frame.index, name=INTEREST_RATE).to_csv(
                    training / INTEREST_RATE_FILE, index_label="Date"
                )
                store = Competition(root / "db.sqlite", root / "rounds")
                team_token = store.create_team("Integration team")
                team = store.authenticate(team_token)
                round_id = store.create_round(
                    "Integration",
                    training,
                    EngineConfig(runtime_id=server.runtime_id),
                    SyntheticConfig(2, 252, 100),
                )
                endpoint = f"http://127.0.0.1:{server.server_port}"
                with patch.dict(
                    os.environ, {"NQC_WORKER_URL": endpoint, "NQC_WORKER_TOKEN": token}
                ):
                    with RemoteTrader(source) as trader:
                        version = store.submit(
                            round_id, team["id"], source, trader.metadata
                        )
                    store.select(round_id, team["id"], version)
                    store.lock_round(round_id)
                    store.score(round_id)
                    self.assertEqual(store.results(round_id), [])
                    store.publish(round_id)
                result = store.results(round_id)[0]
                self.assertIsNone(result["error"], result["error"])
                self.assertEqual(json.loads(result["result"])["paths_tested"], 2)
                self.assertEqual(server.sessions, {})
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_dedicated_virtual_environment_runtime(self):
        import subprocess
        import venv

        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory) / "runtime"
            venv.EnvBuilder(with_pip=False, system_site_packages=True).create(runtime)
            root = str(Path(__file__).resolve().parents[1])
            script = (
                "import sys; sys.path.insert(0," + repr(root) + "); "
                "from strategy_runner import SandboxProcess; "
                "worker=SandboxProcess(" + repr(SOURCE) + "); "
                "print(worker.metadata['name']); worker.close()"
            )
            result = subprocess.run(
                [str(runtime / "bin" / "python"), "-B", "-c", script],
                capture_output=True,
                text=True,
                timeout=25,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Test strategy", result.stdout)


if __name__ == "__main__":
    unittest.main()
