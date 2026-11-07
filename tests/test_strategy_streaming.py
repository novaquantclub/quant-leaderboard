"""Preserve history semantics while sending each newly available bar once."""

import copy
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from test_synthetic_market import training_fixture

from backtester import TradingState
from strategy_bootstrap import HistoryBuffer
from strategy_runner import RemoteTrader


class HistoryStreamingTests(unittest.TestCase):
    def setUp(self):
        self.frame = training_fixture(450)["SPY"]
        self.requests = []
        self.states = []
        self.buffer = HistoryBuffer()
        self.incremental = True

    def state(self, step):
        frame = self.frame.iloc[: 101 + step].copy()
        frame.index = pd.RangeIndex(-100, step + 1)
        rate = pd.Series(
            np.linspace(0, 5, 450)[: 101 + step],
            index=frame.index,
            name="interest rate",
        )
        return TradingState(
            step,
            {"SPY": frame},
            {"SPY": 0},
            100_000,
            100_000,
            observations={"interest rate": rate},
        )

    def request(self, path, body):
        if path == "/sessions":
            return {
                "session": "test-session",
                "metadata": {"name": "Test", "author": "Test", "tickers": ["SPY"]},
                "history_updates": self.incremental,
            }
        if path.endswith("/close"):
            return {"closed": True}
        self.requests.append(copy.deepcopy(body))
        self.states.append(self.buffer.restore(copy.deepcopy(body)))
        return {"orders": {}, "trader_data": ""}

    def client(self):
        return RemoteTrader(
            "source", endpoint="http://127.0.0.1", token="test-token-" * 4
        )

    def test_each_bar_is_sent_once_and_full_history_remains_available(self):
        with (
            patch.object(RemoteTrader, "_request", side_effect=self.request),
            self.client() as client,
        ):
            for step in range(300):
                client.run(self.state(step))
        self.assertEqual(
            sum(len(body["history"]["SPY"]) for body in self.requests), 400
        )
        self.assertEqual(len(self.requests[0]["history"]["SPY"]), 101)
        self.assertEqual(self.requests[1]["history_update_start"], 1)
        self.assertEqual(len(self.states[-1]["history"]["SPY"]), 400)
        self.assertEqual(self.states[-1]["history"]["SPY"].index[-1], 299)
        self.assertNotIn("history_update_start", self.states[-1])
        expected = self.state(299)
        pd.testing.assert_frame_equal(
            self.states[-1]["history"]["SPY"], expected.history["SPY"]
        )
        pd.testing.assert_series_equal(
            self.states[-1]["observations"]["interest rate"],
            expected.observations["interest rate"],
        )

    def test_revised_history_or_rates_are_sent_as_a_full_replacement(self):
        with (
            patch.object(RemoteTrader, "_request", side_effect=self.request),
            self.client() as client,
        ):
            client.run(self.state(0))
            revised = self.state(1)
            revised.history["SPY"].iloc[0, 0] += 1
            client.run(revised)
            revised = self.state(2)
            revised.observations["interest rate"].iloc[0] = -0.5
            client.run(revised)
        self.assertNotIn("history_update_start", self.requests[1])
        self.assertNotIn("history_update_start", self.requests[2])
        self.assertEqual(self.states[-1]["observations"]["interest rate"].iloc[0], -0.5)

    def test_older_workers_receive_complete_states(self):
        self.incremental = False
        with (
            patch.object(RemoteTrader, "_request", side_effect=self.request),
            self.client() as client,
        ):
            client.run(self.state(0))
            client.run(self.state(1))
        self.assertEqual(len(self.requests[-1]["history"]["SPY"]), 102)
        self.assertNotIn("history_update_start", self.requests[-1])

    def test_strategy_mutation_does_not_change_the_canonical_received_prefix(self):
        with (
            patch.object(RemoteTrader, "_request", side_effect=self.request),
            self.client() as client,
        ):
            client.run(self.state(0))
            self.states[0]["history"]["SPY"].iloc[:, :] = -99
            self.states[0]["observations"]["interest rate"].iloc[:] = -99
            client.run(self.state(1))
        expected = self.state(1)
        pd.testing.assert_frame_equal(
            self.states[-1]["history"]["SPY"], expected.history["SPY"]
        )
        pd.testing.assert_series_equal(
            self.states[-1]["observations"]["interest rate"],
            expected.observations["interest rate"],
        )

    def test_repeated_steps_resend_the_complete_state(self):
        with (
            patch.object(RemoteTrader, "_request", side_effect=self.request),
            self.client() as client,
        ):
            client.run(self.state(0))
            client.run(self.state(1))
            client.run(self.state(0))
        self.assertNotIn("history_update_start", self.requests[-1])
        self.assertEqual(len(self.states[-1]["history"]["SPY"]), 101)

    def test_updates_preserve_incoming_asset_order(self):
        first = self.state(0)
        first.history["QQQ"] = first.history["SPY"].copy()
        second = self.state(1)
        second.history = {
            "QQQ": second.history["SPY"].copy(),
            "SPY": second.history["SPY"],
        }
        with (
            patch.object(RemoteTrader, "_request", side_effect=self.request),
            self.client() as client,
        ):
            client.run(first)
            client.run(second)
        self.assertEqual(list(self.states[-1]["history"]), ["QQQ", "SPY"])
        self.assertIn("history_update_start", self.requests[-1])

    def test_noncontiguous_updates_and_misaligned_observations_are_rejected(self):
        body = {
            "step": 0,
            "history_start": -100,
            "history": {"SPY": self.frame.iloc[:101].to_numpy().tolist()},
            "observations": {"interest rate": [1.0] * 101},
        }
        for start, observations in ((2, [1.0]), (1, [1.0, 2.0])):
            with self.subTest(start=start):
                buffer = HistoryBuffer()
                buffer.restore(copy.deepcopy(body))
                update = {
                    "step": 1,
                    "history_start": -100,
                    "history_update_start": start,
                    "history": {"SPY": self.frame.iloc[101:102].to_numpy().tolist()},
                    "observations": {"interest rate": observations},
                }
                with self.assertRaises(ValueError):
                    buffer.restore(update)


if __name__ == "__main__":
    unittest.main()
