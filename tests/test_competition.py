import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from test_synthetic_market import training_fixture

from backtester import Backtester, EngineConfig, evaluation_id
from competition import Competition, CompetitionError
from strategy_runner import StrategyError, WorkerBusyError, WorkerError


class BuyHold:
    tickers = ("SPY",)

    def run(self, state):
        return {"SPY": 1_000_000} if state.step == 0 else {}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class EngineTests(unittest.TestCase):
    def test_supporting_module_changes_invalidate_the_evaluation_identity(self):
        base = evaluation_id("dataset-one", EngineConfig())
        original = Path.read_text
        for name in (
            "market_data.py",
            "synthetic_market.py",
            "secure_random.py",
            "strategy_runner.py",
        ):

            def revised(path, *args, module_name=name, **kwargs):
                text = original(path, *args, **kwargs)
                return (
                    text + "\n# changed support module\n"
                    if path.name == module_name
                    else text
                )

            with self.subTest(module=name), patch.object(Path, "read_text", revised):
                self.assertNotEqual(base, evaluation_id("dataset-one", EngineConfig()))

    def test_buy_hold_has_identical_opportunities_and_costs_to_benchmark(self):
        result = Backtester().run_dataset(BuyHold(), training_fixture(130))
        pd.testing.assert_series_equal(
            result["equity_df"].Strategy,
            result["equity_df"].SPY_Benchmark,
            check_names=False,
        )
        self.assertEqual(result["equity_df"].iloc[0].tolist(), [100_000, 100_000])
        self.assertAlmostEqual(result["sharpe"], result["spy_sharpe"])
        self.assertAlmostEqual(result["beta"], 1)

    def test_history_is_current_prefix_and_mutation_does_not_change_prices(self):
        class Mutator(BuyHold):
            def run(self, state):
                assert len(state.history["SPY"]) == 101 + state.step
                assert state.history["SPY"].index[-1] == state.step
                state.history["SPY"].iloc[:, :] = -1
                return super().run(state)

        reference = Backtester().run_dataset(BuyHold(), training_fixture(130))
        changed = Backtester().run_dataset(Mutator(), training_fixture(130))
        pd.testing.assert_frame_equal(reference["equity_df"], changed["equity_df"])

    def test_unknown_tickers_and_missing_calendars_are_rejected(self):
        market = training_fixture(130)
        market["SPY"] = market["SPY"].iloc[1:]
        with self.assertRaises(ValueError):
            Backtester().run_dataset(BuyHold(), market)
        with self.assertRaises(StrategyError):
            Backtester().run_dataset(BuyHold(), {"QQQ": market["QQQ"]})

    def test_all_engine_settings_affect_evaluation_identity(self):
        config = EngineConfig()
        base = evaluation_id("dataset-one", config)
        for field, value in (
            ("initial_cash", 50_000),
            ("commission_bps", 20),
            ("risk_free_rate", 0.01),
            ("benchmark", "QQQ"),
            ("warmup_bars", 80),
            ("runtime_id", "other"),
        ):
            with self.subTest(field=field):
                self.assertNotEqual(
                    base,
                    evaluation_id("dataset-one", replace(config, **{field: value})),
                )
        self.assertNotEqual(base, evaluation_id("dataset-two", config))

    def test_each_final_path_pins_runtime_and_checks_it_again_after_evaluation(self):
        from synthetic_market import SyntheticConfig, calibrate_market, generate_round

        dataset = generate_round(
            calibrate_market(training_fixture(130)), 12, SyntheticConfig(2, 15, 100)
        )
        engine = Backtester(EngineConfig(runtime_id="pinned-test-runtime"))
        with (
            patch("backtester.RemoteTrader", return_value=BuyHold()) as factory,
            patch("backtester.worker_runtime_id", return_value="pinned-test-runtime"),
        ):
            engine.run_synthetic("source", dataset)
            self.assertEqual(factory.call_count, 2)
            for call in factory.call_args_list:
                self.assertEqual(
                    call.kwargs,
                    {"purpose": "final", "expected_runtime_id": "pinned-test-runtime"},
                )
        with (
            patch("backtester.RemoteTrader", return_value=BuyHold()),
            patch("backtester.worker_runtime_id", return_value="changed-test-runtime"),
            self.assertRaises(WorkerError),
        ):
            engine.run_synthetic("source", dataset)

    def test_non_finite_orders_are_rejected(self):
        class BadOrders(BuyHold):
            def run(self, state):
                return {"SPY": float("nan")}

        with self.assertRaises(RuntimeError):
            Backtester().run_dataset(BadOrders(), training_fixture(130))

    def test_competition_names_work_with_neutral_benchmark_metrics(self):
        market = {"ATLAS": training_fixture(130)["SPY"]}

        class NamedAsset(BuyHold):
            tickers = ("ATLAS",)

            def run(self, state):
                return {"ATLAS": 1_000_000} if state.step == 0 else {}

        result = Backtester(EngineConfig(benchmark="ATLAS")).run_dataset(
            NamedAsset(), market
        )
        self.assertEqual(list(result["equity_df"].columns), ["Strategy", "Benchmark"])
        self.assertAlmostEqual(result["sharpe"], result["benchmark_sharpe"])
        self.assertFalse(any(name.startswith("spy_") for name in result))

    def test_long_only_and_read_only_signals(self):
        class Seller(BuyHold):
            tickers = ("SPY", "^SIGNAL")

            def run(self, state):
                return {"SPY": -999, "^SIGNAL": 999}

        market = training_fixture(130)
        market["^SIGNAL"] = market["SPY"].copy()
        result = Backtester().run_dataset(Seller(), market)
        np.testing.assert_array_equal(
            result["equity_df"].Strategy, np.full(30, 100_000.0)
        )


class CompetitionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        training = self.root / "source"
        training.mkdir()
        for symbol, frame in training_fixture(130).items():
            frame.to_csv(training / f"{symbol}.csv", index_label="Date")
        self.store = Competition(self.root / "league.db", self.root / "rounds")
        self.token_a, self.token_b = (
            self.store.create_team("Team A"),
            self.store.create_team("Team B"),
        )
        self.team_a, self.team_b = (
            self.store.authenticate(self.token_a),
            self.store.authenticate(self.token_b),
        )
        from synthetic_market import SyntheticConfig

        self.round_id = self.store.create_round(
            "Round 1",
            training,
            EngineConfig(runtime_id="test-worker"),
            SyntheticConfig(2, 15, 100),
            max_submissions=2,
            cooldown_seconds=0,
        )
        self.metadata = {"name": "Strategy", "author": "Team B", "tickers": ["SPY"]}

    def submit(self, team=None, source="original source", metadata=None):
        return self.store.submit(
            self.round_id,
            (team or self.team_a)["id"],
            source,
            metadata or self.metadata,
        )

    def test_identity_comes_from_token_and_tokens_are_hashed(self):
        self.assertNotEqual(self.token_a, self.token_b)
        self.assertIsNone(self.store.authenticate("wrong"))
        version = self.submit()
        self.assertEqual(
            self.store.versions(self.round_id, self.team_a["id"])[0]["id"], version
        )
        self.assertEqual(self.store.versions(self.round_id, self.team_b["id"]), [])
        self.assertNotIn(self.token_a.encode(), (self.root / "league.db").read_bytes())
        replacement = self.store.rotate_token(self.team_a["id"])
        self.assertIsNone(self.store.authenticate(self.token_a))
        self.assertEqual(self.store.authenticate(replacement)["id"], self.team_a["id"])

    def test_import_side_effect_is_never_executed_by_storage(self):
        marker = self.root / "must-not-exist"
        self.submit(source=f"open({str(marker)!r}, 'w').write('executed')")
        self.assertFalse(marker.exists())

    def test_name_changes_do_not_reset_limits_and_exact_retries_are_idempotent(self):
        first = self.submit()
        self.assertEqual(first, self.submit())
        changed = dict(self.metadata, name="Another name", author="Someone else")
        second = self.submit(source="variant", metadata=changed)
        self.assertNotEqual(first, second)
        with self.assertRaisesRegex(CompetitionError, "limit"):
            self.submit(source="third variant")

    def test_cooldown_is_team_scoped_and_measured_in_epoch_seconds(self):
        self.submit()
        with self.store.db(write=True) as db:
            db.execute(
                "UPDATE competition_rounds SET cooldown_seconds=3600 WHERE id=?",
                (self.round_id,),
            )
        with self.assertRaisesRegex(CompetitionError, "cooldown"):
            self.submit(
                source="different",
                metadata=dict(self.metadata, name="New", author="Other"),
            )
        self.submit(team=self.team_b)

    def test_cross_team_selection_and_late_mutations_are_rejected(self):
        version = self.submit()
        with self.assertRaises(CompetitionError):
            self.store.select(self.round_id, self.team_b["id"], version)
        self.store.select(self.round_id, self.team_a["id"], version)
        self.store.lock_round(self.round_id)
        with self.assertRaises(CompetitionError):
            self.submit(source="late")
        with self.assertRaises(CompetitionError):
            self.store.select(self.round_id, self.team_a["id"], version)

    def test_blind_round_reveals_no_scores_before_publication(self):
        version = self.submit()
        self.store.select(self.round_id, self.team_a["id"], version)
        self.assertEqual(self.store.results(self.round_id), [])
        with self.assertRaises(CompetitionError):
            self.store.score(
                self.round_id,
                runtime_id="test-worker",
                trader_factory=lambda source: BuyHold(),
            )
        with self.assertRaises(CompetitionError):
            self.store.publish(self.round_id)
        self.store.lock_round(self.round_id)
        self.store.score(
            self.round_id,
            runtime_id="test-worker",
            trader_factory=lambda source: BuyHold(),
        )
        self.assertEqual(self.store.round(self.round_id)["phase"], "scored")
        self.assertEqual(self.store.results(self.round_id), [])
        stored = json.loads(
            self.store.results(self.round_id, organizer=True)[0]["result"]
        )
        self.assertEqual(stored["paths_tested"], 2)
        self.assertEqual(stored["score"], np.median(stored["path_sharpes"]))
        self.store.publish(self.round_id)
        self.assertEqual(len(self.store.results(self.round_id)), 1)
        with self.assertRaises(CompetitionError):
            self.store.score(
                self.round_id,
                runtime_id="test-worker",
                trader_factory=lambda source: BuyHold(),
            )

    def test_runtime_and_snapshot_changes_cannot_silently_change_a_score(self):
        version = self.submit()
        self.store.select(self.round_id, self.team_a["id"], version)
        self.store.lock_round(self.round_id)
        with self.assertRaisesRegex(CompetitionError, "runtime"):
            self.store.score(
                self.round_id,
                runtime_id="new-worker",
                trader_factory=lambda source: BuyHold(),
            )
        manifest_path = (
            Path(self.store.round(self.round_id)["private_dir"]) / "manifest.json"
        )
        manifest = json.loads(manifest_path.read_text())
        manifest["training_sha256"] = "0" * 64
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "committed"):
            self.store.score(
                self.round_id,
                runtime_id="test-worker",
                trader_factory=lambda source: BuyHold(),
            )

    def test_training_download_excludes_final_data_and_detects_revisions(self):
        import io
        import zipfile

        with zipfile.ZipFile(
            io.BytesIO(self.store.training_zip(self.round_id))
        ) as archive:
            self.assertEqual(
                set(archive.namelist()), {"SPY.csv", "QQQ.csv", "GLD.csv", "README.md"}
            )
            self.assertTrue(archive.read("SPY.csv").startswith(b"Date,"))
        public = Path(self.store.round(self.round_id)["public_dir"]) / "SPY.csv"
        frame = pd.read_csv(public)
        frame.loc[0, "Volume"] += 1
        frame.to_csv(public, index=False)
        with self.assertRaisesRegex(CompetitionError, "changed"):
            self.store.public_market(self.round_id)

    def test_failed_strategy_is_recorded_privately_and_can_be_retried(self):
        version = self.submit()
        self.store.select(self.round_id, self.team_a["id"], version)
        self.store.lock_round(self.round_id)

        def fail(source):
            raise StrategyError("Bad strategy")

        self.store.score(self.round_id, runtime_id="test-worker", trader_factory=fail)
        self.assertEqual(self.store.results(self.round_id), [])
        self.assertIn(
            "Bad strategy",
            self.store.results(self.round_id, organizer=True)[0]["error"],
        )
        self.store.score(
            self.round_id,
            runtime_id="test-worker",
            trader_factory=lambda source: BuyHold(),
        )
        self.assertIsNone(self.store.results(self.round_id, organizer=True)[0]["error"])

    def test_strategy_changing_its_declared_universe_is_a_member_error(self):
        version = self.submit()
        self.store.select(self.round_id, self.team_a["id"], version)
        self.store.lock_round(self.round_id)

        class ChangedUniverse(BuyHold):
            tickers = ("UNKNOWN",)

        self.store.score(
            self.round_id,
            runtime_id="test-worker",
            trader_factory=lambda source: ChangedUniverse(),
        )
        self.assertEqual(self.store.round(self.round_id)["phase"], "scored")
        self.assertIn(
            "StrategyError",
            self.store.results(self.round_id, organizer=True)[0]["error"],
        )

    def test_busy_worker_keeps_results_pending_and_the_round_retryable(self):
        version = self.submit()
        self.store.select(self.round_id, self.team_a["id"], version)
        self.store.lock_round(self.round_id)

        def busy(source):
            raise WorkerBusyError("Worker is busy.")

        with self.assertRaises(WorkerBusyError):
            self.store.score(
                self.round_id, runtime_id="test-worker", trader_factory=busy
            )
        row = self.store.round(self.round_id)
        self.assertEqual(row["phase"], "locked")
        self.assertIsNone(row["lease"])
        result = self.store.results(self.round_id, organizer=True)[0]
        self.assertIsNone(result["result"])
        self.assertIsNone(result["error"])
        with self.assertRaises(CompetitionError):
            self.store.publish(self.round_id)
        self.store.score(
            self.round_id,
            runtime_id="test-worker",
            trader_factory=lambda source: BuyHold(),
        )
        self.store.publish(self.round_id)
        self.assertIsNotNone(self.store.results(self.round_id)[0]["result"])

    def test_controller_error_releases_lease_and_preserves_pending_selection(self):
        version = self.submit()
        self.store.select(self.round_id, self.team_a["id"], version)
        self.store.lock_round(self.round_id)

        def controller_error(source):
            raise ValueError("Controller failure.")

        with self.assertRaisesRegex(ValueError, "Controller failure"):
            self.store.score(
                self.round_id, runtime_id="test-worker", trader_factory=controller_error
            )
        self.assertEqual(self.store.round(self.round_id)["phase"], "locked")
        self.assertIsNone(self.store.results(self.round_id, organizer=True)[0]["error"])

    def test_interrupted_grading_preserves_completed_teams_and_resumes_pending_work(
        self,
    ):
        for team in (self.team_a, self.team_b):
            version = self.submit(team=team)
            self.store.select(self.round_id, team["id"], version)
        self.store.lock_round(self.round_id)
        calls = []

        def interrupted(source):
            calls.append(source)
            if len(calls) > 2:
                raise WorkerBusyError("Worker is busy.")
            return BuyHold()

        with self.assertRaises(WorkerBusyError):
            self.store.score(
                self.round_id, runtime_id="test-worker", trader_factory=interrupted
            )
        rows = self.store.results(self.round_id, organizer=True)
        completed = next(row for row in rows if row["result"])
        self.assertEqual(sum(row["result"] is not None for row in rows), 1)
        self.assertTrue(all(row["error"] is None for row in rows))
        self.assertEqual(self.store.round(self.round_id)["phase"], "locked")
        calls.clear()

        def resumed(source):
            calls.append(source)
            return BuyHold()

        self.store.score(
            self.round_id, runtime_id="test-worker", trader_factory=resumed
        )
        self.assertEqual(len(calls), 2)
        after = self.store.results(self.round_id, organizer=True)
        self.assertEqual(
            next(row for row in after if row["team"] == completed["team"])["result"],
            completed["result"],
        )
        self.assertTrue(all(row["result"] for row in after))


if __name__ == "__main__":
    unittest.main()
