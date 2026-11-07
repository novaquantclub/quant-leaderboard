"""Verify observed rates, publication timing, conditional paths and immutable rounds."""

import copy
import io
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from test_synthetic_market import training_fixture

from backtester import Backtester, EngineConfig
from competition import Competition, CompetitionError
from interest_rate import (
    INTEREST_RATE,
    INTEREST_RATE_FILE,
    MarketData,
    align_interest_rate,
)
from prepare_training import save_training
from synthetic_market import (
    SyntheticConfig,
    SyntheticRound,
    calibrate_market,
    generate_round,
    load_training,
)


def rate_fixture(bars=500):
    assets = training_fixture(bars)
    rate = pd.Series(
        np.linspace(0.25, 5.25, bars), index=assets["SPY"].index, name=INTEREST_RATE
    )
    return MarketData(assets, interest_rate=rate)


class InterestRateTests(unittest.TestCase):
    def test_availability_uses_only_past_releases_and_respects_federal_holidays(self):
        source = pd.DataFrame(
            {"DFF": [1.0, 2.0, 3.0, 4.0, 99.0]},
            index=pd.to_datetime(
                ["2023-06-28", "2023-06-29", "2023-06-30", "2023-07-03", "2023-07-05"]
            ),
        )
        calendar = pd.Index(
            ["2023-06-30", "2023-07-03", "2023-07-05", "2023-07-06"], name="Date"
        )
        rate = align_interest_rate(source, calendar)
        np.testing.assert_array_equal(rate, [1, 2, 3, 4])
        self.assertEqual(rate.name, INTEREST_RATE)
        with self.assertRaises(ValueError):
            align_interest_rate(source, pd.Index(["2023-06-27"]))

    def test_rates_are_validated_and_included_in_training_identity(self):
        market = rate_fixture()
        model = calibrate_market(market)
        self.assertIn("interest_rate", model.parameters)
        revised = copy.deepcopy(market)
        revised.interest_rate.iloc[0] += 0.01
        self.assertNotEqual(
            model.training_sha256, calibrate_market(revised).training_sha256
        )
        for invalid in (float("nan"), float("inf"), 101):
            revised.interest_rate.iloc[0] = invalid
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                calibrate_market(revised)
        revised.interest_rate = market.interest_rate.iloc[1:]
        with self.assertRaisesRegex(ValueError, "calendar"):
            calibrate_market(revised)

    def test_zero_and_negative_observed_rates_are_supported(self):
        market = rate_fixture()
        market.interest_rate.iloc[:] = 0
        generate_round(calibrate_market(market), 12, SyntheticConfig(2, 20, 100))
        market.interest_rate.iloc[:] = -0.5
        generate_round(calibrate_market(market), 12, SyntheticConfig(2, 20, 100))

    def test_fresh_seeded_rates_and_prices_reproduce_and_extend(self):
        market = rate_fixture()
        model = calibrate_market(market)
        config = SyntheticConfig(2, 252, 100)
        first = generate_round(model, 76543, config)
        second = generate_round(model, 76543, config)
        np.testing.assert_array_equal(first.prices, second.prices)
        np.testing.assert_array_equal(first.interest_rates, second.interest_rates)
        self.assertFalse(first.interest_rates.flags.writeable)
        self.assertFalse(
            np.array_equal(first.interest_rates[0], first.interest_rates[1])
        )
        self.assertFalse(
            np.array_equal(first.interest_rates[0], market.interest_rate.iloc[:352])
        )
        extended = generate_round(model, 76543, SyntheticConfig(3, 300, 100))
        np.testing.assert_array_equal(
            first.interest_rates, extended.interest_rates[:2, :352]
        )
        np.testing.assert_array_equal(first.prices, extended.prices[:2, :, :352])
        subset = first.path(0, ["SPY"])
        subset.interest_rate.iloc[0] = -99
        self.assertNotEqual(first.path(0).interest_rate.iloc[0], -99)
        self.assertEqual(first.manifest["symbols"], ["GLD", "QQQ", "SPY"])

    def test_rate_dependent_asset_correlations_are_calibrated_and_generated(self):
        rng = np.random.default_rng(78)
        bars = 2000
        index = pd.bdate_range("2007-01-01", periods=bars)
        rate = np.where(np.arange(bars) < bars // 2, 0.5, 5.0)
        common = rng.normal(0, 0.008, bars)
        market = MarketData(
            {}, interest_rate=pd.Series(rate, index=index, name=INTEREST_RATE)
        )
        for symbol, sign in (
            ("LEFT", np.ones(bars)),
            ("RIGHT", np.where(rate < 1, 1, -1)),
        ):
            session = common * sign + rng.normal(0, 0.002, bars)
            gap = rng.normal(0, 0.001, bars)
            close = 100 * np.exp(np.cumsum(gap + session))
            opening = close / np.exp(session)
            market[symbol] = pd.DataFrame(
                {
                    "Open": opening,
                    "High": np.maximum(opening, close) * 1.002,
                    "Low": np.minimum(opening, close) / 1.002,
                    "Close": close,
                    "Volume": np.full(bars, 1_000_000.0),
                },
                index=index,
            )
        model = calibrate_market(market)
        generated = generate_round(model, 9375, SyntheticConfig(8, 800, 100))
        groups = [[], []]
        threshold = model.parameters["interest_rate"]["threshold"]
        for path_index in range(8):
            path = generated.path(path_index)
            returns = pd.DataFrame(
                {symbol: np.log(frame.Close).diff() for symbol, frame in path.items()}
            )
            group = (path.interest_rate.shift(1) > threshold).astype(int)
            for regime in range(2):
                groups[regime].append(returns.loc[group == regime].dropna())
        low, high = (pd.concat(frames) for frames in groups)
        self.assertGreater(low.LEFT.corr(low.RIGHT), 0.5)
        self.assertLess(high.LEFT.corr(high.RIGHT), -0.5)

    def test_engine_exposes_only_current_observations_in_percent(self):
        market = rate_fixture(130)
        observed = []

        class Observer:
            tickers = ("SPY",)

            def run(self, state):
                rate = state.observations[INTEREST_RATE]
                assert len(rate) == 101 + state.step
                assert rate.index[-1] == state.step
                assert INTEREST_RATE not in state.positions
                observed.append(rate.iloc[-1])
                rate.iloc[:] = -99
                return {}

        Backtester().run_dataset(Observer(), market)
        np.testing.assert_array_equal(observed, market.interest_rate.iloc[100:])

        class RateBuyer(Observer):
            def run(self, state):
                return {INTEREST_RATE: 1}

        with self.assertRaises(RuntimeError):
            Backtester().run_dataset(RateBuyer(), market)

    def test_snapshots_hash_rates_and_load_legacy_datasets(self):
        generated = generate_round(
            calibrate_market(rate_fixture()), 1234, SyntheticConfig(2, 20, 100)
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generated.save(root / "with-rate")
            loaded = SyntheticRound.load(
                root / "with-rate", expected_dataset_id=generated.dataset_id
            )
            np.testing.assert_array_equal(
                loaded.interest_rates, generated.interest_rates
            )
            self.assertFalse(loaded.interest_rates.flags.writeable)
            changed = loaded.interest_rates.copy()
            changed[0, 0] += 0.01
            np.savez_compressed(
                root / "with-rate" / "prices.npz",
                prices=loaded.prices,
                interest_rates=changed,
            )
            with self.assertRaisesRegex(ValueError, "interest-rate hash"):
                SyntheticRound.load(root / "with-rate")
            legacy = generate_round(
                calibrate_market(training_fixture()), 1234, SyntheticConfig(2, 20, 100)
            )
            legacy.manifest["generator_version"] = "synthetic-market-v1"
            legacy.save(root / "legacy")
            self.assertIsNone(SyntheticRound.load(root / "legacy").interest_rates)

    def test_round_freezes_downloads_and_checks_interest_rate_revisions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_training(rate_fixture(130), root / "training")
            market = load_training(root / "training")
            self.assertEqual(len(market), 3)
            self.assertIsNotNone(market.interest_rate)
            store = Competition(root / "league.db", root / "rounds")
            round_id = store.create_round(
                "Rates",
                root / "training",
                EngineConfig(runtime_id="test-worker"),
                SyntheticConfig(2, 20, 100),
            )
            with zipfile.ZipFile(io.BytesIO(store.training_zip(round_id))) as archive:
                self.assertIn(INTEREST_RATE_FILE, archive.namelist())
                self.assertIn(
                    "annualized percentages", archive.read("README.md").decode()
                )
            public = Path(store.round(round_id)["public_dir"]) / INTEREST_RATE_FILE
            frame = pd.read_csv(public)
            frame.loc[0, INTEREST_RATE] += 0.01
            frame.to_csv(public, index=False)
            with self.assertRaisesRegex(CompetitionError, "changed"):
                store.public_market(round_id)


if __name__ == "__main__":
    unittest.main()
