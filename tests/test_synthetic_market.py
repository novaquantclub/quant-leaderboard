"""Offline checks for competition data generation and immutable round bundles."""

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from synthetic_market import (
    COLUMNS,
    SyntheticConfig,
    SyntheticRound,
    calibrate_market,
    generate_round,
)


def training_fixture(bars=500):
    rng = np.random.default_rng(1122)
    index = pd.bdate_range("2010-01-01", periods=bars)
    market = rng.normal(0, 0.009, bars)
    result = {}
    for symbol, beta in (("SPY", 1.0), ("QQQ", 1.2), ("GLD", -0.2)):
        gap = rng.normal(0, 0.003, bars)
        session = 0.0002 + beta * market + rng.normal(0, 0.004, bars)
        close = 100 * np.exp(np.cumsum(gap + session))
        opening = close / np.exp(session)
        result[symbol] = pd.DataFrame(
            {
                "Open": opening,
                "High": np.maximum(opening, close)
                * np.exp(rng.uniform(0.001, 0.008, bars)),
                "Low": np.minimum(opening, close)
                * np.exp(-rng.uniform(0.001, 0.008, bars)),
                "Close": close,
                "Volume": rng.lognormal(14, 0.4, bars),
            },
            index=index,
        )
    return result


class SyntheticMarketTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.training = training_fixture()
        cls.model = calibrate_market(cls.training)
        cls.config = SyntheticConfig(paths=3, evaluation_bars=252, warmup_bars=100)
        cls.round = generate_round(cls.model, 654321, cls.config)

    def test_seeded_reproduction_and_input_order(self):
        reversed_training = dict(reversed(list(self.training.items())))
        model = calibrate_market(reversed_training)
        regenerated = generate_round(model, 654321, self.config)
        np.testing.assert_array_equal(self.round.prices, regenerated.prices)
        self.assertEqual(self.round.dataset_id, regenerated.dataset_id)

    def test_generated_bars_are_valid_and_have_no_dates(self):
        for frame in self.round.path(0).values():
            self.assertEqual(list(frame.columns), list(COLUMNS))
            self.assertIsInstance(frame.index, pd.RangeIndex)
            self.assertEqual(list(frame.index), list(range(-100, 252)))
            self.assertTrue(np.isfinite(frame.to_numpy()).all())
            self.assertTrue((frame.to_numpy() > 0).all())
            self.assertTrue((frame.High >= frame[["Open", "Close"]].max(axis=1)).all())
            self.assertTrue((frame.Low <= frame[["Open", "Close"]].min(axis=1)).all())
            self.assertAlmostEqual(frame.loc[0, "Open"], 100)
            self.assertAlmostEqual(frame.loc[-20:-1, "Volume"].mean(), 1)

    def test_warmup_candles_and_volume_are_fresh(self):
        source = self.training["SPY"].iloc[:352]
        generated = self.round.path(0)["SPY"]
        self.assertFalse(
            np.allclose(
                np.diff(np.log(source.Close.iloc[:100])),
                np.diff(np.log(generated.Close.iloc[:100])),
            )
        )
        self.assertFalse(
            np.allclose(
                source.Close / source.Open,
                generated.Close.to_numpy() / generated.Open.to_numpy(),
            )
        )
        self.assertFalse(
            np.allclose(
                source.High / source.Low,
                generated.High.to_numpy() / generated.Low.to_numpy(),
            )
        )
        source_volume = source.Volume / source.Volume.iloc[80:100].mean()
        self.assertFalse(np.allclose(source_volume, generated.Volume))
        # Source prices are not scaled and replayed: their return sequences are
        # unrelated to the generated sequence, including in the warm-up.
        for path_index in range(self.config.paths):
            fresh = self.round.path(path_index)["SPY"]
            correlation = np.corrcoef(
                np.diff(np.log(source.Close)), np.diff(np.log(fresh.Close))
            )[0, 1]
            self.assertLess(abs(correlation), 0.3)

    def test_distinct_paths_and_seeds(self):
        self.assertFalse(np.array_equal(self.round.prices[0], self.round.prices[1]))
        other_seed = generate_round(self.model, 654322, self.config)
        self.assertFalse(np.array_equal(self.round.prices, other_seed.prices))
        self.assertNotEqual(self.round.dataset_id, other_seed.dataset_id)
        first = np.diff(np.log(self.round.path(0)["SPY"].Close))
        second = np.diff(np.log(self.round.path(1)["SPY"].Close))
        self.assertLess(abs(np.corrcoef(first, second)[0, 1]), 0.3)

    def test_more_paths_and_bars_preserve_existing_observations(self):
        extended = generate_round(self.model, 654321, SyntheticConfig(4, 300, 100))
        np.testing.assert_array_equal(self.round.prices, extended.prices[:3, :, :352])

    def test_ticker_selection_does_not_regenerate_the_market(self):
        selected = self.round.path(0, ["SPY"])
        pd.testing.assert_frame_equal(selected["SPY"], self.round.path(0)["SPY"])
        selected["SPY"].iloc[0, 0] = -99
        self.assertGreater(self.round.path(0)["SPY"].iloc[0, 0], 0)
        self.assertFalse(self.round.prices.flags.writeable)
        for symbols in (["OTHER"], ["SPY", "SPY"], []):
            with self.subTest(symbols=symbols), self.assertRaises(ValueError):
                self.round.path(0, symbols)

    def test_freeze_and_reload_without_original_training_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "round-one"
            self.round.save(directory, private_seed=654321)
            loaded = SyntheticRound.load(directory)
            np.testing.assert_array_equal(self.round.prices, loaded.prices)
            self.assertEqual(self.round.dataset_id, loaded.dataset_id)
            self.assertFalse(loaded.prices.flags.writeable)
            self.assertNotIn("seed", loaded.manifest)
            self.assertEqual(
                json.loads((directory / "private_seed.json").read_text())["seed"],
                "654321",
            )
            with self.assertRaises(FileExistsError):
                self.round.save(directory)

    def test_tampered_prices_and_manifest_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "round-one"
            self.round.save(directory)
            altered = self.round.prices.copy()
            altered[0, 0, 0, 4] *= 1.01
            np.savez_compressed(directory / "prices.npz", prices=altered)
            with self.assertRaisesRegex(ValueError, "hash"):
                SyntheticRound.load(directory)
            self.round.save(Path(temporary) / "round-two")
            manifest_file = Path(temporary) / "round-two" / "manifest.json"
            manifest = json.loads(manifest_file.read_text())
            manifest["config"]["paths"] += 1
            manifest_file.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "shape"):
                SyntheticRound.load(manifest_file.parent)

    def test_failed_save_leaves_no_partial_round(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "round-one"
            with self.assertRaisesRegex(ValueError, "seed"):
                self.round.save(destination, private_seed=123)
            self.assertFalse(destination.exists())
            self.assertEqual(list(Path(temporary).iterdir()), [])

    def test_dataset_commitment_detects_manifest_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "round-one"
            self.round.save(directory)
            loaded = SyntheticRound.load(
                directory, expected_dataset_id=self.round.dataset_id
            )
            self.assertEqual(loaded.dataset_id, self.round.dataset_id)
            manifest_file = directory / "manifest.json"
            manifest = json.loads(manifest_file.read_text())
            manifest["training_sha256"] = "0" * 64
            manifest_file.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "committed dataset"):
                SyntheticRound.load(
                    directory, expected_dataset_id=self.round.dataset_id
                )

    def test_cli_freezes_training_csvs_and_keeps_seed_out_of_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            training_directory = Path(temporary) / "training"
            training_directory.mkdir()
            for symbol, frame in self.training.items():
                frame.to_csv(training_directory / f"{symbol}.csv", index_label="Date")
            destination = Path(temporary) / "final"
            seed = "123456789012345678901234567890"
            command = [
                sys.executable,
                "-B",
                str(Path(__file__).resolve().parents[1] / "synthetic_market.py"),
                str(training_directory),
                str(destination),
                "--paths",
                "2",
                "--bars",
                "20",
                "--seed",
                seed,
            ]
            completed = subprocess.run(
                command, capture_output=True, text=True, timeout=30, check=True
            )
            self.assertNotIn(seed, completed.stdout + completed.stderr)
            loaded = SyntheticRound.load(destination)
            self.assertEqual(loaded.prices.shape, (2, 3, 120, 5))
            self.assertIn(loaded.dataset_id, completed.stdout)
            second_attempt = subprocess.run(
                command, capture_output=True, text=True, timeout=30, check=False
            )
            self.assertEqual(second_attempt.returncode, 1)
            self.assertIn("already exists", second_attempt.stderr)
            self.assertEqual(
                SyntheticRound.load(destination).dataset_id, loaded.dataset_id
            )

    def test_training_hash_detects_data_revisions(self):
        revised = copy.deepcopy(self.training)
        revised["SPY"].iloc[0, 4] += 1
        model = calibrate_market(revised)
        self.assertNotEqual(self.model.training_sha256, model.training_sha256)
        self.assertNotEqual(
            self.round.dataset_id, generate_round(model, 654321, self.config).dataset_id
        )

    def test_correlated_assets_and_variable_volatility_survive_generation(self):
        generated = generate_round(self.model, 7711, SyntheticConfig(4, 1500, 100))
        correlations, variance_ratios = [], []
        for path_index in range(4):
            data = generated.path(path_index)
            returns = pd.DataFrame(
                {symbol: np.log(frame.Close).diff() for symbol, frame in data.items()}
            ).dropna()
            correlations.append(returns.SPY.corr(returns.QQQ))
            rolling = returns.SPY.rolling(30).std().dropna()
            variance_ratios.append(rolling.quantile(0.9) / rolling.quantile(0.1))
        self.assertGreater(np.mean(correlations), 0.5)
        self.assertGreater(np.mean(variance_ratios), 1.5)

    def test_single_asset_and_constant_training_are_supported(self):
        constant = pd.DataFrame(
            {
                column: np.full(150, 100 if column != "Volume" else 0)
                for column in COLUMNS
            }
        )
        model = calibrate_market({"SPY": constant})
        generated = generate_round(model, 0, SyntheticConfig(1, 10, 20))
        self.assertTrue(np.isfinite(generated.prices).all())
        self.assertEqual(generated.prices.shape, (1, 1, 30, 5))

    def test_invalid_seeds_and_configuration_are_rejected(self):
        for seed in (-1, 2**128, 1.5, True):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                generate_round(self.model, seed, self.config)
        for config in (
            {"paths": 0},
            {"warmup_bars": 0},
            {"evaluation_bars": 2},
            {"paths": True},
        ):
            with self.subTest(config=config), self.assertRaises(ValueError):
                SyntheticConfig(**config)
        large_model = calibrate_market(
            {f"ASSET{i:02d}": self.training["SPY"] for i in range(20)}
        )
        with self.assertRaises(ValueError):
            generate_round(large_model, 1, SyntheticConfig(256, 2520, 1000))

    def test_invalid_training_is_rejected_without_filling(self):
        invalid = {}
        for name in (
            "missing",
            "nan",
            "zero_price",
            "negative_volume",
            "bad_candle",
            "mismatched_dates",
            "duplicate_dates",
            "unordered_dates",
            "duplicate_columns",
        ):
            frames = copy.deepcopy(self.training)
            frame = frames["SPY"]
            if name == "missing":
                frames["SPY"] = frame.drop(columns="Volume")
            elif name == "nan":
                frame.iloc[3, 0] = np.nan
            elif name == "zero_price":
                frame.iloc[3, 0] = 0
            elif name == "negative_volume":
                frame.iloc[3, 4] = -1
            elif name == "bad_candle":
                frame.iloc[3, 1] = frame.iloc[3, 2] / 2
            elif name == "mismatched_dates":
                frames["SPY"] = frame.iloc[1:]
            elif name == "duplicate_dates":
                frame.index = pd.Index([frame.index[0]] * len(frame))
            elif name == "unordered_dates":
                frames["SPY"] = frame.iloc[::-1]
            elif name == "duplicate_columns":
                frames["SPY"] = pd.concat([frame, frame[["Open"]]], axis=1)
            invalid[name] = frames
        invalid["too_short"] = {"SPY": self.training["SPY"].iloc[:20]}
        invalid["duplicate_symbol"] = {
            "SPY": self.training["SPY"],
            " spy ": self.training["SPY"],
        }
        invalid["empty"] = {}
        for name, frames in invalid.items():
            with self.subTest(case=name), self.assertRaises(ValueError):
                calibrate_market(frames)


if __name__ == "__main__":
    unittest.main()
