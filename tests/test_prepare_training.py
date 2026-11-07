"""Check neutral asset names through preparation, downloads and local practice."""

import hashlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from test_synthetic_market import training_fixture

from backtester import EngineConfig, load_training
from competition import Competition
from prepare_training import repair_ohlc_roundoff, save_training
from synthetic_market import SyntheticConfig


class TrainingNamesTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.training = training_fixture(130)
        self.names = ("ATLAS", "BOREAL", "CIRRUS")

    def test_adjustment_roundoff_is_repaired_but_larger_errors_still_fail(self):
        frame = self.training["SPY"].copy()
        first, second = frame.index[:2]
        high = frame.loc[first, ["Open", "Close"]].max()
        low = frame.loc[second, ["Open", "Close"]].min()
        frame.loc[first, "High"] = np.nextafter(high, -np.inf)
        frame.loc[second, "Low"] = np.nextafter(low, np.inf)
        repaired = repair_ohlc_roundoff(frame)
        self.assertEqual(repaired.loc[first, "High"], high)
        self.assertEqual(repaired.loc[second, "Low"], low)
        self.assertLess(frame.loc[first, "High"], high)
        save_training({"SPY": repaired}, self.root / "valid")
        frame.loc[first, "High"] = high * (1 - 1e-6)
        with self.assertRaisesRegex(ValueError, "OHLC ordering"):
            save_training({"SPY": repair_ohlc_roundoff(frame)}, self.root / "invalid")

    def test_saved_names_and_private_provenance_preserve_source_data(self):
        output = self.root / "training"
        self.assertEqual(
            save_training(self.training, output, asset_names=self.names), self.names
        )
        self.assertEqual({file.stem for file in output.glob("*.csv")}, set(self.names))
        provenance = json.loads((self.root / "training.private.json").read_text())
        market = load_training(output)
        for name, source in provenance["source_tickers"].items():
            pd.testing.assert_frame_equal(
                market[name].reset_index(drop=True),
                self.training[source].reset_index(drop=True),
            )
            self.assertEqual(
                provenance["files_sha256"][f"{name}.csv"],
                hashlib.sha256((output / f"{name}.csv").read_bytes()).hexdigest(),
            )
        self.assertFalse(list(output.glob("*.json")))
        with self.assertRaises(FileExistsError):
            save_training(self.training, output, asset_names=self.names)

    def test_name_validation_rejects_collisions_and_paths(self):
        for names in (
            ["ATLAS"],
            ["ATLAS", "ATLAS", "CIRRUS"],
            ["../ATLAS", "BOREAL", "CIRRUS"],
            ["^ATLAS", "BOREAL", "CIRRUS"],
        ):
            with self.subTest(names=names), self.assertRaises(ValueError):
                save_training(self.training, self.root / "bad", asset_names=names)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_failed_publish_cleans_its_new_provenance(self):
        output = self.root / "training"
        with (
            patch.object(Path, "rename", side_effect=OSError("test failure")),
            self.assertRaises(OSError),
        ):
            save_training(self.training, output, asset_names=self.names)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_training_download_has_neutral_csvs_and_research_instructions(self):
        output = self.root / "training"
        save_training(self.training, output, asset_names=self.names)
        store = Competition(self.root / "league.db", self.root / "rounds")
        round_id = store.create_round(
            "Named round",
            output,
            EngineConfig(benchmark="ATLAS", runtime_id="test-worker"),
            SyntheticConfig(2, 15, 100),
        )
        with zipfile.ZipFile(io.BytesIO(store.training_zip(round_id))) as archive:
            self.assertEqual(
                set(archive.namelist()),
                {f"{name}.csv" for name in self.names} | {"README.md"},
            )
            guide = archive.read("README.md").decode()
            self.assertIn("Investigate asset behaviour and relationships", guide)
            for symbol in self.training:
                self.assertNotIn(symbol, guide)
            self.assertTrue(archive.read("ATLAS.csv").startswith(b"Date,"))

    def test_local_cli_accepts_competition_benchmark(self):
        output = self.root / "training"
        save_training(self.training, output, asset_names=self.names)
        strategy = self.root / "strategy.py"
        strategy.write_text(
            "class Trader:\n    tickers = ('ATLAS',)\n"
            "    def run(self, state):\n        return {}\n"
        )
        result = subprocess.run(
            [
                sys.executable,
                "-B",
                "backtester.py",
                str(strategy),
                "--data",
                str(output),
                "--benchmark",
                "ATLAS",
                "--trusted-local",
            ],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            check=True,
        )
        metrics = json.loads(result.stdout)
        self.assertIn("benchmark_sharpe", metrics)
        self.assertNotIn("spy_sharpe", metrics)


if __name__ == "__main__":
    unittest.main()
