"""Check the exact public starter bundle independently of the application UI."""

import io
import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import pandas as pd
from test_interest_rate import rate_fixture

from backtester import Backtester, EngineConfig, load_local_trader, load_training
from member_package import member_zip, sdk_zip
from prepare_training import save_training


class MemberPackageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.training = self.root / "source"
        save_training(
            rate_fixture(130), self.training, asset_names=["ATLAS", "BOREAL", "CIRRUS"]
        )

    def test_complete_bundle_contains_only_member_inputs_and_an_independent_workspace(
        self,
    ):
        (self.training / "private_seed.json").write_text("private-marker")
        with zipfile.ZipFile(io.BytesIO(member_zip(self.training))) as archive:
            expected = {
                "README.md",
                "requirements.txt",
                "engine_config.json",
                "sample_strategy.py",
                "backtester.py",
                "synthetic_market.py",
                "interest_rate.py",
                "strategy_runner.py",
                "nova-quant-member-starter.code-workspace",
                "training/ATLAS.csv",
                "training/BOREAL.csv",
                "training/CIRRUS.csv",
                "training/interest rate.csv",
            }
            self.assertEqual(set(archive.namelist()), expected)
            self.assertEqual(
                json.loads(archive.read("nova-quant-member-starter.code-workspace")),
                {"folders": [{"path": "."}]},
            )
            self.assertNotIn(
                b"private-marker",
                b"".join(archive.read(name) for name in archive.namelist()),
            )
            self.assertNotIn("{{", archive.read("README.md").decode())
            self.assertNotIn(
                "runtime_id", json.loads(archive.read("engine_config.json"))
            )
            self.assertFalse(any("strategy.py" == name for name in archive.namelist()))
            for name in ("ATLAS", "BOREAL", "CIRRUS"):
                original = load_training(self.training)[name]
                bundled = pd.read_csv(
                    io.BytesIO(archive.read(f"training/{name}.csv")),
                    index_col=0,
                    float_precision="round_trip",
                )
                pd.testing.assert_frame_equal(original, bundled)

    def test_frozen_settings_and_documentation_follow_the_selected_engine(self):
        config = EngineConfig(
            initial_cash=12_345,
            commission_bps=25,
            risk_free_rate=0.02,
            benchmark="BOREAL",
            warmup_bars=80,
            runtime_id="organizer-runtime-marker",
        )
        with zipfile.ZipFile(
            io.BytesIO(sdk_zip(["ATLAS", "BOREAL"], config=config))
        ) as archive:
            text = archive.read("README.md").decode()
            self.assertIn("12,345.00", text)
            self.assertIn("25 basis points", text)
            self.assertIn("-80", text)
            self.assertIn("--benchmark BOREAL", text)
            self.assertIn("reference of 2%", text)
            self.assertNotIn("organizer-runtime-marker", text)
            settings = json.loads(archive.read("engine_config.json"))
            self.assertEqual(settings["benchmark"], "BOREAL")
            self.assertEqual(settings["commission_bps"], 25)

    def test_extracted_package_runs_the_unchanged_example_with_its_settings(self):
        target = self.root / "member-workspace"
        config = EngineConfig(initial_cash=12_345, commission_bps=25, benchmark="ATLAS")
        with zipfile.ZipFile(io.BytesIO(member_zip(self.training, config))) as archive:
            archive.extractall(target)
        expected = Backtester(config).run_dataset(
            load_local_trader(target / "sample_strategy.py"),
            load_training(target / "training"),
        )
        result = subprocess.run(
            [
                sys.executable,
                "-B",
                "backtester.py",
                "sample_strategy.py",
                "--data",
                "training",
                "--config",
                "engine_config.json",
                "--trusted-local",
            ],
            cwd=target,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        actual = json.loads(result.stdout)
        self.assertAlmostEqual(actual["final_equity"], expected["final_equity"])
        self.assertAlmostEqual(actual["sharpe"], expected["sharpe"])

    def test_cli_builds_a_new_package_and_preserves_an_existing_output(self):
        target = self.root / "member.zip"
        command = [
            sys.executable,
            "-B",
            "member_package.py",
            str(self.training),
            str(target),
        ]
        subprocess.run(
            command,
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            timeout=30,
        )
        original = target.read_bytes()
        second = subprocess.run(
            command,
            cwd=Path(__file__).resolve().parents[1],
            check=False,
            capture_output=True,
            timeout=30,
        )
        self.assertNotEqual(second.returncode, 0)
        self.assertEqual(target.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
