import ast
import importlib.util
import io
import os
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch


@unittest.skipUnless(
    importlib.util.find_spec("streamlit"),
    "Streamlit UI checks run in the web app environment",
)
class AppTests(unittest.TestCase):
    def test_blind_round_hides_results_until_published_and_renders_the_leaderboard(
        self,
    ):
        from streamlit.testing.v1 import AppTest
        from test_competition import BuyHold
        from test_synthetic_market import training_fixture

        from backtester import EngineConfig
        from competition import Competition
        from synthetic_market import SyntheticConfig

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            training = root / "training"
            training.mkdir()
            for symbol, frame in training_fixture(130).items():
                frame.to_csv(training / f"{symbol}.csv", index_label="Date")
            store = Competition(root / "league.db", root / "rounds")
            round_id = store.create_round(
                "UI rehearsal",
                training,
                EngineConfig(runtime_id="ui-test-runtime"),
                SyntheticConfig(2, 20, 100),
            )
            token = store.create_team("UI team")
            team = store.authenticate(token)
            version = store.submit(
                round_id,
                team["id"],
                "test source",
                {"name": "UI strategy", "author": "UI author", "tickers": ["SPY"]},
            )
            store.select(round_id, team["id"], version)
            with patch.dict(
                os.environ,
                {
                    "DB_PATH": str(store.db_path),
                    "ROUND_ROOT": str(store.round_root),
                    "ADMIN_PASSWORD": "",
                },
            ):
                app = AppTest.from_file(
                    str(Path(__file__).resolve().parents[1] / "app.py"),
                    default_timeout=20,
                ).run()
                self.assertFalse(app.exception)
                self.assertEqual(len(app.dataframe), 0)
                self.assertEqual(len(app.metric), 0)
                self.assertEqual(
                    len(app.get("vega_lite_chart"))
                    + len(app.get("arrow_vega_lite_chart")),
                    0,
                )
                next(
                    field
                    for field in app.text_input
                    if field.label == "Team access token"
                ).set_value(token).run()
                self.assertFalse(
                    next(
                        button
                        for button in app.button
                        if button.label == "Run practice backtest"
                    ).disabled
                )
                store.lock_round(round_id)
                store.score(
                    round_id,
                    runtime_id="ui-test-runtime",
                    trader_factory=lambda source: BuyHold(),
                )
                app.run()
                self.assertFalse(app.exception)
                self.assertTrue(
                    next(
                        button
                        for button in app.button
                        if button.label == "Run practice backtest"
                    ).disabled
                )
                self.assertEqual(len(app.dataframe), 0)
                self.assertEqual(
                    len(app.get("vega_lite_chart"))
                    + len(app.get("arrow_vega_lite_chart")),
                    0,
                )
                store.publish(round_id)
                app.run()
                self.assertFalse(app.exception)
                self.assertEqual(len(app.dataframe), 1)
                self.assertEqual(
                    len(app.get("vega_lite_chart"))
                    + len(app.get("arrow_vega_lite_chart")),
                    1,
                )
                self.assertEqual(app.dataframe[0].value.iloc[0]["Team"], "UI team")

    def test_extracted_sdk_runs_with_interest_rate_data(self):
        import pandas as pd
        from test_synthetic_market import training_fixture

        from app import sdk_zip
        from interest_rate import INTEREST_RATE, MarketData
        from prepare_training import save_training

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with zipfile.ZipFile(io.BytesIO(sdk_zip(["ATLAS"], "ATLAS"))) as archive:
                archive.extractall(root)
            frame = training_fixture(130)["SPY"]
            save_training(
                MarketData(
                    {"ATLAS": frame},
                    interest_rate=pd.Series(
                        [5.25] * 130, index=frame.index, name=INTEREST_RATE
                    ),
                ),
                root / "training",
            )
            result = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "backtester.py",
                    "sample_strategy.py",
                    "--data",
                    "training",
                    "--benchmark",
                    "ATLAS",
                    "--trusted-local",
                ],
                cwd=root,
                capture_output=True,
                text=True,
                check=True,
                timeout=30,
            )
            self.assertIn("benchmark_sharpe", result.stdout)

    def test_downloaded_starter_uses_round_asset_names(self):
        from app import sdk_zip

        for names, benchmark in (
            (["ATLAS", "BOREAL", "CIRRUS", "DUNE", "EMBER"], "ATLAS"),
            (["SPY", "QQQ", "GLD"], "SPY"),
        ):
            with (
                self.subTest(names=names),
                zipfile.ZipFile(io.BytesIO(sdk_zip(names, benchmark))) as archive,
            ):
                source = archive.read("sample_strategy.py").decode()
                trader = next(
                    node
                    for node in ast.parse(source).body
                    if isinstance(node, ast.ClassDef)
                )
                tickers = next(
                    node for node in trader.body if isinstance(node, ast.Assign)
                )
                self.assertEqual(list(ast.literal_eval(tickers.value)), names)
                self.assertIn(
                    f"--benchmark {benchmark}", archive.read("README.md").decode()
                )

    def test_fresh_app_and_organizer_authentication(self):
        from streamlit.testing.v1 import AppTest

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(
                os.environ,
                {
                    "DB_PATH": str(Path(directory) / "league.db"),
                    "ROUND_ROOT": str(Path(directory) / "rounds"),
                    "ADMIN_PASSWORD": "a" * 40,
                },
            ),
        ):
            app = AppTest.from_file(
                str(Path(__file__).resolve().parents[1] / "app.py"),
                default_timeout=20,
            ).run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.metric), 0)
            self.assertEqual(len(app.dataframe), 0)
            self.assertFalse(
                any(
                    button.label == "Freeze dataset and open round"
                    for button in app.button
                )
            )
            password = next(
                field for field in app.text_input if field.label == "Organizer password"
            )
            password.set_value("a" * 40).run()
            self.assertFalse(app.exception)
            self.assertTrue(
                any(
                    button.label == "Freeze dataset and open round"
                    for button in app.button
                )
            )
            next(
                field for field in app.text_input if field.label == "Organizer password"
            ).set_value("wrong").run()
            self.assertFalse(app.exception)
            self.assertFalse(
                any(
                    button.label == "Freeze dataset and open round"
                    for button in app.button
                )
            )


if __name__ == "__main__":
    unittest.main()
