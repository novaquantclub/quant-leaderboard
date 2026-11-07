import ast
import importlib.util
import io
import os
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
