"""Build the public member SDK and a complete standalone starter workspace."""

import argparse
import io
import json
import re
import zipfile
from dataclasses import asdict
from pathlib import Path

from backtester import EngineConfig
from interest_rate import INTEREST_RATE_FILE
from synthetic_market import load_training


def sdk_zip(tickers=None, benchmark="ATLAS", config=None):
    config = config or EngineConfig(benchmark=benchmark)
    tickers = (
        tuple(tickers)
        if tickers is not None
        else ("ATLAS", "BOREAL", "CIRRUS", "DUNE", "EMBER")
    )
    root = Path(__file__).parent
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in (
            "backtester.py",
            "synthetic_market.py",
            "interest_rate.py",
            "strategy_runner.py",
            "sample_strategy.py",
        ):
            content = (root / name).read_text(encoding="utf-8")
            if name == "sample_strategy.py":
                content = re.sub(
                    r"(?m)^    tickers = .+$",
                    f"    tickers = {tickers!r}",
                    content,
                    count=1,
                )
                content = content.replace(
                    "--benchmark ATLAS",
                    f"--config engine_config.json --benchmark {config.benchmark}",
                )
            archive.writestr(name, content)
        requirements = (
            "\n".join(
                line
                for line in (root / "requirements-worker.txt").read_text().splitlines()
                if line.strip() and not line.startswith("#")
            )
            + "\n"
        )
        archive.writestr("requirements.txt", requirements)
        archive.writestr(
            "engine_config.json",
            json.dumps(
                {
                    key: value
                    for key, value in asdict(config).items()
                    if key != "runtime_id"
                },
                indent=2,
            )
            + "\n",
        )
        readme = (root / "MEMBER_README.md").read_text(encoding="utf-8")
        for marker, value in {
            "ASSETS": ", ".join(tickers),
            "BENCHMARK": config.benchmark,
            "WARMUP_BARS": str(config.warmup_bars),
            "INITIAL_CASH": f"{config.initial_cash:,.2f}",
            "COMMISSION_BPS": f"{config.commission_bps:g}",
            "RISK_FREE_RATE_PCT": f"{config.risk_free_rate * 100:g}",
        }.items():
            readme = readme.replace("{{" + marker + "}}", value)
        archive.writestr("README.md", readme)
    return output.getvalue()


def member_zip(training_directory, config=None):
    market = load_training(training_directory)
    config = config or EngineConfig(benchmark="ATLAS")
    if config.benchmark not in market:
        raise ValueError("The benchmark must belong to the training universe.")
    output = io.BytesIO()
    with (
        zipfile.ZipFile(io.BytesIO(sdk_zip(sorted(market), config=config))) as sdk,
        zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive,
    ):
        for name in sdk.namelist():
            archive.writestr(name, sdk.read(name))
        for symbol, frame in market.items():
            archive.writestr(
                f"training/{symbol}.csv",
                frame.to_csv(index_label=frame.index.name or "Date"),
            )
        if market.interest_rate is not None:
            archive.writestr(
                f"training/{INTEREST_RATE_FILE}",
                market.interest_rate.to_csv(
                    index_label=market.interest_rate.index.name or "Date"
                ),
            )
        archive.writestr(
            "nova-quant-member-starter.code-workspace",
            json.dumps({"folders": [{"path": "."}]}, indent=2) + "\n",
        )
    return output.getvalue()


def main():
    parser = argparse.ArgumentParser(
        description="Package public datasets and the member SDK into one ZIP."
    )
    parser.add_argument("training_directory", type=Path)
    parser.add_argument("output_zip", type=Path)
    parser.add_argument("--benchmark", default="ATLAS")
    args = parser.parse_args()
    content = member_zip(
        args.training_directory, EngineConfig(benchmark=args.benchmark)
    )
    args.output_zip.parent.mkdir(parents=True, exist_ok=True)
    with args.output_zip.open("xb") as output:
        output.write(content)
    print(f"Saved member starter package to {args.output_zip}.")


if __name__ == "__main__":
    main()
