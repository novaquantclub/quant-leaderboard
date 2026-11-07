"""Nova Quant Club: public practice, authenticated final selections, blind rounds."""

import hmac
import json
import os
import sqlite3
from datetime import UTC, datetime

import pandas as pd
import streamlit as st

UI_ERRORS = (ValueError, OSError, RuntimeError, sqlite3.Error)

from backtester import Backtester, EngineConfig
from competition import Competition
from member_package import member_zip, sdk_zip
from strategy_runner import MAX_SOURCE, RemoteTrader, worker_runtime_id
from synthetic_market import SyntheticConfig


def uploaded_source(file):
    if file is None:
        raise ValueError("Attach your strategy .py file.")
    content = file.getvalue()
    if not 0 < len(content) <= MAX_SOURCE:
        raise ValueError("Strategy files must be between 1 byte and 128 KB.")
    return content.decode("utf-8-sig")


def main():
    st.set_page_config(page_title="Nova Quant League", layout="wide")
    st.title("Nova Quant League")
    st.caption(
        "Develop on the public dataset, select a final strategy, then compete on unseen synthetic markets."
    )
    store = Competition(
        os.getenv("DB_PATH", "data/leaderboard.db"),
        os.getenv("ROUND_ROOT", "data/rounds"),
    )
    token = st.sidebar.text_input("Team access token", type="password")
    team = store.authenticate(token)
    if team:
        st.sidebar.success(f"Signed in as {team['name']}")
    elif token:
        st.sidebar.error("Invalid team token.")
    rounds = store.rounds()
    round_id = (
        st.sidebar.selectbox(
            "Competition round",
            [row["id"] for row in rounds],
            format_func=lambda key: next(
                row["name"] for row in rounds if row["id"] == key
            ),
        )
        if rounds
        else None
    )
    current = store.round(round_id) if round_id else None
    leaderboard, practice, submission, guide, organizer = st.tabs(
        ["Leaderboard", "Practice", "Final submission", "Guide", "Organizer"]
    )
    if current:
        st.sidebar.write(f"Round status: **{current['phase']}**")
        st.sidebar.caption("Frozen dataset commitment")
        st.sidebar.code(current["dataset_id"], language=None)
    else:
        st.info("An organizer needs to prepare the first competition round.")

    with leaderboard:
        st.subheader("Official leaderboard")
        if current and current["phase"] == "published":
            rows = store.results(round_id)
            successful = [
                {
                    "Team": row["team"],
                    "Strategy": row["strategy"],
                    **json.loads(row["result"]),
                }
                for row in rows
                if row["result"]
            ]
            successful.sort(key=lambda row: row["score"], reverse=True)
            if successful:
                table = pd.DataFrame(
                    [
                        {
                            "Team": row["Team"],
                            "Strategy": row["Strategy"],
                            "Median Sharpe": row["score"],
                            "Sharpe P10": row["sharpe_p10"],
                            "Sharpe P90": row["sharpe_p90"],
                            "Median CI low": row["score_ci_low"],
                            "Median CI high": row["score_ci_high"],
                            "Return (%)": row["total_return_pct"],
                            "Max DD (%)": row["max_drawdown_pct"],
                        }
                        for row in successful
                    ],
                    index=range(1, len(successful) + 1),
                )
                table.index.name = "Rank"
                st.dataframe(table, width="stretch")
                st.caption(
                    "Ranking uses median Sharpe across independent generated paths. P10/P90 show path dispersion; "
                    "the bootstrap interval describes uncertainty under this generator, not future real-market performance."
                )
                chosen = st.multiselect(
                    "Compare teams",
                    [row["Team"] for row in successful],
                    default=[row["Team"] for row in successful[:3]],
                )
                curves = {
                    row["Team"]: row["equity_curve"]["Strategy"]
                    for row in successful
                    if row["Team"] in chosen
                }
                if curves:
                    benchmark = json.loads(current["config"])["benchmark"]
                    column = "SPY_Benchmark" if benchmark == "SPY" else "Benchmark"
                    curves[f"{benchmark} benchmark"] = successful[0]["equity_curve"][
                        column
                    ]
                    st.line_chart(pd.DataFrame(curves), height=350)
                    st.caption(
                        "Lines show the pointwise median equity across paths; no single path necessarily follows that curve."
                    )
            for row in rows:
                if row["error"]:
                    st.warning(
                        f"{row['team']}: final strategy could not be evaluated. {row['error']}"
                    )
        elif current:
            st.info(
                "This is a blind round. Final scores, benchmark metrics, and curves appear only after submissions lock and results are published."
            )

    with practice:
        st.subheader("Practice on the public dataset")
        st.write(
            "Practice results use only the data available to every team. They never query the final test."
        )
        if current:
            st.download_button(
                "Download training CSVs",
                store.training_zip(round_id),
                "training.zip",
                "application/zip",
            )
            strategy_file = st.file_uploader(
                "Practice strategy (.py)", type=["py"], key="practice_source"
            )
            if not team:
                st.info(
                    "Enter your team token to run an online backtest. You can also test your own code locally using the SDK."
                )
            if current["phase"] != "open":
                st.info(
                    "Online practice runs during the open submission phase. Use the downloaded package for local backtesting."
                )
            if (
                st.button(
                    "Run practice backtest",
                    disabled=not team or current["phase"] != "open",
                )
                and team
                and current["phase"] == "open"
            ):
                try:
                    with st.spinner("Running your strategy on the public dataset..."):
                        config = EngineConfig(**json.loads(current["config"]))
                        result = Backtester(config).run_public(
                            uploaded_source(strategy_file),
                            store.public_market(round_id),
                        )
                    cols = st.columns(4)
                    for column, key, label in zip(
                        cols,
                        (
                            "sharpe",
                            "total_return_pct",
                            "max_drawdown_pct",
                            "final_equity",
                        ),
                        ("Sharpe", "Return (%)", "Max drawdown (%)", "Final equity"),
                    ):
                        column.metric(label, f"{result[key]:,.3f}")
                    st.line_chart(result["equity_df"], height=350)
                except UI_ERRORS as exc:
                    st.error(f"Practice failed: {exc}")

    with submission:
        st.subheader("Choose your team's final strategy")
        if not team:
            st.info("Enter the team access token provided by an organizer.")
        elif current:
            versions = store.versions(round_id, team["id"])
            st.caption(
                f"{len(versions)}/{current['max_submissions']} versions stored. "
                f"Team upload cooldown: {current['cooldown_seconds'] // 60} minutes. Practice runs are separate."
            )
            if current["phase"] == "open":
                with st.form("final_upload"):
                    file = st.file_uploader("Strategy version (.py)", type=["py"])
                    save = st.form_submit_button("Store version")
                if save:
                    try:
                        source = uploaded_source(file)
                        with st.spinner(
                            "Checking strategy metadata in the isolated runner..."
                        ):
                            with RemoteTrader(source) as trader:
                                metadata = trader.metadata
                            store.submit(round_id, team["id"], source, metadata)
                        st.success(
                            "Version stored. Select it below as your final entry when ready."
                        )
                        versions = store.versions(round_id, team["id"])
                    except UI_ERRORS as exc:
                        st.error(f"Upload failed: {exc}")
                if versions:
                    selected = store.selection(round_id, team["id"])
                    identifiers = [version["id"] for version in versions]
                    pick = st.selectbox(
                        "Final version",
                        identifiers,
                        index=identifiers.index(selected)
                        if selected in identifiers
                        else 0,
                        format_func=lambda key: next(
                            f"{v['name']} · {v['source_hash'][:12]} · {datetime.fromtimestamp(v['submitted_at'], UTC):%Y-%m-%d %H:%M UTC}"
                            for v in versions
                            if v["id"] == key
                        ),
                    )
                    if st.button("Select this version as final"):
                        try:
                            store.select(round_id, team["id"], pick)
                            st.success(
                                "Your final selection is saved. You may change it until submissions lock."
                            )
                        except UI_ERRORS as exc:
                            st.error(str(exc))
            else:
                st.info("Submissions and final selections are locked for this round.")
            selected = store.selection(round_id, team["id"])
            if selected:
                st.write(f"Selected version: `{selected}`")
            else:
                st.warning("Your team has not selected a final strategy.")

    with guide:
        st.subheader("Strategy SDK")
        tickers = sorted(store.public_market(round_id)) if current else None
        benchmark = json.loads(current["config"])["benchmark"] if current else "ATLAS"
        config = EngineConfig(**json.loads(current["config"])) if current else None
        if current:
            st.download_button(
                "Download complete member starter package",
                member_zip(current["public_dir"], config),
                "nova-quant-member-starter.zip",
                "application/zip",
            )
        st.download_button(
            "Download SDK and starter strategy",
            sdk_zip(tickers, benchmark, config),
            "nova-quant-sdk.zip",
            "application/zip",
        )
        st.markdown("""
1. Download the public training CSVs and the SDK. Use the same fixed asset universe for practice and finals.
2. Define `Trader.name`, `Trader.author`, `Trader.tickers`, and `Trader.run(state)`.
3. Return `{ticker: quantity}` or `(orders, trader_data)`. Positive quantities buy and negative quantities sell.
4. Orders fill at the next open, with 10 bps costs. Positions are long-only; `^` and `=` symbols are signals only.
5. History includes 100 warm-up bars and the current bar, with integer indices. Both portfolios start in cash at the first close.
6. Online code runs without network, host files, or persistent disk access. Keep `trader_data` under 100 KB.
7. Practice freely on public data, store your versions, and explicitly choose one final entry before the organizer locks the round.
8. Finals use independent synthetic paths fitted to the public data. Scores remain hidden until the organizer publishes the closed round.
9. Research the assets and discover their relationships using the published observations.
10. When supplied, `state.observations["interest rate"]` contains the available rate history in annualized percent, including warm-up. For example, 5.25 means 5.25% per year.

Your team identity comes from your access token. `Trader.author` is descriptive and cannot select another team's leaderboard entry.
""")
        st.code(
            "python backtester.py sample_strategy.py --data training "
            f"--benchmark {benchmark} --trusted-local",
            language="shell",
        )
        st.caption(
            "Use --trusted-local only for your own code on your own computer. Hosted uploads always use the isolated worker."
        )

    with organizer:
        admin_password = os.getenv("ADMIN_PASSWORD", "")
        supplied = st.text_input("Organizer password", type="password")
        if len(admin_password) < 32:
            st.info(
                "Organizer controls require ADMIN_PASSWORD to be configured with at least 32 characters."
            )
            return
        if not hmac.compare_digest(supplied.encode(), admin_password.encode()):
            st.info(
                "Enter the organizer password to manage teams and competition rounds."
            )
            return
        st.subheader("Team access")
        with st.form("create_team"):
            team_name = st.text_input("New team name")
            if st.form_submit_button("Create team token"):
                try:
                    access = store.create_team(team_name)
                    st.success(
                        "Copy this token and give it to the team. Only its hash is stored."
                    )
                    st.code(access, language=None)
                except UI_ERRORS as exc:
                    st.error(str(exc))
        teams = store.teams()
        if teams:
            rotate = st.selectbox(
                "Team whose token needs replacing",
                [item["id"] for item in teams],
                format_func=lambda key: next(
                    item["name"] for item in teams if item["id"] == key
                ),
            )
            if st.button("Replace team token"):
                st.code(store.rotate_token(rotate), language=None)
                st.caption(
                    "The previous token no longer works; submissions stay attached to the same team."
                )
        st.subheader("Prepare a frozen round")
        with st.form("create_round"):
            name = st.text_input("Round name", "Round 1")
            training_dir = st.text_input(
                "Public training CSV folder on this server", "data/training"
            )
            benchmark = st.text_input("Benchmark competition asset", "ATLAS")
            paths = st.number_input(
                "Independent final paths", min_value=10, max_value=100, value=30
            )
            bars = st.number_input(
                "Scored bars per final path", min_value=20, max_value=2520, value=252
            )
            limit = st.number_input(
                "Maximum stored versions per team", min_value=1, max_value=100, value=10
            )
            cooldown = st.number_input(
                "Minutes between team uploads", min_value=0, max_value=1440, value=30
            )
            create = st.form_submit_button("Freeze dataset and open round")
        if create:
            try:
                with st.spinner(
                    "Freezing the public training snapshot and generating final paths..."
                ):
                    config = EngineConfig(
                        benchmark=benchmark.strip().upper(),
                        runtime_id=worker_runtime_id(),
                    )
                    store.create_round(
                        name,
                        training_dir,
                        config,
                        SyntheticConfig(int(paths), int(bars), config.warmup_bars),
                        max_submissions=int(limit),
                        cooldown_seconds=int(cooldown) * 60,
                    )
                st.rerun()
            except UI_ERRORS as exc:
                st.error(f"Round preparation failed: {exc}")
        if current:
            st.subheader("Close, evaluate, and publish")
            st.write(f"Current round: **{current['name']}** · {current['phase']}")
            if current["phase"] == "open" and st.button(
                "Lock submissions and final selections"
            ):
                try:
                    store.lock_round(round_id)
                    st.rerun()
                except UI_ERRORS as exc:
                    st.error(str(exc))
            if current["phase"] in ("locked", "scored", "scoring") and st.button(
                "Evaluate selected strategies / retry failures"
            ):
                try:
                    status = st.empty()
                    with st.spinner(
                        "Evaluating the frozen selections. Results remain private until published..."
                    ):
                        store.score(
                            round_id,
                            progress=lambda index, count, team_name: status.write(
                                f"Evaluating {team_name} ({index + 1}/{count})"
                            ),
                        )
                    st.rerun()
                except UI_ERRORS as exc:
                    st.error(str(exc))
            if current["phase"] == "scored" and st.button("Publish final leaderboard"):
                try:
                    store.publish(round_id)
                    st.rerun()
                except UI_ERRORS as exc:
                    st.error(str(exc))
            rows = store.results(round_id, organizer=True)
            for row in rows:
                st.write(
                    f"{row['team']} — {row['strategy']}: {'scored' if row['result'] else row['error'] or 'awaiting evaluation'}"
                )


if __name__ == "__main__":
    main()
