import json
import os
import sqlite3
import tempfile
from datetime import datetime
import streamlit as st
import pandas as pd
from backtester import Backtester, load_trader_from_file

# --- Configuration & Environment Variables ---
CLUB_PASSWORD = os.getenv("CLUB_PASSWORD", "password")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "password")
# Secret comma-separated regimes (Bull, Bear, Volatile) — NEVER shown in the UI
SECRET_OOS_WINDOWS = os.getenv(
    "SECRET_OOS_WINDOWS",
    "2018-01-01:2018-12-31,2019-01-01:2019-12-31",
)
COOLDOWN_HOURS = int(os.getenv("COOLDOWN_HOURS", "0"))
SUBMISSIONS_DIR = "data/submissions"
DB_PATH = os.getenv("DB_PATH", "data/leaderboard.db")

os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
os.makedirs(SUBMISSIONS_DIR, exist_ok=True)


@st.cache_data(ttl=3600)
def get_oos_spy_benchmark(windows_str: str):
    """Fetches and caches SPY buy-and-hold metrics on the concatenated regimes."""
    bt = Backtester()
    return bt.run_benchmark_multi_window(windows_str)


def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS app_settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        table_sql = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'leaderboard'"
        ).fetchone()
        if table_sql and "member_name TEXT UNIQUE" in table_sql[0]:
            conn.execute("""
                CREATE TABLE leaderboard_new (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    member_name TEXT NOT NULL,
                    strategy_name TEXT NOT NULL,
                    sharpe REAL,
                    excess_sharpe REAL,
                    total_return_pct REAL,
                    alpha_pct REAL,
                    beta REAL,
                    max_drawdown_pct REAL,
                    submitted_at TEXT
                )
                """)
            conn.execute("INSERT INTO leaderboard_new SELECT * FROM leaderboard")
            conn.execute("DROP TABLE leaderboard")
            conn.execute("ALTER TABLE leaderboard_new RENAME TO leaderboard")
        else:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS leaderboard (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    member_name TEXT NOT NULL,
                    strategy_name TEXT NOT NULL,
                    sharpe REAL,
                    excess_sharpe REAL,
                    total_return_pct REAL,
                    alpha_pct REAL,
                    beta REAL,
                    max_drawdown_pct REAL,
                    submitted_at TEXT
                )
                """)
        conn.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS
            idx_leaderboard_member_strategy
            ON leaderboard(member_name, strategy_name)
            """)

        # Store the precomputed sequential equity curve so the leaderboard can
        # display it immediately without re-running any strategy.
        columns = [row[1] for row in conn.execute("PRAGMA table_info(leaderboard)")]
        if "equity_curve" not in columns:
            conn.execute("ALTER TABLE leaderboard ADD COLUMN equity_curve TEXT")


def get_evaluation_windows() -> str:
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT value FROM app_settings WHERE key = 'evaluation_windows'"
        ).fetchone()
    return row[0] if row else SECRET_OOS_WINDOWS


def save_evaluation_windows(windows: str) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO app_settings (key, value)
            VALUES ('evaluation_windows', ?)
            ON CONFLICT(key) DO UPDATE SET value=excluded.value
            """,
            (windows,),
        )


def validate_evaluation_windows(windows: str) -> str | None:
    entries = [entry.strip() for entry in windows.split(",") if entry.strip()]
    if not entries:
        return "Please specify at least one evaluation window."

    for entry in entries:
        parts = [part.strip() for part in entry.split(":")]
        if len(parts) != 2:
            return f"Invalid window `{entry}`. Use YYYY-MM-DD:YYYY-MM-DD."
        try:
            start = datetime.strptime(parts[0], "%Y-%m-%d")
            end = datetime.strptime(parts[1], "%Y-%m-%d")
        except ValueError:
            return f"Invalid window `{entry}`. Use YYYY-MM-DD:YYYY-MM-DD."
        if start >= end:
            return f"Invalid window `{entry}`. The end date must be after the start date."
    return None


def serialize_equity_curve(curve_df: pd.DataFrame) -> str:
    return json.dumps(
        {
            "strategy": [round(float(v), 4) for v in curve_df["Strategy"]],
            "spy": [round(float(v), 4) for v in curve_df["SPY_Benchmark"]],
        },
        separators=(",", ":"),
    )


def upsert_submission(member_name, strategy_name, res):
    excess_sharpe = res["sharpe"] - res["spy_sharpe"]
    equity_curve = serialize_equity_curve(res["equity_curve_df"])
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO leaderboard (
                member_name, strategy_name, sharpe, excess_sharpe,
                total_return_pct, alpha_pct, beta, max_drawdown_pct,
                submitted_at, equity_curve
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(member_name, strategy_name) DO UPDATE SET
                strategy_name=excluded.strategy_name,
                sharpe=excluded.sharpe,
                excess_sharpe=excluded.excess_sharpe,
                total_return_pct=excluded.total_return_pct,
                alpha_pct=excluded.alpha_pct,
                beta=excluded.beta,
                max_drawdown_pct=excluded.max_drawdown_pct,
                submitted_at=excluded.submitted_at,
                equity_curve=excluded.equity_curve
            """,
            (
                member_name,
                strategy_name,
                round(res["sharpe"], 3),
                round(excess_sharpe, 3),
                round(res["total_return_pct"], 2),
                round(res["alpha_pct"], 2),
                round(res["beta"], 2),
                round(res["max_drawdown_pct"], 2),
                datetime.now().strftime("%Y-%m-%d %H:%M UTC"),
                equity_curve,
            ),
        )


def get_strategy_metadata(strategy_filepath):
    trader = load_trader_from_file(strategy_filepath)
    member_name = getattr(trader, "author", None)
    strategy_name = getattr(trader, "name", None)
    if (
        not isinstance(member_name, str)
        or not isinstance(strategy_name, str)
        or not member_name.strip()
        or not strategy_name.strip()
    ):
        raise AttributeError(
            "Strategy Trader must define non-empty string `author` and `name` attributes."
        )
    return member_name.strip(), strategy_name.strip()


def get_leaderboard() -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(
            """
            SELECT 
                member_name AS "Member",
                strategy_name AS "Strategy",
                sharpe AS "Sharpe",
                total_return_pct AS "Return (%)",
                alpha_pct AS "Alpha (%)",
                beta AS "Beta",
                max_drawdown_pct AS "Max DD (%)",
                submitted_at AS "Submitted"
            FROM leaderboard
            ORDER BY sharpe DESC
            """,
            conn,
        )
    df.index = df.index + 1
    df.index.name = "Rank"
    return df


def get_equity_curves() -> dict:
    """Load precomputed sequential equity curves keyed by (member, strategy)."""
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            """
            SELECT member_name, strategy_name, equity_curve
            FROM leaderboard
            WHERE equity_curve IS NOT NULL
            """
        ).fetchall()

    curves = {}
    for member, strategy, payload in rows:
        try:
            data = json.loads(payload)
            curves[(member, strategy)] = pd.DataFrame(
                {"Strategy": data["strategy"], "SPY_Benchmark": data["spy"]}
            )
        except (TypeError, ValueError, KeyError):
            continue
    return curves


init_db()
active_evaluation_windows = get_evaluation_windows()
st.set_page_config(page_title="Quant Club Prosperity Leaderboard", layout="wide")
st.title("Nova Quant Club — Strategy Leaderboard")

leaderboard_tab, backtester_tab, submission_tab, guide_tab, admin_tab = st.tabs(
    ["Leaderboard", "Backtester", "Submit to Leaderboard", "Guide", "Admin"]
)

# --- TAB 1: LEADERBOARD ---
with leaderboard_tab:
    st.subheader("Official Rankings")
    st.caption(
        "Evaluated across undisclosed multi-regime market windows | Ranked by Annualized Sharpe Ratio on the concatenated equity curve (10 bps fee/slippage, 4% RF rate)"
    )

    # Pinned SPY Benchmark Reference Bar
    try:
        spy_bench = get_oos_spy_benchmark(active_evaluation_windows)
        st.markdown(f"##### Benchmark Reference: `SPY` Buy-and-Hold")
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("SPY Sharpe", f"{spy_bench['sharpe']:.3f}")
        b2.metric("SPY Total Return", f"{spy_bench['total_return_pct']:.2f}%")
        b3.metric("SPY Max Drawdown", f"{spy_bench['max_drawdown_pct']:.2f}%")
        b4.metric("SPY Final Equity ($100k)", f"${spy_bench['final_equity']:,.2f}")
    except Exception as e:
        st.warning(f"Could not load SPY benchmark banner: {e}")

    st.divider()
    lb_df = get_leaderboard()
    if lb_df.empty:
        st.info("No strategies submitted yet. Head over to the Submit tab to beat SPY!")
    else:
        st.dataframe(lb_df, width="stretch")

        st.markdown("##### Equity Curve Comparison")
        st.caption(
            "Equity paths are precomputed when a strategy is scored and loaded from the "
            "database. The hidden regimes are stitched together in order as one continuous "
            "portfolio, starting at $100,000."
        )

        options = {
                    f"{row['Member']} — {row['Strategy']}": (row["Member"], row["Strategy"])
                    for _, row in lb_df.iterrows()
        }
        chosen_labels = st.multiselect(
            "Strategies to compare",
            list(options),
            default=list(options)[: min(3, len(options))],
            key="lb_compare_select",
        )

        stored_curves = get_equity_curves()
        chart_df = pd.DataFrame()
        spy_series = None
        missing_curves = []


        for label in chosen_labels:
            member, strategy = options[label]
            curve = stored_curves.get((member, strategy))
            if curve is None:
                missing_curves.append(label)
                continue
            chart_df[label] = curve["Strategy"]
            if spy_series is None:
                spy_series = curve["SPY_Benchmark"]

        if spy_series is not None:
            chart_df["SPY Buy-and-Hold"] = spy_series

        if chart_df.empty:
            if stored_curves:
                st.info("Select at least one strategy to compare.")
            else:
                st.info(
                    "No stored equity curves yet. An organizer can run "
                    "**Re-Score Entire Leaderboard** from the Admin tab to populate them."
                )
        else:
            st.markdown("#### Portfolio Equity")
            st.line_chart(chart_df, height=420, width="stretch")

        if missing_curves:
            st.warning(
                "No stored equity curve for: "
                + ", ".join(missing_curves)
                + ". Re-score from the Admin tab to populate it."
            )


# --- TAB 2: WEB SANDBOX BACKTESTER ---
with backtester_tab:
    st.subheader("Interactive Backtester")
    st.write(
        "Test your `.py` strategy file on historical In-Sample dates and compare your equity curve directly against `SPY`."
    )

    col1, col2 = st.columns(2)
    with col1:
        is_start = st.date_input(
            "Start Date",
            value=pd.to_datetime("2024-01-01"),
        )
    with col2:
        is_end = st.date_input(
            "End Date",
            value=pd.to_datetime("2025-12-31"),
        )

    sandbox_file = st.file_uploader(
        "Upload Strategy (.py)", type=["py"], key="sandbox_up"
    )
    if st.button("Run Sandbox Backtest") and sandbox_file:
        with st.spinner(
            "Fetching yfinance data and simulating bar-by-bar execution vs SPY..."
        ):
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".py") as tmp:
                    tmp.write(sandbox_file.getvalue())
                    tmp_path = tmp.name

                trader = load_trader_from_file(tmp_path)
                bt = Backtester()
                res = bt.run(trader, str(is_start), str(is_end))
                os.remove(tmp_path)

                sharpe_diff = res["sharpe"] - res["spy_sharpe"]
                ret_diff = res["total_return_pct"] - res["spy_return_pct"]
                dd_diff = res["max_drawdown_pct"] - res["spy_max_dd_pct"]

                m1, m2, m3, m4, m5, m6 = st.columns(6)
                m1.metric(
                    "Strategy Sharpe",
                    f"{res['sharpe']:.3f}",
                    f"{sharpe_diff:+.3f} vs SPY",
                )
                m2.metric(
                    "Total Return",
                    f"{res['total_return_pct']:.2f}%",
                    f"{ret_diff:+.2f}% vs SPY",
                )
                m3.metric(
                    "Max Drawdown",
                    f"{res['max_drawdown_pct']:.2f}%",
                    f"{dd_diff:+.2f}% vs SPY",
                )
                m4.metric("Annual Alpha (α)", f"{res['alpha_pct']:+.2f}%")
                m5.metric("Beta to SPY (β)", f"{res['beta']:.2f}")
                m6.metric("Final Equity", f"${res['final_equity']:,.0f}")

                st.markdown("#### Equity Curve: Strategy vs. SPY Buy-and-Hold")
                st.line_chart(
                    res["equity_df"][["Strategy", "SPY_Benchmark"]],
                    width="stretch",
                )
            except Exception as e:
                st.error(f"Backtest Error: {e}")

# --- TAB 3: OFFICIAL SUBMISSION ---
with submission_tab:
    st.subheader("Submit Strategy to the Leaderboard")
    st.caption(
        f"Strategies are evaluated across multiple undisclosed historical market regimes "
        f"with anonymized timestamps and base-100 prices. Submission cooldown: **{COOLDOWN_HOURS:g} hours**."
    )
    submission_success = st.session_state.pop("submission_success", None)
    if submission_success:
        st.success(submission_success)

    with st.form("submission_form"):
        st.info(
            "The leaderboard uses the uploaded strategy's `Trader.author` and `Trader.name` values."
        )
        password_input = st.text_input("Club Upload Password", type="password")
        sub_file = st.file_uploader("Upload Strategy (.py)", type=["py"])
        submitted = st.form_submit_button("Evaluate & Post to Leaderboard")

        if submitted:
            if password_input != CLUB_PASSWORD:
                st.error("Invalid club password.")
            elif not sub_file:
                st.warning("Please attach your strategy file.")
            else:
                tmp_path = None
                saved_path = None
                row = None
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".py") as tmp:
                        tmp.write(sub_file.getvalue())
                        tmp_path = tmp.name

                    clean_member, clean_strat = get_strategy_metadata(tmp_path)
                    safe_name = "".join(
                        c
                        for c in f"{clean_member}_{clean_strat}"
                        if c.isalnum() or c in ("_", "-")
                    )
                    if not safe_name:
                        raise ValueError(
                            "Trader.name must contain at least one letter or number."
                        )

                    with sqlite3.connect(DB_PATH) as conn:
                        row = conn.execute(
                            """
                            SELECT submitted_at
                            FROM leaderboard
                            WHERE member_name = ? AND strategy_name = ?
                            """,
                            (clean_member, clean_strat),
                        ).fetchone()

                    cooldown_blocked = False
                    if row and row[0]:
                        try:
                            last_sub = datetime.strptime(row[0], "%Y-%m-%d %H:%M UTC")
                            hours_since = (
                                datetime.now() - last_sub
                            ).total_seconds() / 3600.0
                            if hours_since < COOLDOWN_HOURS:
                                remaining = COOLDOWN_HOURS - hours_since
                                st.error(
                                    f"⏳ Cooldown active for **{clean_member}**. "
                                    f"You can submit again in **{remaining:.1f} hours**."
                                )
                                cooldown_blocked = True
                        except ValueError:
                            pass

                    if not cooldown_blocked:
                        with st.spinner(
                            "Evaluating across secret multi-regime OOS windows vs SPY..."
                        ):
                            saved_path = os.path.join(
                                SUBMISSIONS_DIR, f"{safe_name}.py"
                            )
                            with open(saved_path, "wb") as f:
                                f.write(sub_file.getvalue())

                            bt = Backtester(anonymize=True)
                            res = bt.run_multi_window(
                                saved_path, active_evaluation_windows
                            )

                            upsert_submission(clean_member, clean_strat, res)
                            diff = res["sharpe"] - res["spy_sharpe"]
                            st.session_state["submission_success"] = (
                                f"✅ Submitted across **{res['regimes_tested']}** hidden regimes! "
                                f"Sharpe: **{res['sharpe']:.3f}** ({diff:+.3f} vs SPY) | "
                                f"Total Return: **{res['total_return_pct']:+.2f}%** | "
                                f"Max Drawdown: **{res['max_drawdown_pct']:.2f}%** | "
                                f"Alpha: **{res['alpha_pct']:+.2f}%** | "
                                f"Beta: **{res['beta']:.2f}**. "
                                "Check the Leaderboard tab to see your submission."
                            )
                            st.rerun()
                except Exception as e:
                    if saved_path and os.path.exists(saved_path):
                        os.remove(saved_path)
                    st.error(f"Evaluation failed: {e}")
                finally:
                    if tmp_path and os.path.exists(tmp_path):
                        os.remove(tmp_path)

# --- TAB 4: LOCAL SDK & RULES ---
with guide_tab:
    st.subheader("Download Local Backtester & Strategy Template")
    st.write(
        "Develop locally using the exact same execution engine and SPY benchmark comparison running on this server."
    )

    col_d1, col_d2 = st.columns(2)
    with open("backtester.py", "r") as f:
        col_d1.download_button(
            "⬇️ Download backtester.py",
            f.read(),
            file_name="backtester.py",
            mime="text/x-python",
        )
    with open("sample_strategy.py", "r") as f:
        sample_code = f.read()
        col_d2.download_button(
            "⬇️ Download sample_strategy.py",
            sample_code,
            file_name="sample_strategy.py",
            mime="text/x-python",
        )

    st.markdown("""
    ### Available Python libraries
    Uploaded strategies can use the packages installed from `requirements.txt`:
    - `numpy` and `pandas` for numerical work, time series, and data manipulation.
    - `scipy` for statistics, optimization, signal processing, and distributions.
    - `statsmodels` for regressions, statistical tests, and time-series models.
    - `scikit-learn` for preprocessing, clustering, and lightweight predictive models.
    - `numba` for accelerating numerical loops with `@numba.njit`.
    - `yfinance` for market data used by the backtester.

    ### Rules for Fair Comparison
    1. **Zero Lookahead Bias:** Your `Trader.run(state)` method is called sequentially day-by-day. `state.history[ticker]` contains OHLCV bars strictly up to current bar $t$.
    2. **Execution at Next Open:** Orders returned at bar $t$ are filled at the **Open price of bar $t+1$** with **10 bps (0.10%)** slippage/commission deducted.
    3. **SPY Benchmark:** `SPY` Buy-and-Hold is initialized with $100,000 at the first day's Open (less the 10 bps fee) and held through the end of each evaluation window.
    4. **Ranking Metric:** Annualized Daily Sharpe Ratio ($R_f = 4\\%$, $\\sqrt{252}$ scaling) computed on the **concatenated** Out-Of-Sample equity curve, alongside CAPM Alpha ($\\alpha$) and Beta ($\\beta$) relative to `SPY`. Regimes are chained in order, with each regime's ending equity carried into the next.
    """)
    st.code(sample_code, language="python")

with admin_tab:
    st.subheader("Organizer Administration")
    admin_pw = st.text_input("Admin Password", type="password", key="admin_password")

    if admin_pw != ADMIN_PASSWORD:
        st.info("Enter the admin password to manage submitted strategies.")
    else:
        st.success("Admin access granted.")
        new_windows = st.text_input(
            "Evaluation Window(s)",
            value=active_evaluation_windows,
            placeholder="YYYY-MM-DD:YYYY-MM-DD,YYYY-MM-DD:YYYY-MM-DD",
            help="Use comma-separated start:end windows.",
            key="admin_windows",
        )

        if st.button("Submit Changes", key="admin_save_windows"):
            validation_error = validate_evaluation_windows(new_windows)
            if validation_error:
                st.error(validation_error)
            else:
                save_evaluation_windows(new_windows.strip())
                active_evaluation_windows = new_windows.strip()
                st.success("Evaluation windows updated.")

        st.markdown("#### Re-evaluate strategies")
        if st.button("Re-Score Entire Leaderboard", key="admin_rescore"):
            py_files = sorted(
                fname
                for fname in os.listdir(SUBMISSIONS_DIR)
                if fname.endswith(".py")
            )
            if not py_files:
                st.info("No uploaded strategies found.")
            else:
                bt = Backtester(anonymize=True)
                progress_bar = st.progress(0)
                status_text = st.empty()
                success_count = 0
                for idx, fname in enumerate(py_files):
                    fpath = os.path.join(SUBMISSIONS_DIR, fname)
                    status_text.text(
                        f"Re-evaluating {fname} ({idx + 1}/{len(py_files)})..."
                    )
                    try:
                        member, strategy = get_strategy_metadata(fpath)
                        result = bt.run_multi_window(fpath, active_evaluation_windows)
                        upsert_submission(member, strategy, result)
                        success_count += 1
                    except Exception as e:
                        st.warning(f"Failed to re-evaluate {fname}: {e}")
                    progress_bar.progress((idx + 1) / len(py_files))
                status_text.empty()
                st.success(
                    f"Successfully re-scored {success_count}/{len(py_files)} strategies."
                )
                st.rerun()

        st.markdown("#### Download uploaded strategies")
        py_files = sorted(
            fname for fname in os.listdir(SUBMISSIONS_DIR) if fname.endswith(".py")
        )
        if not py_files:
            st.info("No uploaded strategies found.")
        else:
            for fname in py_files:
                fpath = os.path.join(SUBMISSIONS_DIR, fname)
                with open(fpath, "rb") as strategy_file:
                    st.download_button(
                        f"Download {fname}",
                        strategy_file.read(),
                        file_name=fname,
                        mime="text/x-python",
                        key=f"download_strategy_{fname}",
                    )

        st.markdown("#### Delete strategies")
        if py_files:
            delete_file = st.selectbox(
                "Strategy to delete",
                py_files,
                key="admin_delete_strategy",
            )
            if st.button("Delete Selected Strategy", key="admin_delete"):
                delete_path = os.path.join(SUBMISSIONS_DIR, delete_file)
                try:
                    member, strategy = get_strategy_metadata(delete_path)
                    with sqlite3.connect(DB_PATH) as conn:
                        conn.execute(
                            """
                            DELETE FROM leaderboard
                            WHERE member_name = ? AND strategy_name = ?
                            """,
                            (member, strategy),
                        )
                    os.remove(delete_path)
                    st.success(f"Deleted {delete_file} and its leaderboard entry.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Could not delete {delete_file}: {e}")
