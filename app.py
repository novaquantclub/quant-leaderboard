import os
import sqlite3
import tempfile
from datetime import datetime
import streamlit as st
import pandas as pd
from backtester import Backtester, load_trader_from_file

# --- Configuration & Environment Variables ---
CLUB_PASSWORD = os.getenv("CLUB_PASSWORD", "password")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin")
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
    """Fetches and caches average SPY Buy-and-Hold metrics across all secret OOS regimes."""
    bt = Backtester()
    pairs = [w.strip().split(":") for w in windows_str.split(",") if ":" in w]
    metrics_list = []

    for start_d, end_d in pairs:
        spy_df = bt._download_single("SPY", start_d.strip(), end_d.strip())
        spy_open = float(spy_df["Open"].iloc[0])
        shares = bt.initial_cash / (spy_open * (1.0 + bt.commission_rate))
        eq_series = shares * spy_df["Close"]
        metrics_list.append(bt._calc_metrics(eq_series))

    return {
        "sharpe": float(sum(m["sharpe"] for m in metrics_list) / len(metrics_list)),
        "total_return_pct": float(
            sum(m["total_return_pct"] for m in metrics_list) / len(metrics_list)
        ),
        "max_drawdown_pct": float(min(m["max_drawdown_pct"] for m in metrics_list)),
        "final_equity": float(
            sum(m["final_equity"] for m in metrics_list) / len(metrics_list)
        ),
        "regimes": len(metrics_list),
    }


def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS leaderboard (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                member_name TEXT UNIQUE,
                strategy_name TEXT,
                sharpe REAL,
                excess_sharpe REAL,
                total_return_pct REAL,
                alpha_pct REAL,
                beta REAL,
                max_drawdown_pct REAL,
                submitted_at TEXT
            )
            """)


def upsert_submission(member_name, strategy_name, res):
    excess_sharpe = res["sharpe"] - res["spy_sharpe"]
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO leaderboard (
                member_name, strategy_name, sharpe, excess_sharpe,
                total_return_pct, alpha_pct, beta, max_drawdown_pct, submitted_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(member_name) DO UPDATE SET
                strategy_name=excluded.strategy_name,
                sharpe=excluded.sharpe,
                excess_sharpe=excluded.excess_sharpe,
                total_return_pct=excluded.total_return_pct,
                alpha_pct=excluded.alpha_pct,
                beta=excluded.beta,
                max_drawdown_pct=excluded.max_drawdown_pct,
                submitted_at=excluded.submitted_at
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
                sharpe AS "OOS Sharpe",
                excess_sharpe AS "vs SPY Sharpe",
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


init_db()
st.set_page_config(page_title="Quant Club Prosperity Leaderboard", layout="wide")
st.title("Nova Quant Club — Strategy Leaderboard")

sections = [
    "Leaderboard",
    "Backtester",
    "Submit to Leaderboard",
    "Guide",
    "Admin",
]
if st.session_state.pop("redirect_to_leaderboard", False):
    st.session_state["section_selector"] = "Leaderboard"
active_section = st.radio(
    "Section",
    sections,
    horizontal=True,
    label_visibility="collapsed",
    key="section_selector",
)

# --- TAB 1: LEADERBOARD ---
if active_section == "Leaderboard":
    st.subheader("Official Out-Of-Sample Rankings")
    st.caption(
        "Evaluated across undisclosed multi-regime market windows | Ranked by Mean Annualized Sharpe Ratio (10 bps fee/slippage, 4% RF rate)"
    )

    # Pinned SPY Benchmark Reference Bar
    try:
        spy_bench = get_oos_spy_benchmark(SECRET_OOS_WINDOWS)
        st.markdown(
            f"##### 📌 Benchmark Reference: `SPY` Buy-and-Hold (Mean across {spy_bench['regimes']} Secret Regimes)"
        )
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("SPY Mean Sharpe", f"{spy_bench['sharpe']:.3f}")
        b2.metric("SPY Mean Return", f"{spy_bench['total_return_pct']:.2f}%")
        b3.metric("SPY Worst Drawdown", f"{spy_bench['max_drawdown_pct']:.2f}%")
        b4.metric("SPY Avg Final Equity ($100k)", f"${spy_bench['final_equity']:,.2f}")
    except Exception as e:
        st.warning(f"Could not load SPY benchmark banner: {e}")

    st.divider()
    lb_df = get_leaderboard()
    if lb_df.empty:
        st.info("No strategies submitted yet. Head over to the Submit tab to beat SPY!")
    else:
        st.dataframe(lb_df, width="stretch")

# --- TAB 2: WEB SANDBOX BACKTESTER ---
if active_section == "Backtester":
    st.subheader("Interactive In-Sample Web Backtester (vs. SPY Buy & Hold)")
    st.write(
        "Test your `.py` strategy file on historical In-Sample dates and compare your equity curve directly against `SPY`."
    )

    col1, col2 = st.columns(2)
    with col1:
        is_start = st.date_input(
            "Start Date",
            value=pd.to_datetime("2023-01-01"),
        )
    with col2:
        is_end = st.date_input(
            "End Date",
            value=pd.to_datetime("2024-12-31"),
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
if active_section == "Submit to Leaderboard":
    st.subheader("Submit Strategy for Official OOS Evaluation")
    st.caption(
        f"Strategies are evaluated across multiple undisclosed historical market regimes "
        f"with anonymized timestamps and base-100 prices. Submission cooldown: **{COOLDOWN_HOURS:g} hours**."
    )

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
                        c for c in clean_strat if c.isalnum() or c in ("_", "-")
                    )
                    if not safe_name:
                        raise ValueError(
                            "Trader.name must contain at least one letter or number."
                        )

                    with sqlite3.connect(DB_PATH) as conn:
                        row = conn.execute(
                            "SELECT submitted_at FROM leaderboard WHERE member_name = ?",
                            (clean_member,),
                        ).fetchone()

                    cooldown_blocked = False
                    if row and row[0]:
                        try:
                            last_sub = datetime.strptime(row[0], "%Y-%m-%d %H:%M UTC")
                            hours_since = (
                                datetime.utcnow() - last_sub
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
                            res = bt.run_multi_window(saved_path, SECRET_OOS_WINDOWS)

                            upsert_submission(clean_member, clean_strat, res)
                            diff = res["sharpe"] - res["spy_sharpe"]
                            st.success(
                                f"✅ Submitted across **{res['regimes_tested']}** hidden regimes! "
                                f"Mean OOS Sharpe: **{res['sharpe']:.3f}** ({diff:+.3f} vs SPY) | "
                                f"Worst Regime Sharpe: **{res['min_regime_sharpe']:.3f}** | "
                                f"Mean Alpha: **{res['alpha_pct']:+.2f}%** | "
                                f"Mean Beta: **{res['beta']:.2f}**"
                            )
                            st.session_state["redirect_to_leaderboard"] = True
                            st.rerun()
                except Exception as e:
                    if saved_path and os.path.exists(saved_path) and not row:
                        os.remove(saved_path)
                    st.error(f"Evaluation failed: {e}")
                finally:
                    if tmp_path and os.path.exists(tmp_path):
                        os.remove(tmp_path)
# --- TAB 4: LOCAL SDK & RULES ---
if active_section == "Guide":
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
    3. **SPY Benchmark:** `SPY` Buy-and-Hold is initialized with $100,000 at the first day's Open (less the 10 bps fee) and held through the end of the evaluation window.
    4. **Ranking Metric:** Annualized Daily Sharpe Ratio ($R_f = 4\\%$, $\\sqrt{252}$ scaling) over the official Out-Of-Sample date window, alongside CAPM Alpha ($\\alpha$) and Beta ($\\beta$) relative to `SPY`.
    """)
    st.code(sample_code, language="python")

if active_section == "Admin":
    st.subheader("Organizer Administration")
    admin_pw = st.text_input("Admin Password", type="password", key="admin_password")

    if admin_pw != ADMIN_PASSWORD:
        st.info("Enter the admin password to manage submitted strategies.")
    else:
        st.success("Admin access granted.")
        new_windows = st.text_input(
            "Evaluation Window(s)",
            value=SECRET_OOS_WINDOWS,
            placeholder="YYYY-MM-DD:YYYY-MM-DD,YYYY-MM-DD:YYYY-MM-DD",
            help="Use comma-separated start:end windows.",
            key="admin_windows",
        )

        st.markdown("#### Re-evaluate strategies")
        if st.button("Re-Score Entire Leaderboard", key="admin_rescore"):
            if not new_windows.strip():
                st.warning("Please specify at least one evaluation window.")
            else:
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
                            result = bt.run_multi_window(fpath, new_windows.strip())
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
                    member, _ = get_strategy_metadata(delete_path)
                    with sqlite3.connect(DB_PATH) as conn:
                        conn.execute(
                            "DELETE FROM leaderboard WHERE member_name = ?",
                            (member,),
                        )
                    os.remove(delete_path)
                    st.success(f"Deleted {delete_file} and its leaderboard entry.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Could not delete {delete_file}: {e}")
