"""Transactional team submissions, frozen rounds, and blind final evaluation."""

import hashlib
import io
import json
import secrets
import sqlite3
import time
import uuid
import zipfile
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

from backtester import Backtester, EngineConfig, evaluation_id, load_training
from strategy_runner import (
    MAX_SOURCE,
    RemoteTrader,
    validate_metadata,
    worker_runtime_id,
)
from synthetic_market import (
    COLUMNS,
    SyntheticConfig,
    SyntheticRound,
    calibrate_market,
    generate_round,
)


class CompetitionError(ValueError):
    pass


class Competition:
    def __init__(self, db_path="data/leaderboard.db", round_root="data/rounds"):
        self.db_path, self.round_root = Path(db_path), Path(round_root)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self.db() as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS teams (
                    id TEXT PRIMARY KEY, name TEXT NOT NULL UNIQUE, token_hash TEXT UNIQUE NOT NULL
                );
                CREATE TABLE IF NOT EXISTS competition_rounds (
                    id TEXT PRIMARY KEY, name TEXT NOT NULL, phase TEXT NOT NULL,
                    dataset_id TEXT NOT NULL, evaluation_id TEXT NOT NULL, training_hash TEXT NOT NULL,
                    config TEXT NOT NULL, public_dir TEXT NOT NULL, private_dir TEXT NOT NULL,
                    max_submissions INTEGER NOT NULL, cooldown_seconds INTEGER NOT NULL,
                    lease TEXT, lease_until REAL
                );
                CREATE TABLE IF NOT EXISTS strategy_versions (
                    id TEXT PRIMARY KEY, round_id TEXT NOT NULL REFERENCES competition_rounds(id),
                    team_id TEXT NOT NULL REFERENCES teams(id), source TEXT NOT NULL,
                    source_hash TEXT NOT NULL, name TEXT NOT NULL, tickers TEXT NOT NULL,
                    submitted_at REAL NOT NULL, UNIQUE(round_id, team_id, source_hash)
                );
                CREATE TABLE IF NOT EXISTS selections (
                    round_id TEXT NOT NULL REFERENCES competition_rounds(id),
                    team_id TEXT NOT NULL REFERENCES teams(id), version_id TEXT NOT NULL REFERENCES strategy_versions(id),
                    result TEXT, error TEXT, PRIMARY KEY(round_id, team_id)
                );
            """)

    @contextmanager
    def db(self, *, write=False):
        db = sqlite3.connect(self.db_path, timeout=20)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            if write:
                db.execute("BEGIN IMMEDIATE")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def create_team(self, name):
        if not isinstance(name, str) or not name.strip() or len(name) > 100:
            raise CompetitionError("Team names must contain 1 to 100 characters.")
        token, team_id = secrets.token_urlsafe(32), uuid.uuid4().hex
        with self.db(write=True) as db:
            db.execute(
                "INSERT INTO teams VALUES (?, ?, ?)",
                (team_id, name.strip(), hashlib.sha256(token.encode()).hexdigest()),
            )
        return token

    def authenticate(self, token):
        if not isinstance(token, str) or not 32 <= len(token) <= 256:
            return None
        with self.db() as db:
            row = db.execute(
                "SELECT id, name FROM teams WHERE token_hash = ?",
                (hashlib.sha256(token.encode()).hexdigest(),),
            ).fetchone()
        return dict(row) if row else None

    def teams(self):
        with self.db() as db:
            return [
                dict(row)
                for row in db.execute("SELECT id, name FROM teams ORDER BY name")
            ]

    def rotate_token(self, team_id):
        token = secrets.token_urlsafe(32)
        with self.db(write=True) as db:
            if not db.execute(
                "UPDATE teams SET token_hash=? WHERE id=?",
                (hashlib.sha256(token.encode()).hexdigest(), team_id),
            ).rowcount:
                raise CompetitionError("Unknown team.")
        return token

    def rounds(self):
        with self.db() as db:
            return [
                dict(row)
                for row in db.execute(
                    "SELECT * FROM competition_rounds ORDER BY rowid DESC"
                )
            ]

    def round(self, round_id):
        with self.db() as db:
            row = db.execute(
                "SELECT * FROM competition_rounds WHERE id=?", (round_id,)
            ).fetchone()
        if row is None:
            raise CompetitionError("Unknown round.")
        return dict(row)

    def create_round(
        self,
        name,
        training_directory,
        config,
        synthetic_config=None,
        *,
        max_submissions=10,
        cooldown_seconds=1800,
    ):
        if (
            not name.strip()
            or len(name) > 100
            or not 1 <= max_submissions <= 100
            or not 0 <= cooldown_seconds <= 86400
        ):
            raise CompetitionError("Invalid round name or submission limits.")
        synthetic_config = synthetic_config or SyntheticConfig(
            warmup_bars=config.warmup_bars
        )
        if (
            config.warmup_bars != synthetic_config.warmup_bars
            or config.runtime_id == "trusted-local"
        ):
            raise CompetitionError(
                "A final round needs matching warm-up settings and a pinned worker runtime."
            )
        training = load_training(training_directory)
        if (
            config.benchmark not in training
            or len(next(iter(training.values()))) < config.warmup_bars + 10
        ):
            raise CompetitionError(
                "Training must contain the benchmark, warm-up, and at least 10 practice bars."
            )
        round_id = uuid.uuid4().hex
        root = (self.round_root / round_id).resolve()
        public_dir, private_dir = root / "training", root / "final"
        public_dir.mkdir(parents=True)
        for symbol, frame in training.items():
            frame.loc[:, list(COLUMNS)].to_csv(
                public_dir / f"{symbol}.csv", index_label=frame.index.name or "Step"
            )
        model = calibrate_market(load_training(public_dir))
        seed = secrets.randbits(128)
        dataset = generate_round(model, seed, synthetic_config)
        dataset.save(private_dir, private_seed=seed)
        score_id = evaluation_id(dataset.dataset_id, config)
        with self.db(write=True) as db:
            db.execute(
                """INSERT INTO competition_rounds
                (id,name,phase,dataset_id,evaluation_id,training_hash,config,public_dir,private_dir,max_submissions,cooldown_seconds)
                VALUES (?,?,'open',?,?,?,?,?,?,?,?)""",
                (
                    round_id,
                    name.strip(),
                    dataset.dataset_id,
                    score_id,
                    model.training_sha256,
                    json.dumps(asdict(config)),
                    str(public_dir),
                    str(private_dir),
                    max_submissions,
                    cooldown_seconds,
                ),
            )
        return round_id

    def public_market(self, round_id):
        row = self.round(round_id)
        market = load_training(row["public_dir"])
        if calibrate_market(market).training_sha256 != row["training_hash"]:
            raise CompetitionError(
                "Public training snapshot changed. Restore the frozen files."
            )
        return market

    def training_zip(self, round_id):
        market = self.public_market(round_id)
        config = EngineConfig(**json.loads(self.round(round_id)["config"]))
        output = io.BytesIO()
        with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
            for symbol, frame in market.items():
                archive.writestr(
                    f"{symbol}.csv",
                    frame.to_csv(index_label=frame.index.name or "Step"),
                )
            archive.writestr(
                "README.md",
                "# Nova Quant League training data\n\n"
                f"Competition assets are {', '.join(sorted(market))}.\n\n"
                "Each CSV has one daily observation per row on a shared calendar. "
                "The first column is the observation date or step, followed by "
                "Open, High, Low, Close and Volume. Prices use consistent corporate-action "
                "adjustments. Volume records share counts.\n\n"
                "Investigate asset behaviour and relationships using the supplied observations. "
                "Teams develop their own signals and allocation rules.\n\n"
                f"The warm-up contains {config.warmup_bars} bars. Strategy history uses integer "
                "steps and normalized prices and volume. Orders execute at the next open. "
                f"Costs are {config.commission_bps:g} basis points per trade. "
                f"The comparison benchmark buys and holds {config.benchmark}.\n\n"
                "Final evaluation uses independent synthetic paths calibrated from this dataset.\n",
            )
        return output.getvalue()

    def submit(self, round_id, team_id, source, metadata):
        metadata = validate_metadata(metadata)
        if not isinstance(source, str) or not 0 < len(source.encode()) <= MAX_SOURCE:
            raise CompetitionError("Strategy source must be between 1 byte and 128 KB.")
        digest = hashlib.sha256(source.encode()).hexdigest()
        universe = set(self.public_market(round_id))
        if set(metadata["tickers"]) - universe:
            raise CompetitionError("Use only assets in this round's training dataset.")
        now = time.time()
        with self.db(write=True) as db:
            row = db.execute(
                "SELECT * FROM competition_rounds WHERE id=?", (round_id,)
            ).fetchone()
            if row["phase"] != "open":
                raise CompetitionError("Submissions are locked for this round.")
            existing = db.execute(
                "SELECT id FROM strategy_versions WHERE round_id=? AND team_id=? AND source_hash=?",
                (round_id, team_id, digest),
            ).fetchone()
            if existing:
                return existing["id"]
            counts = db.execute(
                "SELECT COUNT(*) AS n, MAX(submitted_at) AS latest FROM strategy_versions WHERE round_id=? AND team_id=?",
                (round_id, team_id),
            ).fetchone()
            if counts["n"] >= row["max_submissions"]:
                raise CompetitionError(
                    "Your team has reached this round's submission limit."
                )
            if (
                counts["latest"] is not None
                and now - counts["latest"] < row["cooldown_seconds"]
            ):
                remaining = int(row["cooldown_seconds"] - (now - counts["latest"])) + 1
                raise CompetitionError(
                    f"Team cooldown: try again in {remaining} seconds."
                )
            version_id = uuid.uuid4().hex
            db.execute(
                "INSERT INTO strategy_versions VALUES (?,?,?,?,?,?,?,?)",
                (
                    version_id,
                    round_id,
                    team_id,
                    source,
                    digest,
                    metadata["name"],
                    json.dumps(metadata["tickers"]),
                    now,
                ),
            )
        return version_id

    def versions(self, round_id, team_id):
        with self.db() as db:
            return [
                dict(row)
                for row in db.execute(
                    """SELECT id,name,source_hash,submitted_at FROM strategy_versions
                WHERE round_id=? AND team_id=? ORDER BY submitted_at DESC""",
                    (round_id, team_id),
                )
            ]

    def select(self, round_id, team_id, version_id):
        with self.db(write=True) as db:
            row = db.execute(
                "SELECT phase FROM competition_rounds WHERE id=?", (round_id,)
            ).fetchone()
            if row is None or row["phase"] != "open":
                raise CompetitionError("Final selections are locked.")
            version = db.execute(
                "SELECT id FROM strategy_versions WHERE id=? AND round_id=? AND team_id=?",
                (version_id, round_id, team_id),
            ).fetchone()
            if not version:
                raise CompetitionError(
                    "Select a strategy uploaded by your own team in this round."
                )
            db.execute(
                """INSERT INTO selections(round_id,team_id,version_id) VALUES (?,?,?)
                ON CONFLICT(round_id,team_id) DO UPDATE SET version_id=excluded.version_id,result=NULL,error=NULL""",
                (round_id, team_id, version_id),
            )

    def selection(self, round_id, team_id):
        with self.db() as db:
            row = db.execute(
                "SELECT version_id FROM selections WHERE round_id=? AND team_id=?",
                (round_id, team_id),
            ).fetchone()
        return row[0] if row else None

    def lock_round(self, round_id):
        with self.db(write=True) as db:
            if not db.execute(
                "SELECT 1 FROM selections WHERE round_id=?", (round_id,)
            ).fetchone():
                raise CompetitionError(
                    "At least one team must select a final strategy."
                )
            if not db.execute(
                "UPDATE competition_rounds SET phase='locked' WHERE id=? AND phase='open'",
                (round_id,),
            ).rowcount:
                raise CompetitionError("This round is already locked.")

    def score(
        self, round_id, *, runtime_id=None, trader_factory=RemoteTrader, progress=None
    ):
        row = self.round(round_id)
        config = EngineConfig(**json.loads(row["config"]))
        current_runtime = runtime_id if runtime_id is not None else worker_runtime_id()
        if current_runtime != config.runtime_id:
            raise CompetitionError(
                "Worker runtime changed. Restore the pinned runtime before scoring this round."
            )
        dataset = SyntheticRound.load(
            row["private_dir"], expected_dataset_id=row["dataset_id"]
        )
        if evaluation_id(dataset.dataset_id, config) != row["evaluation_id"]:
            raise CompetitionError(
                "Engine settings changed; this round cannot be silently re-scored."
            )
        lease, now = uuid.uuid4().hex, time.time()
        with self.db(write=True) as db:
            if not db.execute(
                """UPDATE competition_rounds SET phase='scoring',lease=?,lease_until=? WHERE id=?
                AND (phase IN ('locked','scored') OR (phase='scoring' AND lease_until < ?))""",
                (
                    lease,
                    now + dataset.manifest["config"]["paths"] * 46 + 60,
                    round_id,
                    now,
                ),
            ).rowcount:
                raise CompetitionError(
                    "This round is open, published, or already being evaluated."
                )
            jobs = [
                dict(job)
                for job in db.execute(
                    """SELECT s.team_id, v.source, t.name FROM selections s
                JOIN strategy_versions v ON v.id=s.version_id JOIN teams t ON t.id=s.team_id
                WHERE s.round_id=? AND s.result IS NULL""",
                    (round_id,),
                )
            ]
        engine = Backtester(config)
        for index, job in enumerate(jobs):
            if progress:
                progress(index, len(jobs), job["name"])
            with self.db(write=True) as db:
                if not db.execute(
                    "UPDATE competition_rounds SET lease_until=? WHERE id=? AND lease=?",
                    (
                        time.time() + dataset.manifest["config"]["paths"] * 46 + 60,
                        round_id,
                        lease,
                    ),
                ).rowcount:
                    raise CompetitionError("Evaluation lease was lost.")
            result, error = None, None
            try:
                result = json.dumps(
                    engine.run_synthetic(
                        job["source"], dataset, trader_factory=trader_factory
                    ),
                    allow_nan=False,
                )
            except Exception as exc:  # noqa: BLE001 - isolate failures to one team's final entry
                error = f"{type(exc).__name__}: {str(exc)[:600]}"
            with self.db(write=True) as db:
                if (
                    db.execute(
                        "SELECT lease FROM competition_rounds WHERE id=?", (round_id,)
                    ).fetchone()[0]
                    != lease
                ):
                    raise CompetitionError("Evaluation lease was lost.")
                db.execute(
                    "UPDATE selections SET result=?,error=? WHERE round_id=? AND team_id=?",
                    (result, error, round_id, job["team_id"]),
                )
        with self.db(write=True) as db:
            db.execute(
                "UPDATE competition_rounds SET phase='scored',lease=NULL,lease_until=NULL WHERE id=? AND lease=?",
                (round_id, lease),
            )

    def publish(self, round_id):
        with self.db(write=True) as db:
            if db.execute(
                "SELECT 1 FROM selections WHERE round_id=? AND result IS NULL AND error IS NULL",
                (round_id,),
            ).fetchone():
                raise CompetitionError(
                    "Finish evaluating every selected strategy first."
                )
            if not db.execute(
                "UPDATE competition_rounds SET phase='published' WHERE id=? AND phase='scored'",
                (round_id,),
            ).rowcount:
                raise CompetitionError("Only a scored, locked round can be published.")

    def results(self, round_id, *, organizer=False):
        if not organizer and self.round(round_id)["phase"] != "published":
            return []
        with self.db() as db:
            return [
                dict(row)
                for row in db.execute(
                    """SELECT t.name AS team,v.name AS strategy,s.result,s.error
                FROM selections s JOIN teams t ON t.id=s.team_id JOIN strategy_versions v ON v.id=s.version_id
                WHERE s.round_id=?""",
                    (round_id,),
                )
            ]
