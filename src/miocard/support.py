from __future__ import annotations

import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True, slots=True)
class AnalysisCase:
    case_id: str
    user_id: int
    created_at: datetime
    outcome: str


@dataclass(frozen=True, slots=True)
class PendingSupport:
    category: str
    analysis_case_id: str | None


@dataclass(frozen=True, slots=True)
class SupportRoute:
    ticket_id: str
    user_id: int
    moderator_id: int | None


class SupportStore:
    """Persistent support routing metadata without storing message contents."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS analysis_cases (
                    case_id TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    outcome TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_analysis_cases_user_created
                    ON analysis_cases(user_id, created_at DESC);

                CREATE TABLE IF NOT EXISTS pending_support (
                    user_id INTEGER PRIMARY KEY,
                    category TEXT NOT NULL,
                    analysis_case_id TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (analysis_case_id) REFERENCES analysis_cases(case_id)
                );

                CREATE TABLE IF NOT EXISTS support_tickets (
                    ticket_id TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    category TEXT NOT NULL,
                    analysis_case_id TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (analysis_case_id) REFERENCES analysis_cases(case_id)
                );

                CREATE TABLE IF NOT EXISTS support_routes (
                    chat_id INTEGER NOT NULL,
                    message_id INTEGER NOT NULL,
                    ticket_id TEXT NOT NULL,
                    user_id INTEGER NOT NULL,
                    moderator_id INTEGER,
                    PRIMARY KEY (chat_id, message_id),
                    FOREIGN KEY (ticket_id) REFERENCES support_tickets(ticket_id)
                );

                CREATE TABLE IF NOT EXISTS known_users (
                    user_id INTEGER PRIMARY KEY,
                    first_seen TEXT NOT NULL,
                    last_seen TEXT NOT NULL
                );
                """
            )

    def record_user_seen(self, user_id: int) -> None:
        """Remember that this Telegram user has interacted with the bot.

        This is the only record of who has ever used the bot when
        ``ALLOWED_USER_IDS`` is left empty for open access, so it is the list an
        admin broadcast (``/announce``, ``/shutdown``) reaches.
        """
        now = datetime.now().astimezone().isoformat()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO known_users(user_id, first_seen, last_seen)
                VALUES (?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET last_seen = excluded.last_seen
                """,
                (user_id, now, now),
            )

    def list_known_user_ids(self) -> list[int]:
        with self._connect() as connection:
            rows = connection.execute("SELECT user_id FROM known_users").fetchall()
        return [int(row["user_id"]) for row in rows]

    def create_analysis_case(self, user_id: int, created_at: datetime) -> AnalysisCase:
        case = AnalysisCase(
            case_id=uuid.uuid4().hex,
            user_id=user_id,
            created_at=created_at,
            outcome="В обработке",
        )
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO analysis_cases(case_id, user_id, created_at, outcome)
                VALUES (?, ?, ?, ?)
                """,
                (case.case_id, case.user_id, case.created_at.isoformat(), case.outcome),
            )
        return case

    def update_analysis_outcome(self, case_id: str, outcome: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE analysis_cases SET outcome = ? WHERE case_id = ?",
                (outcome, case_id),
            )

    @staticmethod
    def _analysis_case(row: sqlite3.Row | None) -> AnalysisCase | None:
        if row is None:
            return None
        return AnalysisCase(
            case_id=str(row["case_id"]),
            user_id=int(row["user_id"]),
            created_at=datetime.fromisoformat(str(row["created_at"])),
            outcome=str(row["outcome"]),
        )

    def get_analysis_case(self, case_id: str) -> AnalysisCase | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM analysis_cases WHERE case_id = ?",
                (case_id,),
            ).fetchone()
        return self._analysis_case(row)

    def latest_analysis_case(self, user_id: int) -> AnalysisCase | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM analysis_cases
                WHERE user_id = ?
                ORDER BY created_at DESC, rowid DESC
                LIMIT 1
                """,
                (user_id,),
            ).fetchone()
        return self._analysis_case(row)

    def set_pending(
        self,
        user_id: int,
        category: str,
        analysis_case_id: str | None,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO pending_support(user_id, category, analysis_case_id, created_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    category = excluded.category,
                    analysis_case_id = excluded.analysis_case_id,
                    created_at = excluded.created_at
                """,
                (
                    user_id,
                    category,
                    analysis_case_id,
                    datetime.now().astimezone().isoformat(),
                ),
            )

    def get_pending(self, user_id: int) -> PendingSupport | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT category, analysis_case_id FROM pending_support WHERE user_id = ?",
                (user_id,),
            ).fetchone()
        if row is None:
            return None
        return PendingSupport(
            category=str(row["category"]),
            analysis_case_id=(
                str(row["analysis_case_id"])
                if row["analysis_case_id"] is not None
                else None
            ),
        )

    def pop_pending(self, user_id: int) -> PendingSupport | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT category, analysis_case_id FROM pending_support WHERE user_id = ?",
                (user_id,),
            ).fetchone()
            if row is not None:
                connection.execute(
                    "DELETE FROM pending_support WHERE user_id = ?",
                    (user_id,),
                )
        if row is None:
            return None
        return PendingSupport(
            category=str(row["category"]),
            analysis_case_id=(
                str(row["analysis_case_id"])
                if row["analysis_case_id"] is not None
                else None
            ),
        )

    def create_ticket(
        self,
        user_id: int,
        category: str,
        analysis_case_id: str | None,
    ) -> str:
        ticket_id = uuid.uuid4().hex
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO support_tickets(
                    ticket_id, user_id, category, analysis_case_id, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    ticket_id,
                    user_id,
                    category,
                    analysis_case_id,
                    datetime.now().astimezone().isoformat(),
                ),
            )
        return ticket_id

    def add_route(
        self,
        *,
        chat_id: int,
        message_id: int,
        ticket_id: str,
        user_id: int,
        moderator_id: int | None,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO support_routes(
                    chat_id, message_id, ticket_id, user_id, moderator_id
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (chat_id, message_id, ticket_id, user_id, moderator_id),
            )

    def get_route(self, chat_id: int, message_id: int) -> SupportRoute | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT ticket_id, user_id, moderator_id
                FROM support_routes
                WHERE chat_id = ? AND message_id = ?
                """,
                (chat_id, message_id),
            ).fetchone()
        if row is None:
            return None
        return SupportRoute(
            ticket_id=str(row["ticket_id"]),
            user_id=int(row["user_id"]),
            moderator_id=(
                int(row["moderator_id"])
                if row["moderator_id"] is not None
                else None
            ),
        )
