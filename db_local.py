# db_local.py
import os, sqlite3, datetime
from typing import Optional, Tuple, Dict, Any

DB_PATH = os.getenv("APP_DB_PATH", "./app.db")

_SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    tg_id         INTEGER NOT NULL,
    username      TEXT,
    first_name    TEXT,
    last_name     TEXT,
    created_at    TEXT NOT NULL,
    last_seen     TEXT NOT NULL,
    UNIQUE(tg_id)
);
CREATE TABLE IF NOT EXISTS questions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id       INTEGER NOT NULL,
    question      TEXT NOT NULL,
    answered      INTEGER NOT NULL DEFAULT 0,    -- 0/1
    stage_answered TEXT,                         -- 'local' | 'site' | 'web' | NULL
    created_at    TEXT NOT NULL,
    FOREIGN KEY(user_id) REFERENCES users(id)
);
CREATE TABLE IF NOT EXISTS unanswered (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    question_id   INTEGER NOT NULL,
    reason        TEXT,                          -- например: 'no_local_hits'
    created_at    TEXT NOT NULL,
    FOREIGN KEY(question_id) REFERENCES questions(id)
);
CREATE TABLE IF NOT EXISTS answer_scores (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    question_id   INTEGER NOT NULL,
    source        TEXT,               -- имя файла
    source_group  TEXT,               -- spravochnik / teach / ...
    page_from     INTEGER,
    path          TEXT,
    score         REAL,               -- числовая оценка (как пришла из векторного поиска)
    accepted      INTEGER NOT NULL,   -- 1 если показали пользователю, 0 если отфильтровали
    created_at    TEXT NOT NULL,
    FOREIGN KEY(question_id) REFERENCES questions(id)
);

"""

def _conn():
    con = sqlite3.connect(DB_PATH)
    con.execute("PRAGMA foreign_keys=ON;")
    return con

def init_db():
    con = _conn()
    try:
        con.executescript(_SCHEMA)
        con.commit()
    finally:
        con.close()

def upsert_user(tg_id: int, username: Optional[str], first_name: Optional[str], last_name: Optional[str]) -> int:
    now = datetime.datetime.utcnow().isoformat()
    con = _conn()
    try:
        cur = con.cursor()
        cur.execute("SELECT id FROM users WHERE tg_id=?", (tg_id,))
        row = cur.fetchone()
        if row:
            user_id = row[0]
            cur.execute(
                "UPDATE users SET username=?, first_name=?, last_name=?, last_seen=? WHERE id=?",
                (username, first_name, last_name, now, user_id)
            )
        else:
            cur.execute(
                "INSERT INTO users (tg_id, username, first_name, last_name, created_at, last_seen) VALUES (?,?,?,?,?,?)",
                (tg_id, username, first_name, last_name, now, now)
            )
            user_id = cur.lastrowid
        con.commit()
        return user_id
    finally:
        con.close()

def insert_question(user_id: int, question: str) -> int:
    now = datetime.datetime.utcnow().isoformat()
    con = _conn()
    try:
        cur = con.cursor()
        cur.execute(
            "INSERT INTO questions (user_id, question, answered, stage_answered, created_at) VALUES (?,?,?,?,?)",
            (user_id, question, 0, None, now)
        )
        qid = cur.lastrowid
        con.commit()
        return qid
    finally:
        con.close()

def mark_answered(question_id: int, stage: str):
    con = _conn()
    try:
        con.execute(
            "UPDATE questions SET answered=1, stage_answered=? WHERE id=?",
            (stage, question_id)
        )
        con.commit()
    finally:
        con.close()

def log_unanswered(question_id: int, reason: str = "no_local_hits"):
    now = datetime.datetime.utcnow().isoformat()
    con = _conn()
    try:
        con.execute(
            "INSERT INTO unanswered (question_id, reason, created_at) VALUES (?,?,?)",
            (question_id, reason, now)
        )
        con.commit()
    finally:
        con.close()

def log_answer_score(question_id: int, payload: dict, score: float | None, accepted: bool):
    now = datetime.datetime.utcnow().isoformat()
    con = _conn()
    try:
        con.execute(
            """INSERT INTO answer_scores
               (question_id, source, source_group, page_from, path, score, accepted, created_at)
               VALUES (?,?,?,?,?,?,?,?)""",
            (
                question_id,
                (payload or {}).get("source"),
                (payload or {}).get("source_group"),
                (payload or {}).get("page_from"),
                (payload or {}).get("path"),
                float(score) if score is not None else None,
                1 if accepted else 0,
                now,
            ),
        )
        con.commit()
    finally:
        con.close()


def fetch_unanswered(limit: int) -> list[dict[str, str | None]]:
    con = _conn()
    try:
        cur = con.cursor()
        cur.execute(
            """
            SELECT q.question, u.reason
            FROM unanswered u
            JOIN questions q ON u.question_id = q.id
            ORDER BY u.id DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = cur.fetchall()
        return [{"question": r[0], "reason": r[1]} for r in rows]
    finally:
        con.close()
