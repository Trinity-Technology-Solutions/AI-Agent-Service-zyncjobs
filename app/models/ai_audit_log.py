import sqlite3
import os
import json
from datetime import datetime
from typing import Optional

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "ai_audit.db")


def get_db() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ai_audit_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            request_id TEXT NOT NULL,
            feature_name TEXT NOT NULL,
            endpoint TEXT,
            model TEXT NOT NULL,
            user_id TEXT DEFAULT 'anonymous',
            prompt_tokens INTEGER DEFAULT 0,
            completion_tokens INTEGER DEFAULT 0,
            latency_ms INTEGER NOT NULL,
            status TEXT NOT NULL DEFAULT 'SUCCESS',
            fallback_used INTEGER NOT NULL DEFAULT 0,
            error_message TEXT,
            prompt_preview TEXT,
            response_preview TEXT,
            created_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_audit_feature ON ai_audit_logs(feature_name)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_audit_created ON ai_audit_logs(created_at)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_audit_status ON ai_audit_logs(status)
    """)
    conn.commit()
    conn.close()


def save_log(
    request_id: str,
    feature_name: str,
    endpoint: Optional[str],
    model: str,
    user_id: str = "anonymous",
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    latency_ms: int = 0,
    status: str = "SUCCESS",
    fallback_used: bool = False,
    error_message: Optional[str] = None,
    prompt_preview: Optional[str] = None,
    response_preview: Optional[str] = None,
):
    conn = get_db()
    conn.execute(
        """INSERT INTO ai_audit_logs
           (request_id, feature_name, endpoint, model, user_id,
            prompt_tokens, completion_tokens, latency_ms,
            status, fallback_used, error_message,
            prompt_preview, response_preview, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            request_id,
            feature_name,
            endpoint or "",
            model,
            user_id,
            prompt_tokens,
            completion_tokens,
            latency_ms,
            status,
            1 if fallback_used else 0,
            error_message or "",
            (prompt_preview or "")[:500],
            (response_preview or "")[:500],
            datetime.utcnow().isoformat(),
        ),
    )
    conn.commit()
    conn.close()


def get_logs(
    feature: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
) -> list[dict]:
    conn = get_db()
    query = "SELECT * FROM ai_audit_logs WHERE 1=1"
    params: list = []
    if feature:
        query += " AND feature_name = ?"
        params.append(feature)
    if status:
        query += " AND status = ?"
        params.append(status)
    query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_stats() -> dict:
    conn = get_db()
    total = conn.execute("SELECT COUNT(*) as c FROM ai_audit_logs").fetchone()["c"]
    today = datetime.utcnow().strftime("%Y-%m-%d")
    today_count = conn.execute(
        "SELECT COUNT(*) as c FROM ai_audit_logs WHERE created_at LIKE ?", (f"{today}%",)
    ).fetchone()["c"]
    success = conn.execute(
        "SELECT COUNT(*) as c FROM ai_audit_logs WHERE status='SUCCESS'"
    ).fetchone()["c"]
    failed = conn.execute(
        "SELECT COUNT(*) as c FROM ai_audit_logs WHERE status='FAILED'"
    ).fetchone()["c"]
    fallback = conn.execute(
        "SELECT COUNT(*) as c FROM ai_audit_logs WHERE fallback_used=1"
    ).fetchone()["c"]
    avg_latency = conn.execute(
        "SELECT AVG(latency_ms) as a FROM ai_audit_logs"
    ).fetchone()["a"] or 0

    features = conn.execute(
        """SELECT feature_name, COUNT(*) as count,
                  AVG(latency_ms) as avg_latency,
                  SUM(CASE WHEN status='SUCCESS' THEN 1 ELSE 0 END) as success_count,
                  SUM(CASE WHEN status='FAILED' THEN 1 ELSE 0 END) as failed_count,
                  SUM(fallback_used) as fallback_count
           FROM ai_audit_logs
           GROUP BY feature_name
           ORDER BY count DESC"""
    ).fetchall()

    conn.close()
    return {
        "total": total,
        "today": today_count,
        "success": success,
        "failed": failed,
        "fallback": fallback,
        "avg_latency_ms": round(avg_latency, 1),
        "features": [dict(f) for f in features],
    }


init_db()
