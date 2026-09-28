"""Apply migration 011 — AI Performance Insights Refactor."""
import asyncio, sys
asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

async def apply():
    from app.data_sources.postgres import open_pool, close_pool, _get_pool
    await open_pool()
    pool = _get_pool()
    async with pool.connection() as conn:
        print("Applying migration 011...")
        await conn.execute(
            """
            ALTER TABLE ai_suggestions
                ADD COLUMN IF NOT EXISTS content_type       TEXT,
                ADD COLUMN IF NOT EXISTS canonical_url      TEXT,
                ADD COLUMN IF NOT EXISTS metric_name        TEXT,
                ADD COLUMN IF NOT EXISTS likes              BIGINT DEFAULT 0,
                ADD COLUMN IF NOT EXISTS comments           BIGINT DEFAULT 0,
                ADD COLUMN IF NOT EXISTS peer_explanation   TEXT;

            ALTER TABLE ai_suggestions
                DROP COLUMN IF EXISTS velocity_ratio,
                DROP COLUMN IF EXISTS like_acceleration,
                DROP COLUMN IF EXISTS xgboost_surge_probability,
                DROP COLUMN IF EXISTS xgboost_predicted_surge,
                DROP COLUMN IF EXISTS is_surge,
                DROP COLUMN IF EXISTS coverage_hours;

            CREATE INDEX IF NOT EXISTS ai_suggestions_period_idx
                ON ai_suggestions (report_period, platform, analyzed_at DESC);

            CREATE INDEX IF NOT EXISTS ai_suggestions_classification_idx
                ON ai_suggestions (classification, analyzed_at DESC);
            """
        )
        await conn.commit()
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name='ai_suggestions' ORDER BY ordinal_position"
            )
            cols = [r[0] for r in await cur.fetchall()]
            print("Current columns in ai_suggestions:", cols)
    await close_pool()
    print("Migration 011 successfully applied.")

if __name__ == "__main__":
    asyncio.run(apply())
