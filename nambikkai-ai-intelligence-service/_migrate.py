"""Apply migration 010 — add XGBoost columns to ai_suggestions."""
import asyncio, sys
asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

async def apply():
    from app.data_sources.postgres import open_pool, close_pool, _get_pool
    await open_pool()
    pool = _get_pool()
    async with pool.connection() as conn:
        await conn.execute(
            "ALTER TABLE ai_suggestions "
            "ADD COLUMN IF NOT EXISTS xgboost_surge_probability NUMERIC(8,6), "
            "ADD COLUMN IF NOT EXISTS xgboost_predicted_surge BOOLEAN"
        )
        await conn.commit()
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name='ai_suggestions' ORDER BY ordinal_position"
            )
            cols = [r[0] for r in await cur.fetchall()]
            print("Columns:", cols)
    await close_pool()
    print("Migration applied.")

asyncio.run(apply())
