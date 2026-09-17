"""
Service launcher for Nambikkai AI Intelligence Service on Windows.

Must set WindowsSelectorEventLoopPolicy BEFORE uvicorn creates its event loop.
This is required because psycopg3 async mode is incompatible with ProactorEventLoop
(Python 3.8+ Windows default). The policy in app/main.py runs too late when uvicorn
is invoked via the CLI.
"""
import asyncio
import sys

# Must be set BEFORE uvicorn is imported or started
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())  # type: ignore # pyright: ignore[reportDeprecated]

import uvicorn

if __name__ == "__main__":
    from app.core.config import get_settings
    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.APP_HOST,
        port=settings.APP_PORT,
        reload=False,
    )
