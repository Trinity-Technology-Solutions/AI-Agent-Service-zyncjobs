from recruitment_ai.api.main import app
routes = [r for r in app.routes if hasattr(r, "path")]
ai_routes = [r for r in routes if "/ai/" in r.path]
print(f"Total: {len(routes)} | AI: {len(ai_routes)}")
for r in sorted(routes, key=lambda x: x.path):
    if hasattr(r, "methods"):
        methods = ",".join(r.methods)
        print(f"  {methods:7s} {r.path}")
