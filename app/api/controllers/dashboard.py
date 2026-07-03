"""Dashboard and API documentation HTML pages."""
from fastapi import APIRouter
from fastapi.responses import HTMLResponse
from app.config.settings import settings
from app.gateway.service_registry import service_registry
from app.knowledge.knowledge_base import knowledge_base
from app.memory.memory_manager import memory
from app.memory.cache import prompt_cache

router = APIRouter(tags=["Dashboard"])

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{name}</title>
  <script src="https://unpkg.com/lucide@latest"></script>
  <script src="https://cdn.tailwindcss.com"></script>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }
    .card-hover { transition: all 0.2s ease; }
    .card-hover:hover { transform: translateY(-2px); box-shadow: 0 8px 25px rgba(59, 130, 246, 0.15); }
    .glow { box-shadow: 0 0 20px rgba(59, 130, 246, 0.1); }
  </style>
</head>
<body class="bg-[#0F172A] text-white min-h-screen">
  <div class="max-w-5xl mx-auto px-4 py-8">
    <div class="text-center mb-10">
      <h1 class="text-3xl font-bold tracking-tight">{name}</h1>
      <p class="text-gray-400 mt-1 text-sm">Enterprise Agentic AI Platform</p>
    </div>
    <div class="flex justify-center mb-8">
      <div class="flex items-center gap-2 bg-emerald-500/10 text-emerald-400 px-4 py-1.5 rounded-full text-sm font-medium">
        <span class="w-2 h-2 rounded-full bg-emerald-400 inline-block animate-pulse"></span>
        Running
      </div>
    </div>
    <div class="grid grid-cols-2 md:grid-cols-4 gap-4 mb-10">
      <div class="bg-[#1E293B] rounded-xl p-4 card-hover glow">
        <div class="flex items-center gap-2 text-gray-400 text-xs uppercase tracking-wider mb-2"><i data-lucide="cpu" class="w-4 h-4 text-blue-400"></i><span>Model</span></div>
        <p class="text-lg font-semibold">{model}</p>
      </div>
      <div class="bg-[#1E293B] rounded-xl p-4 card-hover glow">
        <div class="flex items-center gap-2 text-gray-400 text-xs uppercase tracking-wider mb-2"><i data-lucide="bot" class="w-4 h-4 text-violet-400"></i><span>Agents</span></div>
        <p class="text-lg font-semibold">{agents}</p>
      </div>
      <div class="bg-[#1E293B] rounded-xl p-4 card-hover glow">
        <div class="flex items-center gap-2 text-gray-400 text-xs uppercase tracking-wider mb-2"><i data-lucide="settings" class="w-4 h-4 text-amber-400"></i><span>Services</span></div>
        <p class="text-lg font-semibold">{services}</p>
      </div>
      <div class="bg-[#1E293B] rounded-xl p-4 card-hover glow">
        <div class="flex items-center gap-2 text-gray-400 text-xs uppercase tracking-wider mb-2"><i data-lucide="wrench" class="w-4 h-4 text-cyan-400"></i><span>Tools</span></div>
        <p class="text-lg font-semibold">{tools}</p>
      </div>
      <div class="bg-[#1E293B] rounded-xl p-4 card-hover glow">
        <div class="flex items-center gap-2 text-gray-400 text-xs uppercase tracking-wider mb-2"><i data-lucide="book-open" class="w-4 h-4 text-emerald-400"></i><span>Knowledge</span></div>
        <p class="text-lg font-semibold">{docs} Docs</p>
      </div>
      <div class="bg-[#1E293B] rounded-xl p-4 card-hover glow">
        <div class="flex items-center gap-2 text-gray-400 text-xs uppercase tracking-wider mb-2"><i data-lucide="hard-drive" class="w-4 h-4 text-pink-400"></i><span>Memory</span></div>
        <p class="text-sm font-semibold text-emerald-400">Active</p>
      </div>
      <div class="bg-[#1E293B] rounded-xl p-4 card-hover glow">
        <div class="flex items-center gap-2 text-gray-400 text-xs uppercase tracking-wider mb-2"><i data-lucide="activity" class="w-4 h-4 text-orange-400"></i><span>Cache</span></div>
        <p class="text-lg font-semibold">{cache} Entries</p>
      </div>
      <div class="bg-[#1E293B] rounded-xl p-4 card-hover glow">
        <div class="flex items-center gap-2 text-gray-400 text-xs uppercase tracking-wider mb-2"><i data-lucide="globe" class="w-4 h-4 text-blue-400"></i><span>Gateway</span></div>
        <p class="text-sm font-semibold text-emerald-400">Ready</p>
      </div>
    </div>
    <h2 class="text-lg font-semibold mb-4 text-gray-300">AI Services</h2>
    <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 mb-10">
      <a href="/docs" class="bg-[#1E293B] rounded-xl p-5 card-hover glow">
        <div class="flex items-center gap-3 mb-3"><div class="w-10 h-10 rounded-lg bg-blue-500/10 flex items-center justify-center"><i data-lucide="file-text" class="w-5 h-5 text-blue-400"></i></div><div><h3 class="font-semibold">Resume AI</h3><p class="text-xs text-gray-400">ATS &bull; Parser &bull; Builder</p></div></div>
        <div class="flex items-center gap-1 text-xs text-blue-400"><span>Explore</span><i data-lucide="arrow-right" class="w-3 h-3"></i></div>
      </a>
      <a href="/docs" class="bg-[#1E293B] rounded-xl p-5 card-hover glow">
        <div class="flex items-center gap-3 mb-3"><div class="w-10 h-10 rounded-lg bg-violet-500/10 flex items-center justify-center"><i data-lucide="trending-up" class="w-5 h-5 text-violet-400"></i></div><div><h3 class="font-semibold">Career AI</h3><p class="text-xs text-gray-400">Roadmap &bull; Skill Gap &bull; Assessment</p></div></div>
        <div class="flex items-center gap-1 text-xs text-violet-400"><span>Explore</span><i data-lucide="arrow-right" class="w-3 h-3"></i></div>
      </a>
      <a href="/docs" class="bg-[#1E293B] rounded-xl p-5 card-hover glow">
        <div class="flex items-center gap-3 mb-3"><div class="w-10 h-10 rounded-lg bg-amber-500/10 flex items-center justify-center"><i data-lucide="briefcase" class="w-5 h-5 text-amber-400"></i></div><div><h3 class="font-semibold">Recruiter AI</h3><p class="text-xs text-gray-400">JD Generator &bull; Job Parser</p></div></div>
        <div class="flex items-center gap-1 text-xs text-amber-400"><span>Explore</span><i data-lucide="arrow-right" class="w-3 h-3"></i></div>
      </a>
      <a href="/docs" class="bg-[#1E293B] rounded-xl p-5 card-hover glow">
        <div class="flex items-center gap-3 mb-3"><div class="w-10 h-10 rounded-lg bg-emerald-500/10 flex items-center justify-center"><i data-lucide="sparkles" class="w-5 h-5 text-emerald-400"></i></div><div><h3 class="font-semibold">Job Match AI</h3><p class="text-xs text-gray-400">Smart Matching</p></div></div>
        <div class="flex items-center gap-1 text-xs text-emerald-400"><span>Explore</span><i data-lucide="arrow-right" class="w-3 h-3"></i></div>
      </a>
      <a href="/docs" class="bg-[#1E293B] rounded-xl p-5 card-hover glow">
        <div class="flex items-center gap-3 mb-3"><div class="w-10 h-10 rounded-lg bg-cyan-500/10 flex items-center justify-center"><i data-lucide="mic" class="w-5 h-5 text-cyan-400"></i></div><div><h3 class="font-semibold">Interview AI</h3><p class="text-xs text-gray-400">Mock Interviews</p></div></div>
        <div class="flex items-center gap-1 text-xs text-cyan-400"><span>Explore</span><i data-lucide="arrow-right" class="w-3 h-3"></i></div>
      </a>
      <a href="/docs" class="bg-[#1E293B] rounded-xl p-5 card-hover glow">
        <div class="flex items-center gap-3 mb-3"><div class="w-10 h-10 rounded-lg bg-pink-500/10 flex items-center justify-center"><i data-lucide="message-square" class="w-5 h-5 text-pink-400"></i></div><div><h3 class="font-semibold">AI Chat</h3><p class="text-xs text-gray-400">Conversational AI</p></div></div>
        <div class="flex items-center gap-1 text-xs text-pink-400"><span>Explore</span><i data-lucide="arrow-right" class="w-3 h-3"></i></div>
      </a>
    </div>
    <div class="flex flex-wrap justify-center gap-4 mb-10">
      <a href="/docs" class="inline-flex items-center gap-2 bg-blue-600 hover:bg-blue-700 text-white px-6 py-2.5 rounded-lg font-medium transition-colors text-sm"><i data-lucide="book-open" class="w-4 h-4"></i> Swagger Documentation</a>
      <a href="/health" class="inline-flex items-center gap-2 bg-[#1E293B] hover:bg-slate-700 text-white px-6 py-2.5 rounded-lg font-medium transition-colors text-sm border border-slate-700"><i data-lucide="heart" class="w-4 h-4 text-emerald-400"></i> Health</a>
      <a href="/metrics" class="inline-flex items-center gap-2 bg-[#1E293B] hover:bg-slate-700 text-white px-6 py-2.5 rounded-lg font-medium transition-colors text-sm border border-slate-700"><i data-lucide="bar-chart-3" class="w-4 h-4 text-blue-400"></i> Metrics</a>
      <a href="/version" class="inline-flex items-center gap-2 bg-[#1E293B] hover:bg-slate-700 text-white px-6 py-2.5 rounded-lg font-medium transition-colors text-sm border border-slate-700"><i data-lucide="info" class="w-4 h-4 text-amber-400"></i> Version</a>
    </div>
    <div class="text-center text-gray-500 text-xs">{name} v{version}</div>
  </div>
  <script>lucide.createIcons();</script>
</body>
</html>"""


@router.get("/", response_class=HTMLResponse, include_in_schema=False)
async def dashboard():
    agent_count = 0
    try:
        from app.agents import resume_agent, career_agent, interview_agent
        from app.agents import recruiter_agent, job_match_agent, chat_agent
        agent_count = 6
    except Exception:
        pass
    tool_count = 0
    try:
        from app.tools import base_tool, resume_parser, ats_tool, grammar_tool
        from app.tools import skill_extractor, summary_tool, keyword_tool, pdf_tool, database_tool
        tool_count = 8
    except Exception:
        pass
    return DASHBOARD_HTML.format(
        name=settings.APP_NAME,
        model=settings.OLLAMA_MODEL,
        agents=agent_count,
        services=len(service_registry.list()),
        tools=tool_count,
        docs=knowledge_base.document_count if knowledge_base else 0,
        cache=prompt_cache.size,
        version=settings.APP_VERSION,
    )


SWAGGER_HEADER = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{name} - API Docs</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <link rel="stylesheet" href="https://unpkg.com/swagger-ui-dist@5/swagger-ui.css">
  <style>
    body {{ margin: 0; background: #0F172A; }}
    .topbar-wrapper {{ display: none !important; }}
    .swagger-ui {{ color: #e2e8f0; }}
  </style>
</head>
<body>
  <div style="background: #1E293B; border-bottom: 1px solid #334155; padding: 12px 24px; display: flex; align-items: center; justify-content: space-between;">
    <div style="display: flex; align-items: center; gap: 12px;">
      <div style="width: 32px; height: 32px; background: #3B82F6; border-radius: 8px; display: flex; align-items: center; justify-content: center; font-size: 16px; font-weight: 700; color: white;">Z</div>
      <div><div style="color: #f1f5f9; font-weight: 600; font-size: 15px;">{name}</div><div style="color: #64748b; font-size: 11px;">API Documentation</div></div>
    </div>
    <div style="display: flex; align-items: center; gap: 8px;">
      <a href="/" style="color: #94a3b8; text-decoration: none; font-size: 13px; padding: 6px 12px; border-radius: 6px;">Dashboard</a>
      <a href="/health" style="color: #94a3b8; text-decoration: none; font-size: 13px; padding: 6px 12px; border-radius: 6px;">Health</a>
    </div>
  </div>
  <div id="swagger-ui" style="padding: 20px 24px;"></div>
  <script src="https://unpkg.com/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
  <script>
    const ui = SwaggerUIBundle({{
        url: '/openapi.json',
        dom_id: '#swagger-ui',
        deepLinking: true,
        presets: [SwaggerUIBundle.presets.apis, SwaggerUIBundle.SwaggerUIStandalonePreset],
        layout: "BaseLayout",
        docExpansion: "list",
        filter: true,
        tryItOutEnabled: true,
    }});
  </script>
</body>
</html>"""


@router.get("/docs", response_class=HTMLResponse, include_in_schema=False)
async def swagger_ui():
    return SWAGGER_HEADER.format(name=settings.APP_NAME)
