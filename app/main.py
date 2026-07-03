from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import httpx
from app.config.settings import settings
from app.api.chat import router as chat_router
from app.api.chatbot_v2 import router as chatbot_v2_router
from app.api.resume import router as resume_router
from app.api.career import router as career_router
from app.api.recruiter import router as recruiter_router
from app.api.interview import router as interview_router
from app.api.job import router as job_router
from app.api.knowledge import router as knowledge_router
from app.api.ranking import router as ranking_router
from app.api.monitoring import router as monitoring_router
from app.gateway.service_registry import service_registry
from app.services import resume_service, career_service, interview_service, recruiter_service, job_service, chat_service
from app.middleware.logging_middleware import LoggingMiddleware
from app.middleware.error_handler import register_error_handlers
from app.knowledge.knowledge_base import knowledge_base
from app.memory.memory_manager import memory
from app.memory.cache import prompt_cache
from app.metrics.collector import metrics_collector
from app.utils.logger import logger


def _count_agents() -> int:
    try:
        from app.agents import resume_agent, career_agent, interview_agent
        from app.agents import recruiter_agent, job_match_agent, chat_agent
        return 6
    except Exception:
        return 0


def _count_tools() -> int:
    try:
        from app.tools import base_tool, resume_parser, ats_tool, grammar_tool
        from app.tools import skill_extractor, summary_tool, keyword_tool, pdf_tool, database_tool
        return 8
    except Exception:
        return 0


@asynccontextmanager
async def lifespan(app: FastAPI):
    service_registry.register(
        name="resume_service",
        service=resume_service,
        version="v1",
        description="Resume AI Services",
    )
    service_registry.register(
        name="career_service",
        service=career_service,
        version="v1",
        description="Career AI Services",
    )
    service_registry.register(
        name="interview_service",
        service=interview_service,
        version="v1",
        description="Interview AI Services",
    )
    service_registry.register(
        name="recruiter_service",
        service=recruiter_service,
        version="v1",
        description="Recruiter AI Services",
    )
    service_registry.register(
        name="job_service",
        service=job_service,
        version="v1",
        description="Job Match AI Services",
    )
    service_registry.register(
        name="chat_service",
        service=chat_service,
        version="v1",
        description="Conversational AI Services",
    )

    agent_count = _count_agents()
    tool_count = _count_tools()

    BANNER = f"""
{'='*50}
  {settings.APP_NAME}
{'='*50}
  Version      : {settings.APP_VERSION}
  LLM          : Ollama ({settings.OLLAMA_MODEL})
  Agents       : {agent_count} registered
  Services     : {len(service_registry.list())} registered
  Tools        : {tool_count} loaded
  Knowledge    : {knowledge_base.document_count} documents
  Memory       : Loaded
  Cache        : {prompt_cache.size} entries
  Metrics      : Active
  Gateway      : Ready
{'='*50}
"""
    logger.info(BANNER)
    yield
    logger.info(f"Shutting down {settings.APP_NAME}")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
)

register_error_handlers(app)
app.add_middleware(LoggingMiddleware)

app.include_router(chat_router, prefix="/api/v1/chat", tags=["Chat"])
app.include_router(chatbot_v2_router, prefix="/api/v1/chatbot", tags=["Chatbot V2"])
app.include_router(resume_router, prefix="/api/v1/resume", tags=["Resume"])
app.include_router(career_router, prefix="/api/v1/career", tags=["Career"])
app.include_router(recruiter_router, prefix="/api/v1/recruiter", tags=["Recruiter"])
app.include_router(interview_router, prefix="/api/v1/interview", tags=["Interview"])
app.include_router(job_router, prefix="/api/v1/job", tags=["Job"])
app.include_router(knowledge_router, prefix="/api/v1/knowledge", tags=["Knowledge"])
app.include_router(ranking_router, prefix="/api/v1/ranking", tags=["Ranking"])
app.include_router(monitoring_router, prefix="/api/v1/admin/ai", tags=["AI Monitoring"])


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    agent_count = _count_agents()
    tool_count = _count_tools()
    service_count = len(service_registry.list())
    doc_count = knowledge_base.document_count if knowledge_base else 0

    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{settings.APP_NAME}</title>
  <script src="https://unpkg.com/lucide@latest"></script>
  <script src="https://cdn.tailwindcss.com"></script>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }}
    .card-hover {{ transition: all 0.2s ease; }}
    .card-hover:hover {{ transform: translateY(-2px); box-shadow: 0 8px 25px rgba(59, 130, 246, 0.15); }}
    .glow {{ box-shadow: 0 0 20px rgba(59, 130, 246, 0.1); }}
  </style>
</head>
<body class="bg-[#0F172A] text-white min-h-screen">
  <div class="max-w-5xl mx-auto px-4 py-8">
    <!-- Header -->
    <div class="text-center mb-10">
      <h1 class="text-3xl font-bold tracking-tight">{settings.APP_NAME}</h1>
      <p class="text-gray-400 mt-1 text-sm">Enterprise Agentic AI Platform</p>
    </div>

    <!-- Status Badge -->
    <div class="flex justify-center mb-8">
      <div class="flex items-center gap-2 bg-emerald-500/10 text-emerald-400 px-4 py-1.5 rounded-full text-sm font-medium">
        <span class="w-2 h-2 rounded-full bg-emerald-400 inline-block animate-pulse"></span>
        Running
      </div>
    </div>

    <!-- Stats Grid -->
    <div class="grid grid-cols-2 md:grid-cols-4 gap-4 mb-10">
      <div class="bg-[#1E293B] rounded-xl p-4 card-hover glow">
        <div class="flex items-center gap-2 text-gray-400 text-xs uppercase tracking-wider mb-2">
          <i data-lucide="cpu" class="w-4 h-4 text-blue-400"></i>
          <span>Model</span>
        </div>
        <p class="text-lg font-semibold">{settings.OLLAMA_MODEL}</p>
      </div>
      <div class="bg-[#1E293B] rounded-xl p-4 card-hover glow">
        <div class="flex items-center gap-2 text-gray-400 text-xs uppercase tracking-wider mb-2">
          <i data-lucide="bot" class="w-4 h-4 text-violet-400"></i>
          <span>Agents</span>
        </div>
        <p class="text-lg font-semibold">{agent_count}</p>
      </div>
      <div class="bg-[#1E293B] rounded-xl p-4 card-hover glow">
        <div class="flex items-center gap-2 text-gray-400 text-xs uppercase tracking-wider mb-2">
          <i data-lucide="settings" class="w-4 h-4 text-amber-400"></i>
          <span>Services</span>
        </div>
        <p class="text-lg font-semibold">{service_count}</p>
      </div>
      <div class="bg-[#1E293B] rounded-xl p-4 card-hover glow">
        <div class="flex items-center gap-2 text-gray-400 text-xs uppercase tracking-wider mb-2">
          <i data-lucide="wrench" class="w-4 h-4 text-cyan-400"></i>
          <span>Tools</span>
        </div>
        <p class="text-lg font-semibold">{tool_count}</p>
      </div>
      <div class="bg-[#1E293B] rounded-xl p-4 card-hover glow">
        <div class="flex items-center gap-2 text-gray-400 text-xs uppercase tracking-wider mb-2">
          <i data-lucide="book-open" class="w-4 h-4 text-emerald-400"></i>
          <span>Knowledge</span>
        </div>
        <p class="text-lg font-semibold">{doc_count} Docs</p>
      </div>
      <div class="bg-[#1E293B] rounded-xl p-4 card-hover glow">
        <div class="flex items-center gap-2 text-gray-400 text-xs uppercase tracking-wider mb-2">
          <i data-lucide="hard-drive" class="w-4 h-4 text-pink-400"></i>
          <span>Memory</span>
        </div>
        <p class="text-sm font-semibold text-emerald-400">Active</p>
      </div>
      <div class="bg-[#1E293B] rounded-xl p-4 card-hover glow">
        <div class="flex items-center gap-2 text-gray-400 text-xs uppercase tracking-wider mb-2">
          <i data-lucide="activity" class="w-4 h-4 text-orange-400"></i>
          <span>Cache</span>
        </div>
        <p class="text-lg font-semibold">{prompt_cache.size} Entries</p>
      </div>
      <div class="bg-[#1E293B] rounded-xl p-4 card-hover glow">
        <div class="flex items-center gap-2 text-gray-400 text-xs uppercase tracking-wider mb-2">
          <i data-lucide="globe" class="w-4 h-4 text-blue-400"></i>
          <span>Gateway</span>
        </div>
        <p class="text-sm font-semibold text-emerald-400">Ready</p>
      </div>
    </div>

    <!-- Service Cards -->
    <h2 class="text-lg font-semibold mb-4 text-gray-300">AI Services</h2>
    <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 mb-10">
      <div class="bg-[#1E293B] rounded-xl p-5 card-hover glow cursor-pointer" onclick="window.location.href='/docs'">
        <div class="flex items-center gap-3 mb-3">
          <div class="w-10 h-10 rounded-lg bg-blue-500/10 flex items-center justify-center">
            <i data-lucide="file-text" class="w-5 h-5 text-blue-400"></i>
          </div>
          <div>
            <h3 class="font-semibold">Resume AI</h3>
            <p class="text-xs text-gray-400">ATS • Parser • Builder</p>
          </div>
        </div>
        <div class="flex items-center gap-1 text-xs text-blue-400">
          <span>Explore</span>
          <i data-lucide="arrow-right" class="w-3 h-3"></i>
        </div>
      </div>

      <div class="bg-[#1E293B] rounded-xl p-5 card-hover glow cursor-pointer" onclick="window.location.href='/docs'">
        <div class="flex items-center gap-3 mb-3">
          <div class="w-10 h-10 rounded-lg bg-violet-500/10 flex items-center justify-center">
            <i data-lucide="trending-up" class="w-5 h-5 text-violet-400"></i>
          </div>
          <div>
            <h3 class="font-semibold">Career AI</h3>
            <p class="text-xs text-gray-400">Roadmap • Skill Gap • Assessment</p>
          </div>
        </div>
        <div class="flex items-center gap-1 text-xs text-violet-400">
          <span>Explore</span>
          <i data-lucide="arrow-right" class="w-3 h-3"></i>
        </div>
      </div>

      <div class="bg-[#1E293B] rounded-xl p-5 card-hover glow cursor-pointer" onclick="window.location.href='/docs'">
        <div class="flex items-center gap-3 mb-3">
          <div class="w-10 h-10 rounded-lg bg-amber-500/10 flex items-center justify-center">
            <i data-lucide="briefcase" class="w-5 h-5 text-amber-400"></i>
          </div>
          <div>
            <h3 class="font-semibold">Recruiter AI</h3>
            <p class="text-xs text-gray-400">JD Generator • Job Parser</p>
          </div>
        </div>
        <div class="flex items-center gap-1 text-xs text-amber-400">
          <span>Explore</span>
          <i data-lucide="arrow-right" class="w-3 h-3"></i>
        </div>
      </div>

      <div class="bg-[#1E293B] rounded-xl p-5 card-hover glow cursor-pointer" onclick="window.location.href='/docs'">
        <div class="flex items-center gap-3 mb-3">
          <div class="w-10 h-10 rounded-lg bg-emerald-500/10 flex items-center justify-center">
            <i data-lucide="sparkles" class="w-5 h-5 text-emerald-400"></i>
          </div>
          <div>
            <h3 class="font-semibold">Job Match AI</h3>
            <p class="text-xs text-gray-400">Smart Matching</p>
          </div>
        </div>
        <div class="flex items-center gap-1 text-xs text-emerald-400">
          <span>Explore</span>
          <i data-lucide="arrow-right" class="w-3 h-3"></i>
        </div>
      </div>

      <div class="bg-[#1E293B] rounded-xl p-5 card-hover glow cursor-pointer" onclick="window.location.href='/docs'">
        <div class="flex items-center gap-3 mb-3">
          <div class="w-10 h-10 rounded-lg bg-cyan-500/10 flex items-center justify-center">
            <i data-lucide="mic" class="w-5 h-5 text-cyan-400"></i>
          </div>
          <div>
            <h3 class="font-semibold">Interview AI</h3>
            <p class="text-xs text-gray-400">Mock Interviews</p>
          </div>
        </div>
        <div class="flex items-center gap-1 text-xs text-cyan-400">
          <span>Explore</span>
          <i data-lucide="arrow-right" class="w-3 h-3"></i>
        </div>
      </div>

      <div class="bg-[#1E293B] rounded-xl p-5 card-hover glow cursor-pointer" onclick="window.location.href='/docs'">
        <div class="flex items-center gap-3 mb-3">
          <div class="w-10 h-10 rounded-lg bg-pink-500/10 flex items-center justify-center">
            <i data-lucide="message-square" class="w-5 h-5 text-pink-400"></i>
          </div>
          <div>
            <h3 class="font-semibold">AI Chat</h3>
            <p class="text-xs text-gray-400">Conversational AI</p>
          </div>
        </div>
        <div class="flex items-center gap-1 text-xs text-pink-400">
          <span>Explore</span>
          <i data-lucide="arrow-right" class="w-3 h-3"></i>
        </div>
      </div>
    </div>

    <!-- Action Buttons -->
    <div class="flex flex-wrap justify-center gap-4 mb-10">
      <a href="/docs" class="inline-flex items-center gap-2 bg-blue-600 hover:bg-blue-700 text-white px-6 py-2.5 rounded-lg font-medium transition-colors text-sm">
        <i data-lucide="book-open" class="w-4 h-4"></i>
        Swagger Documentation
      </a>
      <a href="/health" class="inline-flex items-center gap-2 bg-[#1E293B] hover:bg-slate-700 text-white px-6 py-2.5 rounded-lg font-medium transition-colors text-sm border border-slate-700">
        <i data-lucide="heart" class="w-4 h-4 text-emerald-400"></i>
        Health
      </a>
      <a href="/metrics" class="inline-flex items-center gap-2 bg-[#1E293B] hover:bg-slate-700 text-white px-6 py-2.5 rounded-lg font-medium transition-colors text-sm border border-slate-700">
        <i data-lucide="bar-chart-3" class="w-4 h-4 text-blue-400"></i>
        Metrics
      </a>
      <a href="/version" class="inline-flex items-center gap-2 bg-[#1E293B] hover:bg-slate-700 text-white px-6 py-2.5 rounded-lg font-medium transition-colors text-sm border border-slate-700">
        <i data-lucide="info" class="w-4 h-4 text-amber-400"></i>
        Version
      </a>
    </div>

    <!-- Footer -->
    <div class="text-center text-gray-500 text-xs">
      {settings.APP_NAME} v{settings.APP_VERSION}
    </div>
  </div>

  <script>
    lucide.createIcons();
  </script>
</body>
</html>
"""


@app.get("/docs", response_class=HTMLResponse, include_in_schema=False)
async def custom_swagger():
    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{settings.APP_NAME} - API Docs</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <link rel="stylesheet" href="https://unpkg.com/swagger-ui-dist@5/swagger-ui.css">
  <style>
    body {{ margin: 0; background: #0F172A; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }}
    .topbar-wrapper {{ display: none !important; }}
    .swagger-ui {{ color: #e2e8f0; }}
    .swagger-ui .info .title {{ color: #f1f5f9 !important; font-family: inherit !important; }}
    .swagger-ui .info {{ margin: 0 !important; padding: 0 !important; }}
    .swagger-ui .info p, .swagger-ui .info li, .swagger-ui .info table {{ color: #94a3b8 !important; font-family: inherit !important; }}
    .swagger-ui .info a {{ color: #3B82F6 !important; }}
    .swagger-ui .info a:hover {{ color: #60a5fa !important; }}
    .swagger-ui .opblock-tag {{ color: #e2e8f0 !important; font-family: inherit !important; border-bottom: 1px solid #1e293b !important; }}
    .swagger-ui .opblock-tag:hover {{ background: rgba(30, 41, 59, 0.5) !important; }}
    .swagger-ui .opblock {{ border-radius: 8px !important; border: 1px solid #1e293b !important; margin-bottom: 8px !important; box-shadow: none !important; }}
    .swagger-ui .opblock .opblock-summary {{ border-bottom: 1px solid #1e293b !important; }}
    .swagger-ui .opblock .opblock-summary-description {{ color: #94a3b8 !important; font-family: inherit !important; }}
    .swagger-ui .opblock.opblock-get {{ border-color: #3B82F6 !important; background: #1E293B !important; }}
    .swagger-ui .opblock.opblock-post {{ border-color: #22C55E !important; background: #1E293B !important; }}
    .swagger-ui .opblock.opblock-put {{ border-color: #F59E0B !important; background: #1E293B !important; }}
    .swagger-ui .opblock.opblock-delete {{ border-color: #EF4444 !important; background: #1E293B !important; }}
    .swagger-ui .opblock.opblock-patch {{ border-color: #A855F7 !important; background: #1E293B !important; }}
    .swagger-ui .opblock .opblock-summary-method {{ border-radius: 4px !important; font-size: 12px !important; font-weight: 600 !important; font-family: inherit !important; min-width: 60px !important; padding: 4px 10px !important; text-shadow: none !important; }}
    .swagger-ui .opblock .opblock-summary-path {{ color: #e2e8f0 !important; font-family: monospace !important; font-size: 14px !important; }}
    .swagger-ui .opblock .opblock-summary-path a {{ color: #e2e8f0 !important; }}
    .swagger-ui .opblock .opblock-section-header {{ background: #0F172A !important; border-bottom: 1px solid #1e293b !important; }}
    .swagger-ui .opblock .opblock-section-header h4 {{ color: #e2e8f0 !important; font-family: inherit !important; }}
    .swagger-ui .opblock .opblock-section-header label {{ color: #94a3b8 !important; font-family: inherit !important; }}
    .swagger-ui .opblock .opblock-body .opblock-description-wrapper p {{ color: #94a3b8 !important; }}
    .swagger-ui .opblock-body .opblock-description-wrapper, .swagger-ui .opblock-body .opblock-external-docs-wrapper, .swagger-ui .opblock-body .opblock-title {{ color: #e2e8f0 !important; font-family: inherit !important; }}
    .swagger-ui .opblock-body select {{ background: #0F172A !important; color: #e2e8f0 !important; border-color: #334155 !important; border-radius: 6px !important; }}
    .swagger-ui .opblock-body input[type=text] {{ background: #0F172A !important; color: #e2e8f0 !important; border-color: #334155 !important; border-radius: 6px !important; }}
    .swagger-ui .opblock-body textarea {{ background: #0F172A !important; color: #e2e8f0 !important; border-color: #334155 !important; border-radius: 6px !important; }}
    .swagger-ui .opblock-body .parameter__name {{ color: #e2e8f0 !important; font-family: inherit !important; }}
    .swagger-ui .opblock-body .parameter__type {{ color: #94a3b8 !important; font-family: inherit !important; }}
    .swagger-ui .opblock-body .parameter__in {{ color: #64748b !important; font-family: inherit !important; font-size: 11px !important; }}
    .swagger-ui .opblock-body .parameters-col_description .markdown p {{ color: #94a3b8 !important; }}
    .swagger-ui .opblock-body .responses-inner h4, .swagger-ui .opblock-body .responses-inner h5 {{ color: #e2e8f0 !important; }}
    .swagger-ui .opblock-body .response-col_status {{ color: #e2e8f0 !important; }}
    .swagger-ui .opblock-body .response-col_description .markdown p {{ color: #94a3b8 !important; }}
    .swagger-ui .opblock-body .response-col_links {{ color: #94a3b8 !important; }}
    .swagger-ui .btn {{ border-radius: 6px !important; border: 1px solid #334155 !important; background: #1E293B !important; color: #e2e8f0 !important; font-family: inherit !important; font-size: 13px !important; }}
    .swagger-ui .btn:hover {{ background: #334155 !important; }}
    .swagger-ui .btn.cancel {{ background: #1E293B !important; color: #e2e8f0 !important; }}
    .swagger-ui .btn.execute {{ background: #3B82F6 !important; border-color: #3B82F6 !important; color: white !important; }}
    .swagger-ui .btn.execute:hover {{ background: #2563EB !important; }}
    .swagger-ui .opblock-body .btn-group .btn_curl {{ background: #1E293B !important; border-color: #334155 !important; color: #e2e8f0 !important; }}
    .swagger-ui .opblock-body .btn-group .btn_curl:hover {{ background: #334155 !important; }}
    .swagger-ui .opblock-body .curl-command pre {{ background: #0F172A !important; color: #22C55E !important; border-radius: 6px !important; border: 1px solid #334155 !important; }}
    .swagger-ui .opblock-body .curl-command .copy-to-clipboard {{ background: #1E293B !important; border-left: 1px solid #334155 !important; }}
    .swagger-ui .opblock-body .curl-command .copy-to-clipboard button {{ color: #94a3b8 !important; }}
    .swagger-ui .model-box {{ background: #0F172A !important; border-radius: 6px !important; border: 1px solid #334155 !important; }}
    .swagger-ui .model-box .model {{ color: #e2e8f0 !important; font-family: monospace !important; }}
    .swagger-ui .model-box .model .property {{ color: #60a5fa !important; }}
    .swagger-ui .model-box .model .property .prop-type {{ color: #22C55E !important; }}
    .swagger-ui .model-box .model .bracejs-simple-object {{ color: #e2e8f0 !important; }}
    .swagger-ui .model-box .model .bracejs-object {{ color: #e2e8f0 !important; }}
    .swagger-ui .model-box .model-title {{ color: #e2e8f0 !important; font-family: inherit !important; }}
    .swagger-ui .model-box .model-title span {{ font-family: inherit !important; }}
    .swagger-ui .model-toggle {{ color: #64748b !important; }}
    .swagger-ui .model-toggle::after {{ background: #64748b !important; }}
    .swagger-ui table thead tr td, .swagger-ui table thead tr th {{ color: #94a3b8 !important; font-family: inherit !important; border-bottom: 1px solid #334155 !important; }}
    .swagger-ui table tbody tr td {{ color: #e2e8f0 !important; font-family: inherit !important; }}
    .swagger-ui .loading-container {{ color: #94a3b8 !important; }}
    .swagger-ui .loading-container .loading {{ border: 2px solid #334155 !important; border-top-color: #3B82F6 !important; }}
    .swagger-ui .errors {{ color: #EF4444 !important; }}
    .swagger-ui section.models {{ border: 1px solid #1e293b !important; border-radius: 8px !important; }}
    .swagger-ui section.models.is-open h4 {{ border-bottom: 1px solid #1e293b !important; }}
    .swagger-ui section.models h4 {{ color: #e2e8f0 !important; font-family: inherit !important; }}
    .swagger-ui section.models h4 span {{ font-family: inherit !important; }}
    .swagger-ui section.models .model-container {{ background: #1E293B !important; border-radius: 6px !important; margin: 4px 0 !important; }}
    .swagger-ui section.models .model-container:hover {{ background: #1E293B !important; }}
    .swagger-ui select {{ background: #0F172A !important; color: #e2e8f0 !important; border-color: #334155 !important; border-radius: 6px !important; }}
    .swagger-ui label {{ color: #94a3b8 !important; font-family: inherit !important; }}
    .swagger-ui .response-control-media-type--accept-container select {{ background: #0F172A !important; color: #e2e8f0 !important; }}
    .swagger-ui .responses-inner .responses-table td.response-col_description .response-col_description__inner div.markdown p {{ color: #94a3b8 !important; }}
    .swagger-ui .responses-inner .responses-table td.response-col_description .response-col_description__inner div.markdown p a {{ color: #3B82F6 !important; }}
    .swagger-ui .dialog-ux .backdrop-ux {{ background: rgba(0,0,0,0.7) !important; }}
    .swagger-ui .dialog-ux .modal-ux {{ background: #1E293B !important; border: 1px solid #334155 !important; border-radius: 12px !important; }}
    .swagger-ui .dialog-ux .modal-ux-content h4 {{ color: #e2e8f0 !important; }}
    .swagger-ui .dialog-ux .modal-ux-content p {{ color: #94a3b8 !important; }}
    .swagger-ui .dialog-ux .modal-ux-header h3 {{ color: #e2e8f0 !important; }}
    .swagger-ui .dialog-ux .modal-ux-header .close-modal {{ color: #64748b !important; }}
    .swagger-ui .dialog-ux .modal-ux-header .close-modal:hover {{ color: #e2e8f0 !important; }}
    .swagger-ui .auth-wrapper .authorize {{ border-color: #3B82F6 !important; color: #3B82F6 !important; }}
    .swagger-ui .auth-wrapper .authorize svg {{ fill: #3B82F6 !important; }}
    .swagger-ui .auth-btn-wrapper .btn-done {{ border-color: #22C55E !important; color: #22C55E !important; }}
    .swagger-ui .auth-btn-wrapper .btn-done svg {{ fill: #22C55E !important; }}
    .swagger-ui .wrapper {{ padding: 0 !important; max-width: 100% !important; }}
    .swagger-ui .opblock-body .highlight-code {{ background: #0F172A !important; }}
    .swagger-ui .opblock-body .highlight-code .lang-curl {{ color: #22C55E !important; }}
    .swagger-ui .opblock-body .highlight-code .lang-curl .string {{ color: #22C55E !important; }}
    .swagger-ui .opblock-body .highlight-code .lang-curl .punctuation {{ color: #94a3b8 !important; }}
    .swagger-ui .opblock-body .highlight-code .lang-curl .function {{ color: #60a5fa !important; }}
    .swagger-ui .opblock-body .highlight-code .lang-curl .keyword {{ color: #A855F7 !important; }}
    .swagger-ui .opblock-body .highlight-code .lang-curl .number {{ color: #F59E0B !important; }}
    .swagger-ui .scheme-container {{ background: #1E293B !important; border-radius: 8px !important; margin: 12px 0 !important; padding: 12px 20px !important; box-shadow: none !important; }}
    .swagger-ui .scheme-container .schemes-title {{ color: #94a3b8 !important; font-family: inherit !important; }}
    .swagger-ui .scheme-container .schemes-server-container label {{ color: #94a3b8 !important; }}
    .swagger-ui .scheme-container .schemes-server-container select {{ background: #0F172A !important; color: #e2e8f0 !important; border-color: #334155 !important; }}
    .swagger-ui .info .description p {{ color: #94a3b8 !important; }}
    .swagger-ui .info .description a {{ color: #3B82F6 !important; }}
    .swagger-ui .info .description a:hover {{ color: #60a5fa !important; }}
    .swagger-ui .info .info__extensible-details {{ color: #94a3b8 !important; }}
    .swagger-ui .info .info__extensible-details table td {{ color: #94a3b8 !important; }}
    .swagger-ui .info .info__extensible-details table th {{ color: #e2e8f0 !important; }}
    ::-webkit-scrollbar {{ width: 8px; height: 8px; }}
    ::-webkit-scrollbar-track {{ background: #0F172A; }}
    ::-webkit-scrollbar-thumb {{ background: #334155; border-radius: 4px; }}
    ::-webkit-scrollbar-thumb:hover {{ background: #475569; }}
  </style>
</head>
<body>
  <!-- Custom Header Bar -->
  <div style="background: #1E293B; border-bottom: 1px solid #334155; padding: 12px 24px; display: flex; align-items: center; justify-content: space-between;">
    <div style="display: flex; align-items: center; gap: 12px;">
      <div style="width: 32px; height: 32px; background: #3B82F6; border-radius: 8px; display: flex; align-items: center; justify-content: center; font-size: 16px; font-weight: 700; color: white;">Z</div>
      <div>
        <div style="color: #f1f5f9; font-weight: 600; font-size: 15px;">{settings.APP_NAME}</div>
        <div style="color: #64748b; font-size: 11px;">API Documentation</div>
      </div>
    </div>
    <div style="display: flex; align-items: center; gap: 8px;">
      <a href="/" style="color: #94a3b8; text-decoration: none; font-size: 13px; padding: 6px 12px; border-radius: 6px; transition: all 0.2s;" onmouseover="this.style.background='#334155'" onmouseout="this.style.background='transparent'">Dashboard</a>
      <a href="/health" style="color: #94a3b8; text-decoration: none; font-size: 13px; padding: 6px 12px; border-radius: 6px; transition: all 0.2s;" onmouseover="this.style.background='#334155'" onmouseout="this.style.background='transparent'">Health</a>
      <a href="/metrics" style="color: #94a3b8; text-decoration: none; font-size: 13px; padding: 6px 12px; border-radius: 6px; transition: all 0.2s;" onmouseover="this.style.background='#334155'" onmouseout="this.style.background='transparent'">Metrics</a>
      <a href="/version" style="color: #94a3b8; text-decoration: none; font-size: 13px; padding: 6px 12px; border-radius: 6px; transition: all 0.2s;" onmouseover="this.style.background='#334155'" onmouseout="this.style.background='transparent'">Version</a>
    </div>
  </div>

  <div id="swagger-ui" style="padding: 20px 24px;"></div>

  <script src="https://unpkg.com/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
  <script>
    const ui = SwaggerUIBundle({{
        url: '/openapi.json',
        dom_id: '#swagger-ui',
        deepLinking: true,
        presets: [
            SwaggerUIBundle.presets.apis,
            SwaggerUIBundle.SwaggerUIStandalonePreset,
        ],
        layout: "BaseLayout",
        defaultModelsExpandDepth: 1,
        defaultModelExpandDepth: 1,
        docExpansion: "list",
        filter: true,
        showExtensions: true,
        showCommonExtensions: true,
        tryItOutEnabled: true,
        syntaxHighlight: {{
            activated: true,
            theme: "nord"
        }},
    }});
  </script>
</body>
</html>
"""


@app.get("/health")
async def health():
    ollama_status = "unknown"
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{settings.OLLAMA_BASE_URL}/api/tags", timeout=3)
            ollama_status = "connected" if resp.status_code == 200 else "error"
    except Exception:
        ollama_status = "unreachable"

    return {
        "status": "healthy",
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "ollama": ollama_status,
        "model": settings.OLLAMA_MODEL,
        "memory": "ok" if memory else "error",
        "knowledge": f"{knowledge_base.document_count} docs loaded" if knowledge_base.document_count else "empty",
        "services": len(service_registry.list()),
        "agents": _count_agents(),
        "tools": _count_tools(),
        "cache": prompt_cache.size,
    }


@app.get("/version")
def version():
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "llm": "ollama",
        "model": settings.OLLAMA_MODEL,
        "services": service_registry.list_with_info(),
    }


@app.get("/metrics")
def metrics():
    return metrics_collector.summary()
