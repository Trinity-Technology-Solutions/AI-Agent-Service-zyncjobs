"""PDF generation service — resumes, cover letters, reports."""
import io
import logging
from typing import Optional
from recruitment_ai.config.settings import settings

logger = logging.getLogger(__name__)

try:
    from weasyprint import HTML
    WEASYPRINT_AVAILABLE = True
except ImportError:
    WEASYPRINT_AVAILABLE = False
    logger.info("weasyprint not installed — PDF generation disabled")


_RESUME_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  @page {{ margin: 0.75in; size: A4; }}
  body {{ font-family: 'Helvetica Neue', Arial, sans-serif; font-size: 11pt; color: #333; line-height: 1.5; }}
  h1 {{ font-size: 22pt; margin-bottom: 4px; color: #1a1a2e; }}
  h2 {{ font-size: 14pt; border-bottom: 2px solid #1a1a2e; padding-bottom: 3px; margin-top: 18px; color: #1a1a2e; }}
  .contact {{ font-size: 10pt; color: #666; margin-bottom: 12px; }}
  .section {{ margin-bottom: 14px; }}
  .entry {{ margin-bottom: 8px; }}
  .entry-title {{ font-weight: bold; font-size: 11.5pt; }}
  .entry-sub {{ color: #555; font-size: 10.5pt; }}
  .entry-date {{ float: right; color: #888; font-size: 10pt; }}
  ul {{ margin: 4px 0; padding-left: 18px; }}
  li {{ margin-bottom: 2px; }}
  .skill-tag {{ display: inline-block; background: #eef; padding: 2px 8px; border-radius: 3px; font-size: 10pt; margin: 1px; }}
</style>
</head>
<body>
  <h1>{{ name }}</h1>
  <div class="contact">{{ email }} | {{ phone }} | {{ location }} | {{ linkedin }}</div>
  {% if summary %}
  <div class="section">
    <h2>Summary</h2>
    <p>{{ summary }}</p>
  </div>
  {% endif %}
  {% if experience %}
  <div class="section">
    <h2>Experience</h2>
    {% for exp in experience %}
    <div class="entry">
      <div class="entry-title">{{ exp.title }} <span class="entry-date">{{ exp.dates }}</span></div>
      <div class="entry-sub">{{ exp.company }} — {{ exp.location }}</div>
      <ul>
        {% for bullet in exp.bullets %}
        <li>{{ bullet }}</li>
        {% endfor %}
      </ul>
    </div>
    {% endfor %}
  </div>
  {% endif %}
  {% if education %}
  <div class="section">
    <h2>Education</h2>
    {% for edu in education %}
    <div class="entry">
      <div class="entry-title">{{ edu.degree }} <span class="entry-date">{{ edu.year }}</span></div>
      <div class="entry-sub">{{ edu.institution }} — {{ edu.gpa }}</div>
    </div>
    {% endfor %}
  </div>
  {% endif %}
  {% if skills %}
  <div class="section">
    <h2>Skills</h2>
    <p>{% for s in skills %}<span class="skill-tag">{{ s }}</span> {% endfor %}</p>
  </div>
  {% endif %}
  {% if certifications %}
  <div class="section">
    <h2>Certifications</h2>
    <ul>
      {% for c in certifications %}
      <li>{{ c.name }} ({{ c.issuer }}, {{ c.year }})</li>
      {% endfor %}
    </ul>
  </div>
  {% endif %}
</body>
</html>"""

_COVER_LETTER_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  @page {{ margin: 1in; size: A4; }}
  body {{ font-family: 'Helvetica Neue', Arial, sans-serif; font-size: 11pt; color: #333; line-height: 1.6; }}
  .header {{ margin-bottom: 24px; }}
  .greeting {{ margin-bottom: 16px; }}
  .closing {{ margin-top: 24px; }}
  .signature {{ margin-top: 32px; font-size: 10pt; color: #555; }}
</style>
</head>
<body>
  <div class="header">
    <strong>{{ name }}</strong><br>
    {{ email }} | {{ phone }}<br>
    {{ date }}
  </div>
  <div class="header">
    {{ hiring_manager_name }}<br>
    {{ company_name }}<br>
    {{ company_address }}
  </div>
  <div class="greeting">
    Dear {{ hiring_manager_name }},
  </div>
  <div>
    {{ body }}
  </div>
  <div class="closing">
    <p>Sincerely,</p>
    <p><strong>{{ name }}</strong></p>
  </div>
</body>
</html>"""


class PDFGenerationError(Exception):
    pass


def _render_html(template: str, data: dict) -> str:
    import jinja2
    env = jinja2.Environment(autoescape=True)
    tpl = env.from_string(template)
    return tpl.render(**data)


async def generate_resume_pdf(data: dict) -> bytes:
    if not WEASYPRINT_AVAILABLE:
        raise PDFGenerationError("PDF generation requires weasyprint: pip install weasyprint")

    html = _render_html(_RESUME_TEMPLATE, data)
    pdf_bytes = HTML(string=html).write_pdf()
    return pdf_bytes


async def generate_cover_letter_pdf(data: dict) -> bytes:
    if not WEASYPRINT_AVAILABLE:
        raise PDFGenerationError("PDF generation requires weasyprint: pip install weasyprint")

    html = _render_html(_COVER_LETTER_TEMPLATE, data)
    pdf_bytes = HTML(string=html).write_pdf()
    return pdf_bytes
