# Nambikkai AI Intelligence Service

An AI intelligence layer for the Nambikkai analytics dashboard.

---

## 1. What is this service?

This service analyses content performance events (views, likes, velocity) and decides
whether to generate editorial recommendations using an AI language model.

It is designed to be efficient: most events are handled with pure math and never touch
the AI model at all.

---

## 2. What is an AI Agent?

An AI Agent is a control program that follows a loop:

```
Observe → Evaluate → Decide → Use Tools → Invoke Models → Validate → Act → Audit
```

The Agent reads data, makes decisions, calls tools (like an LLM), checks the results,
and records what happened. It is the brain that controls *when* and *how* the LLM is used.

---

## 3. Why is the Agent NOT the LLM?

The LLM (Gemma, Claude, etc.) is just one tool the Agent can call.

The Agent:
- Decides whether the LLM is even needed
- Prepares a controlled, bounded input for the LLM
- Validates the LLM's output before trusting it
- Handles LLM failures gracefully

If you removed the LLM entirely, the Agent would still run — it would just skip the
AI analysis step. This is exactly what happens for NOMINAL events.

---

## 4. What is Deterministic Gating?

Before calling the LLM, the service runs pure math to classify the event:

**Velocity Ratio** = `current_hour_delta_views / seven_day_rolling_hourly_baseline`

**Like Acceleration** = `(one_hour_delta_likes / one_hour_delta_views) × 100`

| Classification  | Condition                              | LLM Called? |
|-----------------|----------------------------------------|-------------|
| NOMINAL         | velocity_ratio < 1.5                   | No          |
| ELEVATED        | 1.5 ≤ velocity_ratio < 3.0             | No          |
| BOOMING_SURGE   | velocity_ratio ≥ 3.0                   | Yes         |

Thresholds are configurable via environment variables.
The LLM is never involved in this decision — it is authoritative and deterministic.

---

## 5. What is LM Studio doing?

LM Studio is a desktop application that runs AI models locally on your machine.
It exposes an OpenAI-compatible HTTP API at `http://localhost:1234/v1`.

This service sends a structured prompt to LM Studio and receives a JSON response.
LM Studio handles all the local GPU/CPU inference.

---

## 6. What is Gemma doing?

Gemma 3 4B Instruct is the language model loaded inside LM Studio.
It reads the evidence package (content title, velocity, signals) and generates
editorial recommendations: title suggestions, hashtags, writer advice, etc.

The model name is configured via `LMSTUDIO_MODEL` — it is never hardcoded.

---

## 7. How is AWS Bedrock used in production?

In production, LM Studio is replaced by AWS Bedrock (managed cloud inference).
The `BedrockProvider` is fully implemented. To activate it:

1. Set `AI_PROVIDER=bedrock` in your environment
2. Set `AWS_REGION`, `BEDROCK_MODEL_ID`, `BEDROCK_MAX_TOKENS`, `BEDROCK_TEMPERATURE`
3. Provide AWS credentials via IAM role or environment variables
4. Ensure the model is enabled in the AWS Bedrock console (Model access)

The Agent orchestrator does not change at all — it only talks to the `LLMProvider` interface.

---

## 8. Why does provider abstraction exist?

```
Agent Orchestrator
      ↓
LLMProvider (abstract interface)
      ├── LMStudioProvider  ← used locally
      └── BedrockProvider   ← used in production
```

The Agent never imports `LMStudioProvider` or `BedrockProvider` directly.
It only calls methods on `LLMProvider`. This means swapping providers requires
zero changes to orchestration logic.

---

## 9. How is token usage controlled?

- NOMINAL events: LLM never called
- ELEVATED events: LLM never called (monitoring only)
- BOOMING_SURGE: LLM called with a bounded `EvidencePackage` — not raw database JSON
- Transcript excerpts are capped at 500 characters
- Structured output format limits response size
- `max_tokens=1024` cap on every request
- Temperature set to 0.3 for focused, consistent output

---

## 10. How to start the service

```bash
# Navigate to the service directory
cd nambikkai-ai-intelligence-service

# Create a local virtual environment
python -m venv venv

# Activate it (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment config
copy .env.example .env

# Start the service
uvicorn app.main:app --host 0.0.0.0 --port 8010 --reload
```

API docs: http://localhost:8010/docs

---

## 11. How to configure LM Studio

1. Download and install LM Studio from https://lmstudio.ai
2. Download the **Gemma 3 4B Instruct** model inside LM Studio
3. Go to the **Local Server** tab in LM Studio
4. Click **Start Server** (default port: 1234)
5. Confirm the model is loaded
6. In your `.env` file set:
   ```
   LMSTUDIO_BASE_URL=http://localhost:1234/v1
   LMSTUDIO_MODEL=gemma-3-4b-it
   ```
   (Check the exact model identifier shown in LM Studio's server tab)

---

## 12. How to run tests

```bash
# From inside the service directory with venv activated
pytest
```

All tests run without a real LLM. LM Studio is mocked in unit tests.

---

## 13. What is intentionally NOT implemented yet

- Real database / analytics API connection (schema not yet confirmed)
- AWS Bedrock production inference (requires AWS credentials and IAM model access)
- Authentication / API keys on endpoints
- Rate limiting
- Dashboard frontend changes
- Email or push notifications
- ML model training (XGBoost, LightGBM, QLoRA)
- Multi-agent architecture
- LangChain / LangGraph
- Production Docker / Kubernetes configuration for this service

---

## Project Structure

```
nambikkai-ai-intelligence-service/
├── app/
│   ├── main.py                    # FastAPI app entry point
│   ├── api/routes/
│   │   ├── health.py              # GET /health, GET /ready
│   │   └── analysis.py            # POST /internal/test-analysis
│   ├── core/
│   │   ├── config.py              # All environment config
│   │   ├── logging.py             # Structured logging
│   │   └── exceptions.py          # Custom exceptions
│   ├── domain/
│   │   └── models.py              # All Pydantic domain models
│   ├── agent/
│   │   ├── orchestrator.py        # Agent control loop
│   │   └── decision_engine.py     # LLM invocation decisions
│   ├── providers/
│   │   ├── base.py                # Abstract LLMProvider interface
│   │   ├── lmstudio.py            # LM Studio implementation
│   │   └── bedrock.py             # Bedrock production provider
│   ├── services/
│   │   ├── gating.py              # Deterministic velocity/acceleration math
│   │   └── evidence_builder.py    # Bounded evidence package assembly
│   ├── validation/
│   │   ├── output_validator.py    # Schema + confidence checks
│   │   └── policy_validator.py    # Language policy enforcement
│   ├── data_sources/
│   │   └── base.py                # Abstract data source boundary
│   └── audit/
│       └── models.py              # AuditRecord re-export
└── tests/
    └── test_service.py            # Full test suite
```
