"""
Sequential live test — dev only, deleted after use.
Tests: direct call, single endpoint call, 10s wait, repeat, then 3x sequential.
"""
import urllib.request, json, time, sys
sys.path.insert(0, '.')
from app.core.config import get_settings

s = get_settings()
BASE = "http://127.0.0.1:8010"
LMS  = s.LMSTUDIO_BASE_URL

SURGE_PAYLOAD = json.dumps({
    "event_id": "seq-test",
    "content_id": "c-seq",
    "title": "Sequential Test Video",
    "creator_id": "u-seq",
    "platform": "youtube",
    "current_hour_delta_views": 400.0,
    "seven_day_rolling_hourly_baseline": 100.0,
    "one_hour_delta_likes": 10.0,
    "one_hour_delta_views": 400.0,
}).encode()

NOMINAL_PAYLOAD = json.dumps({
    "event_id": "nom-test",
    "content_id": "c-nom",
    "title": "Nominal Test",
    "creator_id": "u-nom",
    "platform": "youtube",
    "current_hour_delta_views": 100.0,
    "seven_day_rolling_hourly_baseline": 100.0,
    "one_hour_delta_likes": 1.0,
    "one_hour_delta_views": 100.0,
}).encode()


def post(url, data, timeout=300):
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    t = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            ms = int((time.monotonic() - t) * 1000)
            body = json.loads(r.read().decode())
            return r.status, ms, body, None
    except urllib.error.HTTPError as e:
        ms = int((time.monotonic() - t) * 1000)
        return e.code, ms, None, e.read().decode()[:300]
    except Exception as e:
        ms = int((time.monotonic() - t) * 1000)
        return None, ms, None, str(e)


def get(url, timeout=10):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode())
    except Exception as e:
        return None, str(e)


def report(label, status, ms, body, err):
    if err:
        print(f"  [{label}] status={status} ms={ms} ERROR={err}")
        return
    analysis = body.get("analysis", {}) if body else {}
    audit    = body.get("audit", {}) if body else {}
    print(f"  [{label}] status={status} ms={ms} "
          f"analysis_status={analysis.get('status')} "
          f"llm_invoked={audit.get('llm_invoked')} "
          f"validation={audit.get('validation_passed')}")


print("=" * 60)
print("TEST 1 — Direct LM Studio /v1/models reachability")
status, body = get(f"{LMS}/models")
print(f"  status={status} models={[m['id'] for m in body.get('data',[])] if isinstance(body,dict) else body}")

print()
print("TEST 2 — NOMINAL event (LLM must NOT be called)")
status, ms, body, err = post(f"{BASE}/internal/test-analysis", NOMINAL_PAYLOAD)
report("NOMINAL", status, ms, body, err)

print()
print("TEST 3 — BOOMING_SURGE #1")
status, ms, body, err = post(f"{BASE}/internal/test-analysis", SURGE_PAYLOAD)
report("SURGE-1", status, ms, body, err)

print()
print("TEST 4 — Waiting 10 seconds...")
time.sleep(10)

print()
print("TEST 5 — BOOMING_SURGE #2 (after 10s gap)")
status, ms, body, err = post(f"{BASE}/internal/test-analysis", SURGE_PAYLOAD)
report("SURGE-2", status, ms, body, err)

print()
print("TEST 6 — BOOMING_SURGE #3 (sequential, no gap)")
status, ms, body, err = post(f"{BASE}/internal/test-analysis", SURGE_PAYLOAD)
report("SURGE-3", status, ms, body, err)

print()
print("TEST 7 — BOOMING_SURGE #4 (sequential, no gap)")
status, ms, body, err = post(f"{BASE}/internal/test-analysis", SURGE_PAYLOAD)
report("SURGE-4", status, ms, body, err)

print()
print("TEST 8 — /health and /ready")
status, body = get(f"{BASE}/health")
print(f"  /health  status={status} body={body}")
status, body = get(f"{BASE}/ready")
print(f"  /ready   status={status} body={body}")
print("=" * 60)
