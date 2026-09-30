"""Quick probe of the configured LLM judge endpoint: one real call, print raw response
and the utility LLMJudgeUtility would parse from it. Run before any full experiment.
Needs OPENAI_API_KEY (DeepSeek) in the environment."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
ENGA_ROOT = ROOT if (ROOT / "enga").is_dir() else ROOT.parent / "experiments"
sys.path.insert(0, str(ENGA_ROOT))

from enga.config import Config
from enga.data import Query
from enga.evaluator import EvalBudget, LLMJudgeUtility

q = Query(qid="probe_001", text="Find the release date and rating of the TV show 'Severance' and generate a QR code linking to its page.", gold_ids=[])

cfg = Config()
print(f"endpoint : {cfg.llm.base_url}")
print(f"model    : {cfg.llm.model}")
print(f"api_key  : {cfg.llm.api_key[:8]}...{cfg.llm.api_key[-4:]}")

judge = LLMJudgeUtility.__new__(LLMJudgeUtility)  # bypass cache loading for the raw test
judge.cfg, judge.tools, judge.budget = cfg, {}, EvalBudget()
judge.cache, judge.cache_path, judge._client, judge._lock = {}, Path("results/_probe_cache.jsonl"), None, None
judge._tok_usage = {"prompt": 0, "completion": 0}

prompt = judge._render(q, ["tvshows_search", "qrcode_generate", "weather_lookup"])
client = judge._client_()
resp = client.chat.completions.create(
    model=cfg.llm.model,
    messages=[{"role": "user", "content": prompt}],
    temperature=cfg.llm.temperature,
    max_tokens=cfg.llm.max_tokens,
    timeout=cfg.llm.timeout,
)
raw = resp.choices[0].message.content or ""
print("\n--- finish_reason:", resp.choices[0].finish_reason)
print("--- raw response (first 800 chars) ---")
print(raw[:800])

u = judge._strip_think(raw)
import json as _json, re
m = re.search(r"\{.*?\}", u, re.S)
parsed = _json.loads(m.group(0)) if m else None
print("\n--- parsed JSON ---")
print(parsed)
if parsed and "utility" in parsed:
    print("OK: utility =", parsed["utility"])
elif parsed:
    comp, rel, red = float(parsed.get("completeness", .5)), float(parsed.get("relevance", .5)), float(parsed.get("redundancy_penalty", 0))
    print("OK: fallback formula utility =", max(0., min(1., 0.7 * comp + 0.3 * rel - 0.2 * red)))
else:
    print("PARSE FAILED — judge output not understood; fix prompt/model before running!")
