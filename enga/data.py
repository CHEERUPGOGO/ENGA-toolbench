"""Data loading: ToolRet (parquet mirrors), raw ToolBench instruction JSONs,
and a deterministic synthetic fallback.

Unified in-memory representation
--------------------------------
ToolDoc  : one tool/API with name, description, params, out fields.
Query    : one user query with gold tool ids (the annotation used by the
           oracle utility U(q,S) and by Coverage@K).
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import pandas as pd

from .utils import seed_everything, stable_hash


# --------------------------------------------------------------------- schemas
@dataclass
class Param:
    name: str
    type: str = "string"
    description: str = ""


@dataclass
class ToolDoc:
    id: str
    name: str
    description: str
    category: str = ""
    method: str = ""
    required_params: list[Param] = field(default_factory=list)
    optional_params: list[Param] = field(default_factory=list)
    out_fields: list[Param] = field(default_factory=list)  # from template_response / outputParams
    library: str = ""

    def doc_text(self) -> str:
        req = ", ".join(p.name for p in self.required_params)
        opt = ", ".join(p.name for p in self.optional_params)
        parts = [self.name, self.category, self.method, self.description, f"required: {req}", f"optional: {opt}"]
        return " | ".join(x for x in parts if x)


@dataclass
class Query:
    qid: str
    text: str
    gold_ids: list[str]


# ------------------------------------------------------------------ ToolRet IO
_TYPE_MAP = {
    "NUMBER": "number", "number": "number", "float": "number", "double": "number",
    "INTEGER": "integer", "integer": "number", "int": "number",
    "STRING": "string", "string": "string", "str": "string", "text": "string",
    "BOOLEAN": "boolean", "boolean": "boolean", "bool": "boolean", "array": "array",
    "object": "object", "list": "array", "any": "string",
}


def norm_type(t: str | None) -> str:
    if not t:
        return "string"
    return _TYPE_MAP.get(str(t).strip().upper(), "string")


def _parse_toolbench_style(doc: dict, tid: str, library: str) -> ToolDoc:
    """ToolBench-family schema: category_name/required_parameters/optional_parameters/..."""
    rp = [Param(p.get("name", ""), norm_type(p.get("type")), p.get("description", ""))
          for p in doc.get("required_parameters") or []]
    op = [Param(p.get("name", ""), norm_type(p.get("type")), p.get("description", ""))
          for p in doc.get("optional_parameters") or []]
    tr = doc.get("template_response")
    if isinstance(tr, dict):
        out = [Param(k, norm_type(v), "") for k, v in tr.items()]
    else:
        out = []
    return ToolDoc(
        id=tid,
        name=doc.get("name") or doc.get("api_name") or tid,
        description=doc.get("description") or doc.get("api_description") or "",
        category=doc.get("category_name") or "",
        method=doc.get("method", ""),
        required_params=rp, optional_params=op, out_fields=out, library=library,
    )


def _parse_jsonschema_style(doc: dict, tid: str, library: str) -> ToolDoc:
    """OpenAI-function/JSON-schema family: {name, description, doc_arguments:{properties, required}}."""
    args = doc.get("doc_arguments") or doc.get("parameters") or {}
    if isinstance(args, list):  # some schemas store a flat [Param, ...] list
        rp, op = [], []
        for p in args:
            if isinstance(p, dict) and p.get("name"):
                par = Param(p["name"], norm_type(p.get("type")), p.get("description", ""))
                (rp if p.get("required") else op).append(par)
        return ToolDoc(id=tid, name=doc.get("name") or tid,
                       description=doc.get("description") or "",
                       category=doc.get("category_name") or doc.get("category") or "",
                       method=doc.get("method", ""), required_params=rp,
                       optional_params=op, library=library)
    props = args.get("properties") or {}
    req_names = set(args.get("required") or [])
    rp, op = [], []
    for name, spec in props.items():
        if isinstance(spec, str):
            p = Param(name, norm_type(spec), "")
        elif isinstance(spec, dict):
            p = Param(name, norm_type(spec.get("type")), spec.get("description", ""))
        else:
            p = Param(name, "string", "")
        (rp if name in req_names else op).append(p)
    return ToolDoc(
        id=tid,
        name=doc.get("name") or tid,
        description=doc.get("description") or "",
        category=doc.get("category_name") or doc.get("category") or "",
        method=doc.get("method", ""),
        required_params=rp, optional_params=op, library=library,
    )


def _parse_tool_doc(raw: str | dict, tid: str, library: str) -> ToolDoc:
    """Best-effort schema parsing; never raises -- messy real-world variants
    degrade to a minimal ToolDoc (name/description only)."""
    try:
        doc = json.loads(raw) if isinstance(raw, str) else raw
        if not isinstance(doc, dict):
            raise TypeError(f"doc is {type(doc).__name__}")
        if "required_parameters" in doc or "template_response" in doc:
            return _parse_toolbench_style(doc, tid, library)
        return _parse_jsonschema_style(doc, tid, library)
    except Exception:
        name = tid
        desc = ""
        if isinstance(raw, str):
            desc = raw[:300]
        try:
            d = json.loads(raw) if isinstance(raw, str) else (raw or {})
            name = (d or {}).get("name") or (d or {}).get("api_name") or tid
            desc = (d or {}).get("description") or desc
        except Exception:
            pass
        return ToolDoc(id=tid, name=name, description=desc, library=library)


def load_toolret(data_dir: str | Path, source: str = "toolbench",
                 library: str = "web") -> tuple[dict[str, ToolDoc], list[Query]]:
    """Load ToolRet parquet mirrors: {source}/queries-*.parquet + {library}/tools-*.parquet."""
    data_dir = Path(data_dir)
    qfile = sorted((data_dir / source).glob("queries-*.parquet"))
    tfile = sorted((data_dir / library).glob("tools-*.parquet"))
    if not qfile or not tfile:
        raise FileNotFoundError(
            f"ToolRet parquets not found under {data_dir} "
            f"(need {source}/queries-*.parquet and {library}/tools-*.parquet). "
            "Run scripts/download_data.py first.")
    qdf = pd.read_parquet(qfile[0])
    tdf = pd.read_parquet(tfile[0])

    tools = {r["id"]: _parse_tool_doc(r["documentation"], r["id"], library)
             for _, r in tdf.iterrows()}

    queries = []
    for _, r in qdf.iterrows():
        labels = json.loads(r["labels"]) if isinstance(r["labels"], str) else r["labels"]
        gold = [l["id"] for l in labels if l.get("relevance", 1) > 0]
        gold = [g for g in gold if g in tools]
        if not gold:
            continue
        queries.append(Query(qid=str(r["id"]), text=str(r["query"]), gold_ids=sorted(set(gold))))
    return tools, queries


# ------------------------------------------------------------- raw ToolBench IO
def load_toolbench_raw(data_dir: str | Path, split: str = "G1_test") -> tuple[dict[str, ToolDoc], list[Query]]:
    """Load raw ToolBench test_instruction jsons.

    Expects {data_dir}/{split}.json (list of {query, api_list}) and optionally
    {data_dir}/toolenv/** schema jsons. api_list entries carry the schema inline,
    which is enough to build both the gold set and the tool docs.
    """
    data_dir = Path(data_dir)
    with open(data_dir / f"{split}.json", encoding="utf-8") as f:
        raw = json.load(f)
    tools: dict[str, ToolDoc] = {}
    queries: list[Query] = []
    for i, item in enumerate(raw):
        gold, docs = [], []
        for api in item.get("api_list", []):
            tid = f"{api.get('category_name','')}/{api.get('tool_name','')}/{api.get('api_name','')}"
            doc = dict(api)
            doc.setdefault("name", f"{api.get('tool_name','')}_{api.get('api_name','')}")
            td = _parse_toolbench_style(doc, tid, "toolenv")
            docs.append(td)
            gold.append(tid)
        text = item.get("query") or (item.get("instruction") or [""])[0]
        queries.append(Query(qid=str(item.get("query_id", i)), text=text, gold_ids=gold))
        for td in docs:
            tools.setdefault(td.id, td)
    return tools, queries


# ------------------------------------------------------------------- synthetic
_TOOL_THEMES = [
    ("flight", ["search_flights", "book_flight", "get_ticket_price"],
     [("origin", "string"), ("destination", "string"), ("date", "string")], [("itinerary_id", "string")]),
    ("hotel", ["search_hotels", "book_hotel", "hotel_reviews"],
     [("city", "string"), ("checkin", "string"), ("nights", "number")], [("hotel_id", "string")]),
    ("weather", ["current_weather", "weather_forecast", "air_quality"],
     [("location", "string"), ("days", "number")], [("report", "string")]),
    ("currency", ["exchange_rate", "convert_currency"],
     [("base", "string"), ("target", "string"), ("amount", "number")], [("rate", "number")]),
    ("email", ["send_email", "search_inbox", "draft_reply"],
     [("to", "string"), ("subject", "string"), ("body", "string")], [("message_id", "string")]),
    ("calendar", ["create_event", "list_events", "find_free_slot"],
     [("title", "string"), ("start", "string"), ("duration_min", "number")], [("event_id", "string")]),
    ("maps", ["geocode", "route_planner", "nearby_search"],
     [("address", "string"), ("lat", "number"), ("lng", "number")], [("place_id", "string")]),
    ("shopping", ["search_products", "add_to_cart", "track_order"],
     [("keyword", "string"), ("order_id", "string")], [("product_id", "string")]),
    ("finance", ["stock_quote", "company_filings", "portfolio_value"],
     [("ticker", "string"), ("period", "string")], [("quote", "number")]),
    ("sports", ["match_scores", "league_table", "player_stats"],
     [("league", "string"), ("team", "string")], [("match_id", "string")]),
]

_QUERY_TMPL = [
    "I need to {a} and then {b}.",
    "Help me {a}, after that also {b}.",
    "First {a}; once that is done, please {b}.",
    "Can you {a}? Also {b} would be useful.",
]


def _verb(name: str) -> str:
    return name.replace("_", " ")


def make_synthetic(n_tools: int = 200, n_queries: int = 40, seed: int = 0
                   ) -> tuple[dict[str, ToolDoc], list[Query]]:
    """Deterministic ToolBench-like fallback (smoke tests only; no network needed)."""
    rng = random.Random(seed)
    tools: dict[str, ToolDoc] = {}
    per_theme = max(2, n_tools // len(_TOOL_THEMES))
    for theme, apis, req, out in _TOOL_THEMES:
        for k in range(per_theme):
            base = apis[k % len(apis)]
            tid = f"syn_{theme}_{base}_{k}"
            td = ToolDoc(
                id=tid, name=f"{base}_{k}",
                description=f"{base.replace('_', ' ')} for {theme} tasks (variant {k})",
                category=theme, method="GET",
                required_params=[Param(n, t) for n, t in req],
                out_fields=[Param(n, t) for n, t in out],
                library="synthetic",
            )
            tools[tid] = td
    ids = sorted(tools)
    queries = []
    for i in range(n_queries):
        t1, t2 = rng.sample(_TOOL_THEMES, 2)
        k1, k2 = rng.randrange(per_theme), rng.randrange(per_theme)
        g1 = f"syn_{t1[0]}_{t1[1][k1 % len(t1[1])]}_{k1}"
        g2 = f"syn_{t2[0]}_{t2[1][k2 % len(t2[1])]}_{k2}"
        gold = [g1, g2] if g1 != g2 else [g1]
        # ensure gold exists
        gold = [g for g in gold if g in tools] or [ids[stable_hash(f"q{i}") % len(ids)]]
        tmpl = _QUERY_TMPL[i % len(_QUERY_TMPL)]
        va, vb = _verb(tools[gold[0]].name), _verb(tools[gold[-1]].name)
        queries.append(Query(qid=f"syn_query_{i}", text=tmpl.format(a=va, b=vb), gold_ids=gold))
    return tools, queries


# ------------------------------------------------------------------ sub-library
def build_experiment_set(
    tools: dict[str, ToolDoc], queries: list[Query], n_queries: int, n_tools: int, seed: int
) -> tuple[dict[str, ToolDoc], list[Query], list[str]]:
    """Sample queries, then build a size-N sub-library that always contains gold tools.

    Mirrors the ToolRet difficulty scaling: larger N = harder discrimination task.
    """
    rng = seed_everything(seed)
    queries = list(queries)
    rng.shuffle(queries)
    queries = queries[:n_queries]

    gold_needed: set[str] = set()
    for q in queries:
        gold_needed.update(q.gold_ids)
    rest = [t for t in tools if t not in gold_needed]
    take = max(0, n_tools - len(gold_needed))
    chosen = rng.permutation(len(rest))[:take]
    keep = set(gold_needed) | {rest[i] for i in chosen}
    sub = {tid: tools[tid] for tid in tools if tid in keep}
    ids = sorted(sub)
    return sub, queries, ids
