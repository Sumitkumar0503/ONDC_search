# server.py — ONDC Semantic Search (OpenAI embeddings + GPT-4.1 + bge-reranker-large)
import hashlib
import os, re, json, glob, unicodedata
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, Body, Query
from pydantic import BaseModel
import time
from fastapi.responses import Response

# ---------- Config ----------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY","")
EMBEDDING_MODEL = "text-embedding-3-large"
RERANKER_MODEL_NAME = "BAAI/bge-reranker-large"  # ~1.3–1.4 GB download
CHROMA_DIR = "./data/chroma"
INBOX_DEFAULT = "./data/inbox"
COLLECTION_NAME = "ondc_catalog"
TERMS_JSON = "./data/terms.json"
DIET_CACHE_JSON = "./data/diet_cache.json"
SYNONYMS_JSON = "./data/synonyms.json"

# scoring weights
W_COSINE = 0.70
W_LEXICAL = 0.10
W_CONSTRAINT = 0.10
W_DISTANCE = 0.10
W_RERANKER = 0.60
W_BUSINESS = 0.40

MIN_FINAL_SCORE = 0.30
MIN_FINAL_SCORE_ALT = 0.35
FUZZY_CUTOFF = 0.78
MIN_COSINE = 0.35         # gate out weak vector matches early
MIN_RERANKER = 0.5        # gate out weak cross-encoder matches

# ---------- Libs ----------
import chromadb
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

WS = re.compile(r"\s+")
def norm_text(s: str) -> str:
    if not s: return ""
    s = unicodedata.normalize("NFKC", s).lower()
    s = s.replace("\u20b9","₹")
    s = re.sub(r"<[^>]+>", " ", s)
    s = WS.sub(" ", s).strip()
    return s

def tokenize_name(name: str):
    name = norm_text(name)
    return re.findall(r"[a-z0-9₹]+", name)

def jaccard(a, b):
    A, B = set(a), set(b)
    if not A or not B: return 0.0
    return len(A & B) / len(A | B)

def to_number(x):
    try:
        return float(x)
    except Exception:
        return None

# ---------- OpenAI: embeddings + GPT intent ----------
_oai = None
def oai():
    global _oai
    if _oai is None:
        _oai = OpenAI(api_key=OPENAI_API_KEY)
    return _oai

def embed_texts(texts: List[str]) -> List[List[float]]:
    resp = oai().embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [d.embedding for d in resp.data]

def parse_intent_gpt(q: str, pincode: Optional[str]):
    sys_prompt = (
        "Extract structured constraints for commerce search from a user query. "
        "Return strict JSON with keys: must_have_keywords (array of strings), "
        "diet ('veg'|'non_veg'|null), numeric_filters (array of {field, op, value}), "
        "pincode (string or null). Allowed fields: price_value, protein_g, kcal. Allowed ops: >=, <=."
    )
    try:
        resp = oai().chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role":"system","content":sys_prompt},
                {"role":"user","content":q}
            ],
            temperature=0
        )
        content = resp.choices[0].message.content.strip()
        m = re.search(r"\{.*\}", content, re.S)
        if m:
            data = json.loads(m.group(0))
            if pincode and not data.get("pincode"):
                data["pincode"] = pincode
            return data
    except Exception:
        pass
    return {"must_have_keywords":[], "diet": None, "numeric_filters": [], "pincode": pincode}

# ---------- Reranker (bge-large) ----------
_tok, _mdl = None, None
@torch.no_grad()
def rerank(query: str, passages: List[str]) -> List[float]:
    global _tok, _mdl
    if _mdl is None:
        _tok = AutoTokenizer.from_pretrained(RERANKER_MODEL_NAME)
        _mdl = AutoModelForSequenceClassification.from_pretrained(RERANKER_MODEL_NAME)
        _mdl.eval()
    if not passages:
        return []
    pairs = [f"query: {query} document: {p}" for p in passages]
    enc = _tok(pairs, padding=True, truncation=True, return_tensors="pt", max_length=512)
    logits = _mdl(**enc).logits.view(-1).tolist()
    mn, mx = min(logits), max(logits)
    if mx == mn:
        return [0.5] * len(logits)
    return [(x - mn) / (mx - mn) for x in logits]

# ---------- Chroma ----------
_ch_client, _col = None, None
def get_collection():
    global _ch_client, _col
    if _col is None:
        _ch_client = chromadb.PersistentClient(path=CHROMA_DIR)
        _col = _ch_client.get_or_create_collection(name=COLLECTION_NAME, metadata={"hnsw:space":"cosine"})
    return _col

# ---------- Parser for /on_search ----------
def _get_tag_value(tag_list, code, subcode=None):
    if not isinstance(tag_list, list): return None
    for blk in tag_list:
        if blk.get("code") == code:
            for e in blk.get("list", []) or []:
                if subcode is None or e.get("code") == subcode:
                    return e.get("value")
    return None

def iter_items(payload: Dict[str, Any]):
    ctx = payload.get("context", {}) or {}
    catalog = (payload.get("message", {}) or {}).get("catalog", {}) or {}
    domain = ctx.get("domain") or ""
    bpp_id = ctx.get("bpp_id") or ""
    providers = catalog.get("bpp/providers") or catalog.get("providers") or []

    for p in providers:
        provider_id = str(p.get("id",""))
        provider_name = ((p.get("descriptor") or {}) or {}).get("name","")
        locations = { (loc.get("id") or ""): loc for loc in (p.get("locations") or []) }

        current_base = None  # (base_item_dict, base_meta, base_doc_id, base_embedding_text_parts)

        for it in p.get("items") or []:
            tlist = it.get("tags") or []
            it_type_raw = _get_tag_value(tlist, "type", "type")
            it_type = str(it_type_raw).strip().lower() if it_type_raw else None
            is_related = bool(it.get("related"))

            desc = it.get("descriptor") or {}
            name = (desc.get("name") or "").strip()
            short_desc = (desc.get("short_desc") or "").strip()
            long_desc  = (desc.get("long_desc")  or "").strip()
            image_url = (desc.get("images") or [None])[0]
            category_id  = it.get("category_id")
            category_ids = it.get("category_ids") or []
            price = it.get("price") or {}
            price_val = to_number(price.get("value"))
            price_currency = price.get("currency") or ""
            vn = _get_tag_value(tlist, "veg_nonveg", "veg")
            seller_diet = "veg" if vn == "yes" else ("non_veg" if vn == "no" else "unknown")
            location_id = it.get("location_id") or ""
            loc = locations.get(location_id) or {}
            area_code = ((loc.get("address") or {}).get("area_code"))

            meta_common = {
                "bpp_id": bpp_id, "domain": domain,
                "provider_id": provider_id, "provider_name": provider_name,
                "image_url": image_url,
                "price_currency": price_currency,
                "category_id": category_id, "category_ids": category_ids,
                "location_id": location_id, "provider_area_code": area_code,
                "message_id": ctx.get("message_id"),
                "transaction_id": ctx.get("transaction_id"),
                "timestamp": ctx.get("timestamp"),
            }

            def build_embedding_parts(base_name, base_short, base_long, extra_bits=None):
                parts = [base_name]
                if base_short: parts.append(base_short)
                if base_long and base_long != base_short: parts.append(base_long)
                if seller_diet and seller_diet != "unknown":
                    pretty = seller_diet.replace("_", " ")
                    parts.append(f"Dietary: {seller_diet}")
                    parts.append(f"Dietary: {pretty}")
                    if seller_diet == "non_veg":
                        parts.append("non-veg")
                if category_id: parts.append(f"Category: {category_id}")
                if category_ids: parts.append("Categories: " + " | ".join(map(str, category_ids)))
                parts.append(f"Domain: {domain}")
                if extra_bits:
                    parts.extend(extra_bits)
                return parts

            # ---------- Base item ----------
            if it_type == "item" or (it_type is None and not is_related):
                tx = str(ctx.get("transaction_id") or "")
                msg = str(ctx.get("message_id") or "")
                item_id = str(it.get("id",""))
                base_id = f"{tx}:{msg}:{bpp_id}:{provider_id}:{item_id}"
                if not tx or not msg:
                    sig = json.dumps({
                        "bpp_id": bpp_id, "provider_id": provider_id, "item_id": item_id,
                        "timestamp": ctx.get("timestamp"), "name": name
                    }, sort_keys=True)
                    base_id = f"{base_id}:{hashlib.sha1(sig.encode('utf-8')).hexdigest()[:8]}"

                meta_base = dict(meta_common)
                meta_base.update({
                    "item_id": item_id,
                    "name": name,
                    "price_value": price_val,
                    "veg_non_veg": seller_diet,
                })

                emb_parts = build_embedding_parts(name, short_desc, long_desc)
                emb_text = "\n".join([norm_text(x) for x in emb_parts if x])

                # yield base
                yield base_id, emb_text, meta_base
                current_base = (it, meta_base, base_id, emb_parts)
                continue

            # ---------- Customization as derived SKU ----------
            if it_type == "customization" and is_related and current_base:
                base_it, base_meta, base_id, base_emb_parts = current_base
                child_name = name or "Customization"
                disp_name = f"{base_meta.get('name','') } - {child_name}".strip()
                child_price = price_val if (price_val is not None) else base_meta.get("price_value")
                child_item_id = str(it.get("id",""))
                child_doc_id = f"{base_id}::cust::{child_item_id}"

                meta_child = dict(meta_common)
                meta_child.update({
                    "item_id": child_item_id,
                    "name": disp_name,
                    "price_value": child_price,
                    "veg_non_veg": base_meta.get("veg_non_veg") or seller_diet,
                    "parent_item_id": base_meta.get("item_id"),
                })

                extra = []
                if child_name:
                    extra.append(f"Variant: {child_name}")
                if short_desc:
                    extra.append(short_desc)
                if long_desc and long_desc != short_desc:
                    extra.append(long_desc)
                emb_parts_child = build_embedding_parts(
                    base_meta.get("name",""),
                    "", "",
                    extra_bits=extra
                )
                emb_text_child = "\n".join([norm_text(x) for x in emb_parts_child if x])

                yield child_doc_id, emb_text_child, meta_child
            # else ignore other types

# ---------- Robust payload reader ----------
def iter_payloads_from_text(text: str):
    text = text.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            for o in obj:
                if isinstance(o, dict): yield o
        elif isinstance(obj, dict):
            yield obj
        return
    except Exception:
        pass

    good = True
    items = []
    for line in text.splitlines():
        line = line.strip()
        if not line: continue
        try:
            items.append(json.loads(line))
        except Exception:
            good = False
            break
    if good and items:
        for o in items:
            if isinstance(o, dict): yield o
        return

    dec = json.JSONDecoder()
    i = 0; L = len(text)
    while i < L:
        j = text.find('{"context"', i)
        if j == -1:
            break
        i = j
        try:
            obj, n = dec.raw_decode(text[i:])
            if isinstance(obj, dict):
                yield obj
            i += n
        except json.JSONDecodeError:
            i += 1

def iter_payloads_from_file(path: str):
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read()
    yield from iter_payloads_from_text(txt)

# ---------- Metadata sanitizer ----------
def sanitize_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in meta.items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            out[k] = v
        elif isinstance(v, list):
            out[k] = json.dumps(v, ensure_ascii=False)
        elif isinstance(v, dict):
            out[k] = json.dumps(v, ensure_ascii=False)
        else:
            out[k] = str(v)
    return out

# ---------- Diet enrichment ----------
_DIET_CACHE = None

def _load_diet_cache() -> Dict[str, Dict[str, Any]]:
    global _DIET_CACHE
    if _DIET_CACHE is None:
        try:
            with open(DIET_CACHE_JSON, "r", encoding="utf-8") as f:
                _DIET_CACHE = json.load(f)
        except Exception:
            _DIET_CACHE = {}
    return _DIET_CACHE

def _save_diet_cache():
    if _DIET_CACHE is None:
        return
    os.makedirs(os.path.dirname(DIET_CACHE_JSON), exist_ok=True)
    with open(DIET_CACHE_JSON, "w", encoding="utf-8") as f:
        json.dump(_DIET_CACHE, f, ensure_ascii=False, indent=2)

_NONVEG_TOKENS = {
    "chicken","mutton","egg","fish","prawn","prawns","shrimp","lamb","beef","pork",
    "bacon","ham","pepperoni","sausage","salami","keema","kebab","seekh","boneless",
    "tuna","anchovy","anchovies"
}
_VEG_TOKENS = {
    "veg","vegetable","paneer","aloo","dal","dhal","chana","chole","gobi","cauliflower",
    "palak","mushroom","bhindi","baingan","eggless","sambar","idli","dosa","parotta",
    "chapati","roti","poha","upma","falafel","soy","soya"
}

def _diet_from_heuristic(text: str) -> str:
    if not text:
        return "unknown"
    t = norm_text(text)
    has_nonveg = any(tok in t for tok in _NONVEG_TOKENS)
    has_veg    = any(tok in t for tok in _VEG_TOKENS)
    if has_nonveg and not has_veg:
        return "non_veg"
    if has_veg and not has_nonveg:
        return "veg"
    if has_nonveg and has_veg:
        return "non_veg"
    return "unknown"

def _diet_from_llm(name: str, desc: str) -> tuple[str, float]:
    try:
        prompt_user = (
            "Classify the food item for Indian context as exactly one of: "
            "'veg' or 'non_veg'. If you cannot tell, say 'unknown'. "
            "Consider that eggs, fish and all meat imply 'non_veg'.\n\n"
            f"Name: {name or ''}\n"
            f"Description: {desc or ''}\n\n"
            "Respond ONLY as strict JSON: {\"label\":\"veg|non_veg|unknown\",\"confidence\":0..1}"
        )
        resp = oai().chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": "You are a precise classifier for veg vs non_veg in Indian F&B menus."},
                {"role": "user", "content": prompt_user},
            ],
            temperature=0
        )
        content = (resp.choices[0].message.content or "").strip()
        m = re.search(r"\{.*\}", content, re.S)
        if not m:
            return "unknown", 0.0
        data = json.loads(m.group(0))
        label = data.get("label") or "unknown"
        conf  = float(data.get("confidence") or 0.0)
        if label not in ("veg","non_veg","unknown"):
            label = "unknown"
        return label, max(0.0, min(1.0, conf))
    except Exception:
        return "unknown", 0.0

def enrich_diet(meta: Dict[str, Any], embedding_text: str) -> Dict[str, Any]:
    seller = meta.get("veg_non_veg") or "unknown"
    if seller in ("veg", "non_veg"):
        meta["veg_non_veg_llm"] = "unknown"
        meta["veg_non_veg_llm_conf"] = 0.0
        meta["veg_non_veg_final"] = seller
        meta["diet_source"] = "seller"
        return meta
    key = f"{meta.get('bpp_id','')}|{meta.get('provider_id','')}|{meta.get('item_id','')}"
    cache = _load_diet_cache()
    if key in cache:
        cached = cache[key]
        meta["veg_non_veg_llm"] = cached.get("llm", "unknown")
        meta["veg_non_veg_llm_conf"] = cached.get("conf", 0.0)
        meta["veg_non_veg_final"] = cached.get("final", "unknown")
        meta["diet_source"] = cached.get("source", "unknown")
        return meta
    name = meta.get("name") or ""
    heur = _diet_from_heuristic((name + " " + (embedding_text or "")).strip())
    if heur in ("veg","non_veg"):
        meta["veg_non_veg_llm"] = "unknown"
        meta["veg_non_veg_llm_conf"] = 0.0
        meta["veg_non_veg_final"] = heur
        meta["diet_source"] = "heuristic"
        cache[key] = {"llm":"unknown","conf":0.0,"final":heur,"source":"heuristic"}
        _save_diet_cache()
        return meta
    label, conf = _diet_from_llm(name, embedding_text)
    meta["veg_non_veg_llm"] = label
    meta["veg_non_veg_llm_conf"] = conf
    meta["veg_non_veg_final"] = label if label in ("veg","non_veg") else "unknown"
    meta["diet_source"] = "llm" if label in ("veg","non_veg") else "unknown"
    cache[key] = {"llm": label, "conf": conf, "final": meta["veg_non_veg_final"], "source": meta["diet_source"]}
    _save_diet_cache()
    return meta

# ---------- Terms updater (for autocomplete) ----------
def _update_terms(names: List[str]):
    from collections import Counter
    freq = Counter()
    if os.path.exists(TERMS_JSON):
        try:
            with open(TERMS_JSON, "r", encoding="utf-8") as f:
                prev = json.load(f)
                if isinstance(prev, dict):
                    freq.update(prev)
        except Exception:
            pass
    for n in names:
        n = norm_text(n or "")
        toks = [t for t in n.split() if t]
        for t in toks:
            if len(t) >= 2:
                freq[t] += 1
        for i in range(len(toks) - 1):
            bg = f"{toks[i]} {toks[i+1]}"
            freq[bg] += 1
    os.makedirs(os.path.dirname(TERMS_JSON), exist_ok=True)
    with open(TERMS_JSON, "w", encoding="utf-8") as f:
        json.dump(dict(freq.most_common(20000)), f, ensure_ascii=False, indent=2)
    global _terms_cache
    _terms_cache = None

# ---------- Synonyms ----------
_syn_c2v, _syn_v2c, _syn_variants = None, None, None

def _load_synonyms():
    """
    Returns (canon->variants, variant->canon, variants_sorted_by_length_desc)
    All normalized via norm_text().
    """
    global _syn_c2v, _syn_v2c, _syn_variants
    if _syn_c2v is not None:
        return _syn_c2v, _syn_v2c, _syn_variants
    try:
        with open(SYNONYMS_JSON, "r", encoding="utf-8") as f:
            raw = json.load(f) or {}
    except Exception:
        raw = {}
    c2v = {}
    if isinstance(raw, dict):
        for canon, variants in raw.items():
            c = norm_text(canon or "")
            vs = sorted({norm_text(v) for v in (variants or []) if v}, key=lambda s: (-len(s), s))
            vs = [v for v in vs if v and v != c]
            if c:
                c2v[c] = vs
    v2c = {}
    for c, vs in c2v.items():
        for v in [c] + vs:
            v2c[v] = c
    variants_sorted = sorted(v2c.keys(), key=lambda s: (-len(s), s))
    _syn_c2v, _syn_v2c, _syn_variants = c2v, v2c, variants_sorted
    return _syn_c2v, _syn_v2c, _syn_variants

def _phrase_regex(phrase: str):
    esc = re.escape(phrase).replace(r"\ ", r"\s+")
    return re.compile(rf"\b{esc}\b", flags=re.IGNORECASE)

def normalize_with_synonyms(q: str) -> str:
    """
    Query-side canonicalization: replace any known variant with its canonical form.
    Longest-first order to avoid partial overlaps.
    """
    text = norm_text(q)
    _, v2c, variants = _load_synonyms()
    if not variants:
        return text
    for v in variants:
        canon = v2c.get(v)
        if not canon or canon == v:
            continue
        text = _phrase_regex(v).sub(canon, text)
    return text

def augment_text_with_synonyms(text: str) -> str:
    """
    Index-side augmentation: if the item text contains a canonical or any variant,
    append a synthetic line listing that cluster to teach the embedding.
    """
    c2v, _, _ = _load_synonyms()
    if not c2v:
        return norm_text(text)
    hay = norm_text(text)
    extra = []
    for canon, variants in c2v.items():
        hit = _phrase_regex(canon).search(hay) is not None
        if not hit:
            for v in variants:
                if _phrase_regex(v).search(hay):
                    hit = True
                    break
        if hit:
            extra.append("synonyms: " + " ".join([canon] + variants))
    if not extra:
        return hay
    return (hay + "\n" + "\n".join(extra)).strip()

# ---------- Ingestion ----------
def ingest_payload(payload: Dict[str, Any]) -> int:
    col = get_collection()
    by_id: Dict[str, tuple[str, Dict[str, Any]]] = {}
    names: List[str] = []

    for doc_id, emb_text, meta in iter_items(payload):
        emb_text = emb_text or ""
        try:
            meta = enrich_diet(meta, emb_text)
        except Exception as e:
            meta = dict(meta)
            meta.setdefault("veg_non_veg_llm", None)
            meta.setdefault("veg_non_veg_final", meta.get("veg_non_veg") or "unknown")
            meta["veg_non_veg_llm_error"] = str(e)[:120]

        final_diet = meta.get("veg_non_veg_final")
        if final_diet in ("veg", "non_veg") and "dietary:" not in emb_text:
            pretty = final_diet.replace("_", " ")
            extras = [f"dietary: {final_diet}", f"dietary: {pretty}"]
            if final_diet == "non_veg":
                extras.append("non-veg")
            emb_text = emb_text + "\n" + "\n".join(extras)

        # ---- NEW: add synonyms expansion before embedding ----
        emb_text = augment_text_with_synonyms(emb_text)
        emb_text = norm_text(emb_text)
        # ------------------------------------------------------

        meta_s = sanitize_meta(meta)
        by_id[doc_id] = (emb_text, meta_s)

        nm = meta.get("name")
        if nm:
            names.append(nm)

    if not by_id:
        return 0

    ids = list(by_id.keys())
    docs = [by_id[i][0] for i in ids]
    metas = [by_id[i][1] for i in ids]
    embs = embed_texts(docs)
    col.upsert(ids=ids, documents=docs, embeddings=embs, metadatas=metas)

    _update_terms(names)
    return len(ids)

def ingest_folder(folder: str) -> int:
    count = 0
    for path in glob.glob(os.path.join(folder, "**/*.json"), recursive=True):
        for payload in iter_payloads_from_file(path):
            count += ingest_payload(payload)
    return count

# ---------- Retrieval + rank ----------
def retrieve(query: str, top_k: int = 100):
    col = get_collection()
    emb = embed_texts([query])[0]
    out = col.query(query_embeddings=[emb], n_results=top_k, include=["documents","metadatas","distances"])
    docs  = out.get("documents", [[]])[0]
    metas = out.get("metadatas", [[]])[0]
    dists = out.get("distances", [[]])[0]
    cos   = [1 - d for d in dists]
    return docs, metas, cos

def lexical_overlap_score(query: str, item_name: str) -> float:
    return jaccard(tokenize_name(query), tokenize_name(item_name or ""))

def constraint_score(intent: Dict[str, Any], meta: Dict[str, Any]) -> float:
    scores = []
    diet = intent.get("diet")
    if diet:
        final_label = meta.get("veg_non_veg_final") or meta.get("veg_non_veg") or "unknown"
        scores.append(1.0 if final_label == diet else (0.5 if final_label == "unknown" else 0.0))
    for nf in intent.get("numeric_filters", []):
        field, op, val = nf.get("field"), nf.get("op"), nf.get("value")
        if field == "price_value" and op == "<=":
            have = meta.get("price_value")
            scores.append(1.0 if (have is not None and float(have) <= float(val)) else (0.5 if have is None else 0.0))
        if field == "protein_g" and op == ">=":
            have = meta.get("protein_g")
            scores.append(1.0 if (have is not None and float(have) >= float(val)) else (0.5 if have is None else 0.0))
    if not scores:
        return 0.5
    return sum(scores)/len(scores)

def distance_score(pincode: Optional[str], meta: Dict[str, Any]) -> float:
    if not pincode: return 0.0
    return 1.0 if meta.get("provider_area_code")==pincode else 0.0

@torch.no_grad()
def final_rank(query: str, docs: List[str], metas: List[Dict[str,Any]], cosines: List[float], intent: Dict[str, Any], top_n: int = 20):
    rr = rerank(query, docs) if docs else []
    results = []
    for i, m in enumerate(metas):
        lex = lexical_overlap_score(query, m.get("name",""))
        cst = constraint_score(intent, m)
        dst = distance_score(intent.get("pincode"), m)
        business = (W_COSINE*cosines[i]) + (W_LEXICAL*lex) + (W_CONSTRAINT*cst) + (W_DISTANCE*dst)
        rrs = rr[i] if i < len(rr) else 0.5
        final = (W_RERANKER*rrs) + (W_BUSINESS*business)
        if rrs < MIN_RERANKER or final < MIN_FINAL_SCORE:
            continue
        item = dict(m); item["_scores"] = {
            "cosine": round(cosines[i],4),
            "lexical": round(lex,4),
            "constraint": round(cst,4),
            "distance": round(dst,4),
            "business": round(business,4),
            "reranker": round(rrs,4),
            "final": round(final,4),
        }
        item["_snippet"] = docs[i][:200]
        results.append(item)
    results.sort(key=lambda x: -x["_scores"]["final"])
    return results[:top_n]

# ---------- Autocomplete ----------
import difflib
_terms_cache = None

def _load_terms_cache():
    global _terms_cache
    if _terms_cache is None:
        try:
            with open(TERMS_JSON,"r",encoding="utf-8") as f:
                _terms_cache = json.load(f)
        except Exception:
            _terms_cache = {}
    return _terms_cache

def autocorrect_query(q: str) -> str:
    terms = _load_terms_cache()
    vocab = list(terms.keys()) if terms else []
    if not vocab:
        return q
    toks = q.split()
    fixed = []
    for t in toks:
        if re.fullmatch(r"[a-zA-Z]+", t) and len(t) >= 2:
            cand = difflib.get_close_matches(norm_text(t), vocab, n=1, cutoff=FUZZY_CUTOFF)
            fixed.append(cand[0] if cand else t)
        else:
            fixed.append(t)
    return " ".join(fixed)

def collection_domains() -> List[str]:
    col = get_collection()
    got = col.get(include=["metadatas"], limit=200000)
    return sorted({m.get("domain") for m in got.get("metadatas",[]) if m and m.get("domain")})

def suggest(prefix: str, n: int = 8):
    terms = _load_terms_cache()
    p = norm_text(prefix)
    if len(p) < 2:
        return [k for k,_ in sorted(terms.items(), key=lambda x: -x[1])[:n]]
    cand = [(k,v) for k,v in terms.items() if k.startswith(p)]
    cand.sort(key=lambda x: (-x[1], len(x[0])))
    return [k for k,_ in cand[:n]]

# ---------- FastAPI ----------
app = FastAPI(title="ONDC Semantic Search (GPT + bge-rerank)", version="1.0")

class IngestFolder(BaseModel):
    folder: str = INBOX_DEFAULT

@app.post("/ingest")
def api_ingest(p: IngestFolder):
    os.makedirs(p.folder, exist_ok=True)
    count = ingest_folder(p.folder)
    return {"indexed_items": count}

@app.post("/ingest_raw")
def api_ingest_raw(payload: Dict[str, Any] = Body(...)):
    count = ingest_payload(payload)
    return {"indexed_items": count}

@app.get("/suggest")
def api_suggest(q: str = Query("")):
    return {"q": q, "suggestions": suggest(q)}

class SearchResponse(BaseModel):
    q: str
    intent: Dict[str, Any]
    top_k: int
    top_n: int
    results: List[Dict[str, Any]]

def _passes_numeric_filters(meta: Dict[str, Any], intent: Dict[str, Any]) -> bool:
    nfs = intent.get("numeric_filters") or []
    if not nfs:
        return True
    for nf in nfs:
        field = nf.get("field")
        op    = nf.get("op")
        val   = nf.get("value")
        if field not in ("price_value", "protein_g", "kcal"):
            continue
        have = meta.get(field)
        if have is None:
            continue
        try:
            v = float(val)
        except Exception:
            continue
        if op == "<=" and not (float(have) <= v):
            return False
        if op == ">=" and not (float(have) >= v):
            return False
    return True

def _is_diet_compatible(requested: Optional[str], meta: Dict[str,Any]) -> bool:
    if not requested:
        return True
    final_label = meta.get("veg_non_veg_final") or meta.get("veg_non_veg") or "unknown"
    if requested == "veg" and final_label == "non_veg":
        return False
    if requested == "non_veg" and final_label == "veg":
        return False
    return True

def _has_all_keywords(meta: Dict[str,Any], doc: str, kws: List[str]) -> bool:
    if not kws: return True
    hay = norm_text((meta.get("name","") or "") + " " + (doc or ""))
    return all(k in hay for k in kws)

def _prefilter(docs, metas, cos, intent):
    kws = intent.get("must_have_keywords", []) or []
    requested_diet = intent.get("diet")
    cos_gate = MIN_COSINE if not kws else max(0.15, MIN_COSINE - 0.10)
    kept = []
    for d, m, c in zip(docs, metas, cos):
        if c < cos_gate:
            continue
        if not _has_all_keywords(m, d, kws):
            continue
        if not _is_diet_compatible(requested_diet, m):
            continue
        if not _passes_numeric_filters(m, intent):
            continue
        kept.append((d, m, c))
    if not kept:
        return [], [], []
    d, m, c = zip(*kept)
    return list(d), list(m), list(c)

# --- replace the existing /search handler with this version ---
@app.get("/search", response_model=SearchResponse)
def api_search(
    q: str,
    pincode: Optional[str] = Query(None),
    top_k: int = 100,
    top_n: int = 20,
    pretty: Optional[bool] = Query(False)
):
    t0 = time.perf_counter()

    qn = normalize_with_synonyms(q)  # includes norm_text inside
    t_intent0 = time.perf_counter()
    intent = parse_intent_gpt(qn, pincode)
    t_intent1 = time.perf_counter()

    # 1) retrieval on original query + prefilter
    t_vec0 = time.perf_counter()
    docs, metas, cos = retrieve(qn, top_k=top_k)
    t_vec1 = time.perf_counter()

    t_pref0 = time.perf_counter()
    docs, metas, cos = _prefilter(docs, metas, cos, intent)
    t_pref1 = time.perf_counter()

    # If nothing passes the prefilter, try autocorrect once
    did_autocorrect = False
    corrected_q = None
    t_auto_ms = 0.0
    if not docs:
        t_auto0 = time.perf_counter()
        corrected_q = autocorrect_query(qn)
        if corrected_q != qn:
            corrected_q = normalize_with_synonyms(corrected_q)
            docs2, metas2, cos2 = retrieve(corrected_q, top_k=top_k)
            docs2, metas2, cos2 = _prefilter(docs2, metas2, cos2, intent)
            if docs2:
                did_autocorrect = True
                docs, metas, cos = docs2, metas2, cos2
        t_auto1 = time.perf_counter()
        t_auto_ms = (t_auto1 - t_auto0) * 1000.0

    # Still empty? return clean no-results envelope
    if not docs:
        t_end = time.perf_counter()
        payload = {
            "q": q, 
            "normalized_q": qn,
            "used_q": corrected_q or qn,
            "intent": intent, 
            "top_k": top_k, 
            "top_n": top_n,
            "results": [],
            "no_results": True,
            "reason": "No strong semantic matches in the indexed catalog.",
            "domains_present": collection_domains(),
            "suggestions": suggest(qn),
            "did_autocorrect": did_autocorrect,
            "corrected_q": corrected_q,
            "timings": {
                "intent_ms": (t_intent1 - t_intent0) * 1000.0,
                "vector_retrieval_ms": (t_vec1 - t_vec0) * 1000.0,
                "prefilter_ms": (t_pref1 - t_pref0) * 1000.0,
                "autocorrect_ms": t_auto_ms,
                "rank_ms": 0.0,
                "total_ms": (t_end - t0) * 1000.0
            }
        }
        if pretty:
            return Response(content=json.dumps(payload, ensure_ascii=False, indent=2),
                            media_type="application/json")
        return payload

    # 2) final rank (includes reranker + blending)
    t_rank0 = time.perf_counter()
    ranked = final_rank(corrected_q or qn, docs, metas, cos, intent, top_n=top_n)
    t_rank1 = time.perf_counter()

    # If ranking filtered everything, treat as no-results
    if not ranked:
        t_end = time.perf_counter()
        payload = {
            "q": q, 
            "normalized_q": qn,
            "used_q": corrected_q or qn,
            "intent": intent, 
            "top_k": top_k, 
            "top_n": top_n,
            "results": [],
            "no_results": True,
            "reason": "Matches failed quality thresholds.",
            "domains_present": collection_domains(),
            "suggestions": suggest(qn),
            "did_autocorrect": did_autocorrect,
            "corrected_q": corrected_q,
            "timings": {
                "intent_ms": (t_intent1 - t_intent0) * 1000.0,
                "vector_retrieval_ms": (t_vec1 - t_vec0) * 1000.0,
                "prefilter_ms": (t_pref1 - t_pref0) * 1000.0,
                "autocorrect_ms": t_auto_ms,
                "rank_ms": (t_rank1 - t_rank0) * 1000.0,
                "total_ms": (t_end - t0) * 1000.0
            }
        }
        if pretty:
            return Response(content=json.dumps(payload, ensure_ascii=False, indent=2),
                            media_type="application/json")
        return payload

    t_end = time.perf_counter()
    payload = {
        "q": q,
        "normalized_q": qn,
        "used_q": corrected_q or qn,
        "intent": intent,
        "top_k": top_k,
        "top_n": top_n,
        "results": ranked,
        "did_autocorrect": did_autocorrect,
        "corrected_q": corrected_q,
        "timings": {
            "intent_ms": (t_intent1 - t_intent0) * 1000.0,
            "vector_retrieval_ms": (t_vec1 - t_vec0) * 1000.0,
            "prefilter_ms": (t_pref1 - t_pref0) * 1000.0,
            "autocorrect_ms": t_auto_ms,
            "rank_ms": (t_rank1 - t_rank0) * 1000.0,
            "total_ms": (t_end - t0) * 1000.0
        }
    }
    if pretty:
        return Response(content=json.dumps(payload, ensure_ascii=False, indent=2),
                        media_type="application/json")
    return payload


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080, reload=True)
