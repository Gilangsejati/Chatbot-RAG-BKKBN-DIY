# server_web.py (revisi: OOC -> opsi eskalasi ke WA admin)
import os
import sys
import json
import uuid
import tempfile
import traceback
import re
from pathlib import Path
from typing import Dict, List

from flask import Flask, request, jsonify, render_template

# try optional fuzzy libs (may be absent)
try:
    from rapidfuzz import process, fuzz
except Exception:
    process = None
    fuzz = None

# -----------------------
# Paths
# -----------------------
ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
STATIC_FOLDER = ROOT / "static"
TEMPLATES_FOLDER = ROOT / "templates"
METADATA_JSON = ROOT / "metadata.json"

# ensure project root on path so import src.* works
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import importlib
import importlib.util

def import_src(name: str):
    full = f"src.{name}"
    try:
        return importlib.import_module(full)
    except Exception:
        path = SRC_DIR / f"{name}.py"
        if not path.exists():
            return None
        try:
            spec = importlib.util.spec_from_file_location(full, str(path))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
        except Exception:
            print(f"[WARN] fallback import failed for {name}:\n{traceback.format_exc()}")
            return None

# load optional src modules
preprocessing_mod = import_src("preprocessing")
embedding_store_mod = import_src("embedding_store")
retriever_mod = import_src("retriever")
generator_mod = import_src("generator")
voice_mod = import_src("voice")
config_mod = import_src("config")

TOP_K_DEFAULT = getattr(config_mod, "TOP_K", 5) if config_mod else 5

# -----------------------
# Behaviour config (env override)
# -----------------------
ADMIN_WA = os.environ.get("ADMIN_WA", "6287819985271")  # change to real number
OOC_SCORE_THRESHOLD = float(os.environ.get("OOC_SCORE_THRESHOLD", "0.75"))  # below => considered OOC
FUZZY_MIN_SCORE = int(os.environ.get("FUZZY_MIN_SCORE", "80"))

# Hard blacklist keywords that are always OOC (adjust to your needs)
OOC_KEYWORDS = set([
    "presiden", "politik", "pemilu", "jokowi",
    "prabowo", "agama", "korupsi", "partai", "film", "lagu", "music",
    "movie", "sejarah"
])

def make_wa_link(phone: str, text: str = ""):
    import urllib.parse
    t = urllib.parse.quote_plus(text or "")
    return f"https://wa.me/{phone}?text={t}"

# -----------------------
# Flask app
# -----------------------
app = Flask(__name__, static_folder=str(STATIC_FOLDER), template_folder=str(TEMPLATES_FOLDER))

# Globals for pipeline
PREPROCESSOR = None
EMB_STORE = None
RETRIEVER = None
GENERATOR = None
VOICE = None
DF_FAQ = None

# quick map cleaned question -> payload
QUESTION_TO_ANSWER: Dict[str, dict] = {}

# -----------------------
# Utilities
# -----------------------
def safe_instance(mod, class_name):
    if not mod:
        return None
    cls = getattr(mod, class_name, None)
    if not cls:
        return None
    try:
        return cls()
    except Exception:
        try:
            return cls(None)
        except Exception:
            print(f"[WARN] Cannot instantiate {class_name} from {mod}")
            return None

def load_metadata():
    if METADATA_JSON.exists():
        try:
            with open(METADATA_JSON, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None

def normalize_text(s: str) -> str:
    if s is None:
        return ""
    try:
        if PREPROCESSOR and hasattr(PREPROCESSOR, "text_cleanup"):
            return PREPROCESSOR.text_cleanup(s).strip().lower()
    except Exception:
        pass
    s2 = str(s).strip().lower()
    s2 = re.sub(r"[^\w\s]", " ", s2, flags=re.UNICODE)
    s2 = " ".join(s2.split())
    return s2

def strip_leading_question_phrases(s: str) -> str:
    if not s:
        return ""
    s2 = str(s).strip().lower()
    s2 = re.sub(r'^\s*(apa yang dimaksud( dengan)?\s+|apa itu\s+|apa\s+|siapa\s+|mengapa\s+|bagaimana\s+|jelaskan\s+)', '', s2, flags=re.I)
    s2 = s2.strip()
    return s2

# -----------------------
# Naturalizer
# -----------------------
def _clean_whitespace(s: str) -> str:
    return re.sub(r'\s{2,}', ' ', (s or "").strip())

def naturalize_answer(raw_answer: str, question: str = None) -> str:
    if not raw_answer:
        return "Maaf, saya belum menemukan informasi tersebut dalam FAQ."
    a = str(raw_answer).strip()
    a = re.sub(r'(\d+)[\.\)]\s*', '\n\\1) ', a)
    a = a.replace(';', '. ')
    parts = [p.strip() for p in re.split(r'\n+', a) if p.strip()]
    if len(parts) > 1:
        a = '\n\n'.join(parts)
    else:
        a = parts[0] if parts else a
    a = _clean_whitespace(a)
    if not re.search(r'[\.!?]$', a):
        a = a + '.'
    if question:
        intro = "Tentu. Berikut penjelasan singkat mengenai pertanyaan Anda:"
    else:
        intro = "Berikut penjelasan singkat:"
    if '\n' in a:
        out = f"{intro}\n\n{a}"
    else:
        out = f"{intro} {a}"
    return out

# -----------------------
# Init once
# -----------------------
_init_done = False

@app.before_request
def init_pipeline_once():
    global _init_done
    if _init_done:
        return
    _init_done = True

    global PREPROCESSOR, EMB_STORE, RETRIEVER, GENERATOR, VOICE, DF_FAQ

    PREPROCESSOR = safe_instance(preprocessing_mod, "Preprocessor")
    EMB_STORE = safe_instance(embedding_store_mod, "EmbeddingStore")

    # instantiate retriever if available (try multiple signatures)
    if retriever_mod:
        RetrieverCls = getattr(retriever_mod, "Retriever", None)
        if RetrieverCls:
            try:
                if PREPROCESSOR and hasattr(PREPROCESSOR, "preprocess"):
                    maybe_df = PREPROCESSOR.preprocess()
                    DF_FAQ = maybe_df
                    try:
                        RETRIEVER = RetrieverCls(DF_FAQ, EMB_STORE)
                    except Exception:
                        RETRIEVER = RetrieverCls(EMB_STORE)
                else:
                    RETRIEVER = RetrieverCls(EMB_STORE)
            except Exception:
                try:
                    RETRIEVER = RetrieverCls()
                except Exception:
                    RETRIEVER = None

    GENERATOR = safe_instance(generator_mod, "GeminiGenerator") or safe_instance(generator_mod, "Generator")
    VOICE = voice_mod

    try:
        if PREPROCESSOR and hasattr(PREPROCESSOR, "preprocess") and DF_FAQ is None:
            DF_FAQ = PREPROCESSOR.preprocess()
    except Exception:
        DF_FAQ = None

    print(">>> [INIT] PREPROCESSOR=%s EMB_STORE=%s RETRIEVER=%s GENERATOR=%s DF_FAQ=%s" %
          (bool(PREPROCESSOR), bool(EMB_STORE), bool(RETRIEVER), bool(GENERATOR), "yes" if DF_FAQ is not None else "no"))

# -----------------------
# Build quick qmap
# -----------------------
def build_qmap():
    global QUESTION_TO_ANSWER
    if QUESTION_TO_ANSWER:
        return
    meta = load_metadata()
    if meta:
        for item in meta:
            raw_q = item.get("question") or item.get("title") or item.get("question_text") or ""
            qc_full = normalize_text(raw_q)
            qc_short = normalize_text(strip_leading_question_phrases(raw_q))
            payload = {
                "answer": item.get("answer") or item.get("answer_text") or item.get("text") or "",
                "index": item.get("index") or item.get("id"),
                "category": item.get("category") or item.get("kategori") or "Umum"
            }
            if qc_full:
                QUESTION_TO_ANSWER[qc_full] = payload
            if qc_short and qc_short not in QUESTION_TO_ANSWER:
                QUESTION_TO_ANSWER[qc_short] = payload
    elif DF_FAQ is not None:
        for i, row in DF_FAQ.iterrows():
            raw_q = row.get("question") or ""
            qc_full = normalize_text(raw_q)
            qc_short = normalize_text(strip_leading_question_phrases(raw_q))
            payload = {
                "answer": row.get("answer") or row.get("jawaban") or "",
                "index": int(i),
                "category": row.get("kategori") if "kategori" in row else "Umum"
            }
            if qc_full:
                QUESTION_TO_ANSWER[qc_full] = payload
            if qc_short and qc_short not in QUESTION_TO_ANSWER:
                QUESTION_TO_ANSWER[qc_short] = payload

# -----------------------
# Routes
# -----------------------
@app.route("/")
def index():
    try:
        return render_template("index.html", admin_wa=ADMIN_WA)
    except Exception:
        return "<h3>Chatbot server running</h3>"

@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "preprocessor": bool(PREPROCESSOR),
        "emb_store": bool(EMB_STORE),
        "retriever": bool(RETRIEVER),
        "generator": bool(GENERATOR),
        "voice": bool(VOICE),
    })

@app.route("/api/admin")
def api_admin():
    return jsonify({"wa": ADMIN_WA})

@app.route("/api/categories")
def api_categories():
    meta = load_metadata()
    if meta:
        cats = {}
        for item in meta:
            cat = (item.get("category") or item.get("kategori") or "Umum")
            cats.setdefault(cat, 0)
            cats[cat] += 1
        return jsonify(categories=[{"name": k, "count": v} for k, v in cats.items()])
    if DF_FAQ is not None:
        if "kategori" in DF_FAQ.columns:
            vc = DF_FAQ["kategori"].fillna("Umum").value_counts().to_dict()
            return jsonify(categories=[{"name": k, "count": v} for k, v in vc.items()])
        if "category" in DF_FAQ.columns:
            vc = DF_FAQ["category"].fillna("Umum").value_counts().to_dict()
            return jsonify(categories=[{"name": k, "count": v} for k, v in vc.items()])
    return jsonify(categories=[])

@app.route("/api/questions")
def api_questions():
    category = request.args.get("category")
    limit = int(request.args.get("limit", 50))
    meta = load_metadata()
    out = []
    if meta:
        cat_lower = category.strip().lower() if category else None
        for item in meta:
            item_cat = (item.get("category") or item.get("kategori") or "Umum")
            if cat_lower is None or item_cat.strip().lower() == cat_lower:
                out.append({
                    "id": item.get("index") or item.get("id") or str(uuid.uuid4()),
                    "text": item.get("question") or item.get("question_text") or item.get("title")
                })
                if len(out) >= limit:
                    break
        return jsonify(questions=out)
    if DF_FAQ is not None:
        if category:
            if "kategori" in DF_FAQ.columns:
                df_filtered = DF_FAQ[DF_FAQ["kategori"].fillna("").str.strip().str.lower() == category.strip().lower()]
            elif "category" in DF_FAQ.columns:
                df_filtered = DF_FAQ[DF_FAQ["category"].fillna("").str.strip().str.lower() == category.strip().lower()]
            else:
                df_filtered = DF_FAQ
            texts = df_filtered["question"].fillna("").tolist()[:limit]
            return jsonify(questions=[{"id": i, "text": t} for i, t in enumerate(texts)])
        texts = DF_FAQ["question"].fillna("").tolist()[:limit]
        return jsonify(questions=[{"id": i, "text": t} for i, t in enumerate(texts)])
    return jsonify(questions=[])

@app.route("/api/suggest-questions")
def api_suggest_questions():
    category = request.args.get("category", "")
    limit = int(request.args.get("limit", 5))
    q_resp = api_questions()
    try:
        data = q_resp.get_json()
        qs = data.get("questions", [])[:limit]
        return jsonify({"category": category or "Umum", "questions": [q["text"] for q in qs]})
    except Exception:
        meta = load_metadata()
        out = []
        if meta:
            cat_lower = (category or "").strip().lower() or None
            for item in meta:
                item_cat = (item.get("category") or item.get("kategori") or "Umum")
                if cat_lower is None or item_cat.strip().lower() == cat_lower:
                    out.append(item.get("question") or "")
                    if len(out) >= limit:
                        break
        return jsonify({"category": category or "Umum", "questions": out})

# -----------------------
# Chat endpoint (with OOC -> optional escalate to WA)
# -----------------------
@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.get_json(force=True, silent=True) or {}
    message = data.get("message")
    if not message:
        return jsonify(error="missing 'message' field"), 400
    session_id = data.get("session_id") or f"s_{uuid.uuid4().hex[:8]}"
    top_k = int(data.get("top_k", TOP_K_DEFAULT)) if data.get("top_k") else TOP_K_DEFAULT
    category = data.get("category")
    escalate = bool(data.get("escalate", True))  # if True, include wa_link when unsupported

    # ensure qmap built
    if not QUESTION_TO_ANSWER:
        build_qmap()

    # preprocess
    proc_text = message
    try:
        if PREPROCESSOR and hasattr(PREPROCESSOR, "preprocess_text"):
            proc_text = PREPROCESSOR.preprocess_text(message)
        elif PREPROCESSOR and hasattr(PREPROCESSOR, "text_cleanup"):
            proc_text = PREPROCESSOR.text_cleanup(message)
    except Exception:
        proc_text = message

    key_full = normalize_text(proc_text)
    key_short = normalize_text(strip_leading_question_phrases(proc_text))

    # exact / short-key
    item = None
    if key_full and key_full in QUESTION_TO_ANSWER:
        item = QUESTION_TO_ANSWER[key_full]
    elif key_short and key_short in QUESTION_TO_ANSWER:
        item = QUESTION_TO_ANSWER[key_short]

    if item:
        raw_answer = item.get("answer") or ""
        user_answer = naturalize_answer(raw_answer, message)
        messages = [{"role": "assistant", "id": f"a_{uuid.uuid4().hex[:8]}", "text": user_answer}]
        retrieved = [{"index": item.get("index"), "score": 1.0, "question": message, "answer": raw_answer}]
        return jsonify(session_id=session_id, messages=messages, retrieved=retrieved)

    # fuzzy (rapidfuzz)
    if process is not None and fuzz is not None and len(QUESTION_TO_ANSWER) > 0:
        try:
            choices = list(QUESTION_TO_ANSWER.keys())
            target = key_short or key_full
            best = process.extractOne(target, choices, scorer=fuzz.token_sort_ratio)
            if best and best[1] >= FUZZY_MIN_SCORE:
                matched_key = best[0]
                item = QUESTION_TO_ANSWER.get(matched_key)
                if item:
                    raw_answer = item.get("answer") or ""
                    user_answer = naturalize_answer(raw_answer, message)
                    messages = [{"role": "assistant", "id": f"a_{uuid.uuid4().hex[:8]}", "text": user_answer}]
                    retrieved = [{"index": item.get("index"), "score": float(best[1]) / 100.0, "question": message, "answer": raw_answer}]
                    return jsonify(session_id=session_id, messages=messages, retrieved=retrieved)
        except Exception:
            traceback.print_exc()

    # retriever fallback
    retrieved = []
    if RETRIEVER:
        try:
            if category and hasattr(RETRIEVER, "retrieve_by_category"):
                try:
                    retrieved = RETRIEVER.retrieve_by_category(category, proc_text, k=top_k)
                except TypeError:
                    retrieved = RETRIEVER.retrieve_by_category(category, proc_text)
            elif hasattr(RETRIEVER, "get_similar_questions"):
                retrieved = RETRIEVER.get_similar_questions(proc_text, k=top_k)
            elif hasattr(RETRIEVER, "retrieve"):
                try:
                    retrieved = RETRIEVER.retrieve(proc_text, top_k)
                except TypeError:
                    retrieved = RETRIEVER.retrieve(proc_text, k=top_k)
        except Exception:
            traceback.print_exc()
            retrieved = []

    # normalize retrieved objects
    norm = []
    for r in (retrieved or []):
        if isinstance(r, dict):
            norm.append(r)
        else:
            try:
                norm.append(dict(r))
            except Exception:
                norm.append({"text": str(r)})
    retrieved = norm

    # determine top retrieved score
    top_score = None
    top_answer_text = ""
    if retrieved:
        top = retrieved[0]
        if "score" in top:
            try:
                top_score = float(top.get("score", 0.0))
            except Exception:
                top_score = 0.0
        elif "distance" in top:
            try:
                d = float(top.get("distance", 1.0) or 1.0)
                top_score = max(0.0, 1.0 - d)
            except Exception:
                top_score = 0.0
        top_answer_text = top.get("answer") or top.get("answer_text") or top.get("text") or ""

    # heuristic: check OOC keywords
    lowered = (message or "").lower()
    if any(k in lowered for k in OOC_KEYWORDS):
        # immediately treat as OOC
        if escalate:
            return jsonify(session_id=session_id, ooc=True,
                           messages=[{"text": "Maaf, pertanyaan ini di luar cakupan FAQ. Silakan hubungi admin untuk bantuan lebih lanjut."}],
                           wa_link=make_wa_link(ADMIN_WA, f"Permintaan bantuan: {message}"))
        else:
            return jsonify(session_id=session_id, ooc=True,
                           messages=[{"text": "Maaf, pertanyaan ini di luar cakupan FAQ."}])

    # If top score exists and is >= threshold -> answer
    if top_score is not None and top_score >= OOC_SCORE_THRESHOLD and top_answer_text:
        user_answer = naturalize_answer(top_answer_text, message)
        messages = [{"role": "assistant", "id": f"a_{uuid.uuid4().hex[:8]}", "text": user_answer}]
        return jsonify(session_id=session_id, messages=messages, retrieved=retrieved)

    # If generator available and allowed, try generator but be conservative
    answer_text = ""
    gen_sources = []
    if GENERATOR and hasattr(GENERATOR, "generate_answer"):
        try:
            gen = GENERATOR.generate_answer(retrieved, proc_text, session_id=session_id)
            if isinstance(gen, dict):
                answer_text = gen.get("text") or gen.get("answer") or ""
                gen_sources = gen.get("sources") or []
            else:
                answer_text = str(gen)
        except Exception:
            traceback.print_exc()
            answer_text = ""

    # If we got a generator answer but confidence is low (or no retrieved), we should treat cautiously
    if answer_text:
        # If there was no retrieved or low top_score, treat generator answer as NOT authoritative:
        if top_score is None or top_score < OOC_SCORE_THRESHOLD:
            # decide to escalate or not
            if escalate:
                return jsonify(session_id=session_id, ooc=True,
                               messages=[{"text": "Maaf, saya belum menemukan sumber terpercaya di FAQ untuk menjawab itu. Silakan hubungi admin untuk bantuan lebih lanjut."}],
                               wa_link=make_wa_link(ADMIN_WA, f"Permintaan bantuan (gen fallback): {message}"),
                               sources=gen_sources)
            else:
                # return polite generated text but mark as unverified (optional)
                user_answer = naturalize_answer(answer_text, message)
                messages = [{"role": "assistant", "id": f"a_{uuid.uuid4().hex[:8]}", "text": user_answer}]
                return jsonify(session_id=session_id, messages=messages, retrieved=retrieved, sources=gen_sources)

    # If we reached here and there's a top retrieved answer (but low confidence), offer escalation
    if retrieved and top_answer_text:
        if escalate:
            return jsonify(session_id=session_id, ooc=True,
                           messages=[{"text": "Maaf, saya tidak cukup yakin dengan jawaban di FAQ. Jika Anda ingin, silakan hubungi admin untuk konfirmasi."}],
                           wa_link=make_wa_link(ADMIN_WA, f"Permintaan verifikasi jawaban: {message}"),
                           retrieved=retrieved)
        else:
            user_answer = naturalize_answer(top_answer_text, message)
            messages = [{"role": "assistant", "id": f"a_{uuid.uuid4().hex[:8]}", "text": user_answer}]
            return jsonify(session_id=session_id, messages=messages, retrieved=retrieved)

    # ultimate fallback: no answer at all
    if escalate:
        return jsonify(session_id=session_id, ooc=True,
                       messages=[{"text": "Maaf, saya belum menemukan informasi yang relevan dalam FAQ. Silakan hubungi admin untuk bantuan."}],
                       wa_link=make_wa_link(ADMIN_WA, f"Permintaan bantuan: {message}"))
    else:
        return jsonify(session_id=session_id, messages=[{"text": "Maaf, saya belum menemukan informasi yang relevan dalam FAQ."}])

# -----------------------
# Run
# -----------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=True)
