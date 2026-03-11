# src/generator.py
"""
Safe Gemini generator for FAQ-based chatbot.

Perilaku utama:
- Menerima `retrieved` (list[dict]) dan `query` -> menghasilkan jawaban ter-grounded.
- Jika `retrieved` kosong -> kembalikan fallback yang sopan.
- Gunakan Gemini (google.generativeai) jika tersedia; kalau gagal gunakan deterministic fallback.
- Lakukan post-check sederhana: jika jawaban model mengandung banyak token
  yang TIDAK ditemukan di konteks retrieved, anggap model berpotensi "halusinasi"
  dan fallback ke jawaban deterministik (top retrieved).
"""

import re
import textwrap
import traceback
from typing import List, Dict, Any, Optional

# config values (user project should provide these in src/config.py)
try:
    from .config import GEMINI_API_KEY, GEMINI_MODEL, DEBUG
except Exception:
    GEMINI_API_KEY = None
    GEMINI_MODEL = None
    DEBUG = False

# Try load Google GenAI SDK
try:
    import google.generativeai as genai
    from google.generativeai import GenerationConfig
    SDK_AVAILABLE = True
except Exception:
    genai = None
    GenerationConfig = None
    SDK_AVAILABLE = False

# initialize model at module load (safe)
_gemini_model = None
if SDK_AVAILABLE and GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        # note: we won't store instance unless needed; use genai.generate in call_model
        _gemini_model = GEMINI_MODEL or "gemini-2.5-flash"
    except Exception as e:
        if DEBUG: print("Gemini init error:", e)
        _gemini_model = None

# -------------------------
# Helpers
# -------------------------
def _build_context_text(retrieved: List[Dict[str, Any]], max_chars: int = 4000) -> str:
    """
    Gabungkan retrieved passages jadi satu teks konteks ringkas.
    Potong jika terlalu panjang.
    """
    parts = []
    for r in retrieved:
        q = r.get("question", "")
        a = r.get("answer") or r.get("answer_text") or r.get("text") or ""
        idx = r.get("index", "")
        snippet = f"[Sumber {idx}] Q: {q.strip()}\nA: {a.strip()}"
        parts.append(snippet)
    context = "\n\n".join(parts)
    if len(context) > max_chars:
        context = context[: max_chars - 3].rsplit(" ", 1)[0] + "..."
    return context

def _build_prompt(retrieved: List[Dict[str, Any]], query: str) -> str:
    """
    Template prompt yang ketat: minta model hanya menggunakan konteks, jangan mengarang.
    Bahasa: Indonesia.
    """
    system_instructions = textwrap.dedent("""
    Kamu adalah asisten FAQ resmi. Jawab hanya menggunakan informasi yang ada dalam bagian KONTEXT di bawah.
    Jangan menambahkan informasi, data, angka, atau klaim yang tidak ada di konteks.
    Jika konteks tidak memuat jawaban, katakan: "Maaf, saya belum menemukan informasi tersebut dalam FAQ."
    Jawaban boleh menggunakan gaya yang ramah dan jelas, singkat (1-6 kalimat).
    Bila jawaban berasal dari sumber dalam konteks, tambahkan referensi singkat seperti "[Sumber <index>]".
    """).strip()

    context = _build_context_text(retrieved)
    prompt = f"{system_instructions}\n\nPertanyaan: {query}\n\nKONTEXT:\n{context}\n\nBuat jawaban singkat (1-6 kalimat)."
    return prompt

def _tokenize_significant(s: str) -> List[str]:
    """
    Ambil token penting: kata >=4 huruf atau angka (mengurangi kata hubung kecil).
    Semua lowercased.
    """
    if not s:
        return []
    s = s.lower()
    # ambil kata alfanumerik
    words = re.findall(r"[a-z0-9]+", s)
    sig = [w for w in words if len(w) >= 4 or re.search(r"\d", w)]
    return sig

def _post_check_answer(answer: str, retrieved: List[Dict[str, Any]], min_overlap_ratio: float = 0.4) -> bool:
    """
    Periksa apakah jawaban 'answer' cukup grounded oleh 'retrieved'.
    - hitung token penting jawaban dan bandingkan berapa banyak muncul di teks konteks.
    - jika rasio token yang ditemukan >= min_overlap_ratio => pass
    - jika tidak => fail (kemungkinan hallucination)
    Return True jika pass (aman), False jika fail.
    """
    try:
        answer_tokens = _tokenize_significant(answer)
        if not answer_tokens:
            # tidak ada token penting (jawaban sangat pendek) -> anggap aman
            return True

        context_text = " ".join([ (r.get("answer") or r.get("answer_text") or r.get("text") or "") for r in retrieved ])
        context_text = context_text.lower()

        found = 0
        for t in answer_tokens:
            if t in context_text:
                found += 1

        ratio = found / len(answer_tokens)
        if DEBUG:
            print(f"[post_check] tokens={len(answer_tokens)} found={found} ratio={ratio:.2f}")
        return ratio >= min_overlap_ratio
    except Exception:
        return False

def _top_retrieved_answer(retrieved: List[Dict[str, Any]]) -> str:
    """
    Ambil jawaban deterministik dari top-1 retrieved item.
    """
    if not retrieved:
        return "Maaf, saya belum menemukan informasi tersebut dalam FAQ."
    top = retrieved[0]
    ans = top.get("answer") or top.get("answer_text") or top.get("text") or ""
    ans = (ans or "").strip()
    if ans:
        return ans
    return "Maaf, saya belum menemukan informasi tersebut dalam FAQ."

# -------------------------
# Model call wrapper
# -------------------------
def call_gemini_model(prompt: str, temperature: float = 0.0, max_output_tokens: int = 256) -> str:
    """
    Panggil Google Gemini via google.generativeai SDK (genai).
    Jika SDK tidak tersedia atau error, raise Exception.
    Returns generated text (string).
    """
    if not SDK_AVAILABLE or _gemini_model is None:
        raise RuntimeError("Gemini SDK atau model tidak tersedia")

    try:
        # prefer genai.generate (simple)
        # use generation config where available
        if GenerationConfig is not None:
            gen_cfg = GenerationConfig(max_output_tokens=max_output_tokens, temperature=temperature)
            resp = genai.generate(model=_gemini_model, prompt=prompt, temperature=temperature, max_output_tokens=max_output_tokens)
        else:
            # fallback older API shape
            resp = genai.generate(model=_gemini_model, input=prompt, max_output_tokens=max_output_tokens, temperature=temperature)

        # resp may have .text or structure; try common places
        text = getattr(resp, "text", None)
        if not text:
            # some SDK returns output/candidates
            out = getattr(resp, "output", None) or getattr(resp, "result", None)
            if out:
                # try to extract text fields
                if isinstance(out, str):
                    text = out
                else:
                    # try common nested fields
                    cand = None
                    if hasattr(out, "candidates"):
                        cand = out.candidates[0]
                    elif isinstance(out, dict) and "candidates" in out:
                        cand = out["candidates"][0]
                    if cand:
                        # try various attribute names
                        text = getattr(cand, "content", None) or getattr(cand, "text", None) or (cand.get("content") if isinstance(cand, dict) else None)
                        if isinstance(text, dict) and "parts" in text:
                            parts = text.get("parts", [])
                            text = " ".join([p.get("text","") for p in parts if isinstance(p, dict)])
        if not text:
            # as last resort stringify resp
            text = str(resp)
        return str(text).strip()
    except Exception as e:
        if DEBUG:
            print("call_gemini_model error:", e)
            traceback.print_exc()
        raise

# -------------------------
# Public API: generate_answer
# -------------------------
class GeminiGenerator:
    """
    Generator yang dipakai server:
    - generate_answer(retrieved, query, session_id) -> dict {"text":..., "sources":[...]}
    """

    def __init__(self, model_name: Optional[str] = None, temperature: float = 0.0, max_tokens: int = 300):
        self.model_name = model_name or _gemini_model
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)

    def generate_answer(self, retrieved: List[Dict[str, Any]], query: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Main entry:
        - jika retrieved kosong -> fallback
        - buat prompt ketat, panggil model, lakukan post-check
        - jika post-check gagal -> fallback deterministik top retrieved
        """
        try:
            # 1) if no retrieved
            if not retrieved:
                return {"text": "Maaf, saya belum menemukan informasi tersebut dalam FAQ.", "sources": []}

            # 2) prepare prompt & context
            prompt = _build_prompt(retrieved, query)

            # 3) call model safely
            model_text = None
            try:
                model_text = call_gemini_model(prompt, temperature=self.temperature, max_output_tokens=self.max_tokens)
            except Exception:
                # gemini not available or failed; fallback to top retrieved
                if DEBUG:
                    print("Gemini call failed, falling back to top retrieved.")
                fallback = _top_retrieved_answer(retrieved)
                return {"text": fallback, "sources": [retrieved[0].get("index")]}

            # 4) post-check answer grounding
            safe = _post_check_answer(model_text, retrieved, min_overlap_ratio=0.4)
            if safe:
                # optionally append source references: pick top 1-3 indices
                srcs = []
                for r in retrieved[:3]:
                    idx = r.get("index")
                    if idx is not None:
                        srcs.append(idx)
                return {"text": model_text, "sources": srcs}
            else:
                # if model likely hallucinated, fallback to deterministic answer
                if DEBUG:
                    print("Model answer failed post-check, returning top retrieved instead.")
                fallback = _top_retrieved_answer(retrieved)
                return {"text": fallback, "sources": [retrieved[0].get("index")]}

        except Exception as e:
            if DEBUG:
                traceback.print_exc()
            # ultimate fallback
            fallback = _top_retrieved_answer(retrieved) if retrieved else "Maaf, saya belum menemukan informasi tersebut dalam FAQ."
            return {"text": fallback, "sources": [retrieved[0].get("index")] if retrieved else []}


# Backwards-compatible convenience function (optional)
def generate_with_gemini_from_retrieved(retrieved: List[Dict[str, Any]], query: str, temperature: float = 0.0, max_output_tokens: int = 300) -> Dict[str, Any]:
    gen = GeminiGenerator(temperature=temperature, max_tokens=max_output_tokens)
    return gen.generate_answer(retrieved, query)
