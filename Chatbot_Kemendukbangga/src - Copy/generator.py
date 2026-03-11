# src/generator.py
import traceback
from .config import GEMINI_API_KEY, GEMINI_MODEL, DEBUG

# Load Google GenAI SDK
try:
    import google.generativeai as genai
    from google.generativeai import GenerationConfig
    SDK_AVAILABLE = True
except Exception:
    genai = None
    GenerationConfig = None
    SDK_AVAILABLE = False

# Inisialisasi model sekali saja
_gemini_model = None
if SDK_AVAILABLE and GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        _gemini_model = genai.GenerativeModel(GEMINI_MODEL or "gemini-2.5-flash")
    except Exception as e:
        if DEBUG: print("Gagal inisialisasi Gemini:", e)
        _gemini_model = None

# =====================================================================
# 🔹 FUNGSI UTAMA: generate_with_gemini (versi sederhana & kuat)
# =====================================================================
def generate_with_gemini(prompt: str, max_output_tokens=300, temperature=0.75):
    """
    Menghasilkan teks dari Gemini dengan cara paling sederhana,
    namun aman dari error (fallback jika resp.text kosong).
    """

    if not SDK_AVAILABLE or _gemini_model is None:
        return "⚠️ Model Gemini belum siap. Periksa API key / SDK."

    # Bungkus prompt menjadi messages
    messages = [{"role": "user", "parts": [{"text": prompt}]}]

    # Build config
    try:
        gen_cfg = GenerationConfig(
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            top_p=0.95
        )
    except Exception:
        gen_cfg = {"max_output_tokens": max_output_tokens, "temperature": temperature}

    # ------------------------------
    # 1) Coba generate_content()
    # ------------------------------
    try:
        resp = _gemini_model.generate_content(messages, generation_config=gen_cfg)

        # a) resp.text jika ada
        try:
            if getattr(resp, "text", None):
                return resp.text.strip()
        except:
            pass

        # b) coba ambil dari parts
        res = getattr(resp, "result", None)
        if res and hasattr(res, "candidates"):
            for cand in res.candidates:
                content = getattr(cand, "content", None)
                if content:
                    parts = getattr(content, "parts", None)
                    if parts:
                        texts = [getattr(p, "text", "") for p in parts if getattr(p, "text", None)]
                        if texts:
                            return "\n".join(texts).strip()

        # c) resp.output
        out = getattr(resp, "output", None)
        if out:
            return str(out).strip()

    except Exception as e:
        if DEBUG:
            print("Error generate_content:", e)
            traceback.print_exc()

    # ------------------------------
    # 2) Fallback: genai.generate()
    # ------------------------------
    try:
        fallback = genai.generate(
            model=GEMINI_MODEL,
            input=prompt,
            max_output_tokens=max_output_tokens,
            temperature=temperature
        )
        text = getattr(fallback, "text", None) or getattr(fallback, "output", None)
        if text:
            return str(text).strip()

    except Exception as e:
        if DEBUG:
            print("Fallback error:", e)

    return "⚠️ Gemini tidak mengembalikan teks."


# =====================================================================
# 🔹 FUNGSI CHATBOT (menggunakan retrieval lokal + prompt Indonesia)
# =====================================================================
def chatbot_response(user_query, collection, embed_model):

    # Import fungsi koreksi & retrieval
    from .preprocessing import correct_typo
    from .retriever import retrieve_answer_local

    # Koreksi ejaan
    corrected = correct_typo(user_query)
    if corrected != user_query.lower():
        print(f"🔧 Koreksi ejaan: '{user_query}' → '{corrected}'")

    # Ambil dokumen FAQ terdekat
    top_docs, top_scores = retrieve_answer_local(corrected, collection, embed_model)

    if not top_docs:
        print("❌ Maaf, saya belum menemukan jawaban yang sesuai.")
        return

    # Gunakan jawaban FAQ teratas
    faq_text = top_docs[0].strip()

    # Ringkas FAQ jika terlalu panjang
    if len(faq_text) > 900:
        faq_text = faq_text[:890].rsplit(" ", 1)[0] + "..."

    # =========================
    # PROMPT BAHASA INDONESIA
    # =========================
    prompt = f"""
Anda adalah asisten AI yang ramah dan sopan.

Pertanyaan pengguna:
{user_query}

Berdasarkan informasi FAQ berikut:
{faq_text}

Tulis jawaban dengan bahasa alami, sopan, mudah dipahami, dan panjang 1–6 kalimat.
Pastikan makna tetap sesuai FAQ, namun ditulis dengan kata-kata sendiri.

Tambahkan penutup singkat:
"Semoga penjelasan ini membantu ya."
"""

    # Panggil Gemini
    response = generate_with_gemini(prompt)

    print(f"\n📌 Jawaban Chatbot:\n{response}\n")


# ========================================================
# --- compatibility wrapper untuk kode lama yang mengharapkan GeminiGenerator ---
# ========================================================
class GeminiGenerator:
    """
    Wrapper kompatibilitas agar app.py yang lama tetap bisa memakai GeminiGenerator.
    Internally memanggil fungsi generate_with_gemini(prompt, ...).
    """

    def __init__(self, api_key: str = None, model: str = None):
        # Inisialisasi ringan — modul-level sudah mengonfigurasi genai/model.
        self.api_key = api_key
        self.model = model

    def _build_prompt_from_contexts(self, user_query: str, contexts):
        """
        Gabungkan contexts (list of strings) menjadi blok Sumber yang ringkas,
        lalu bangun prompt Bahasa Indonesia yang singkat.
        """
        sumber = ""
        if contexts:
            if isinstance(contexts, (list, tuple)):
                sumber = "\n\n".join([c.strip() for c in contexts if c])
            else:
                sumber = str(contexts).strip()

        # Pastikan sumber tidak terlalu panjang
        if len(sumber) > 1200:
            sumber = sumber[:1197].rsplit(" ", 1)[0] + "..."

        prompt = f"""Anda adalah asisten AI yang ramah dan sopan.

Pertanyaan pengguna:
{user_query}

Berdasarkan sumber berikut:
{sumber}

Tuliskan jawaban dengan bahasa yang natural, sopan, mudah dipahami, dan panjang 1–6 kalimat.
Pastikan makna tetap sesuai sumber, gunakan kata-kata sendiri.
Tambahkan penutup singkat: "Semoga penjelasan ini membantu ya."
"""
        return prompt

    def generate(
        self,
        user_query: str,
        contexts=None,
        paraphrase: bool = True,
        paraphrase_strength: str = "high",
        style: str = "formal",
        max_output_tokens: int = 320,
    ) -> str:
        """
        API compat: menerima parameter seperti sebelumnya, lalu memanggil generate_with_gemini.
        """
        try:
            prompt = self._build_prompt_from_contexts(user_query, contexts)
            # map paraphrase_strength -> temperature (mirip behavior sebelumnya)
            temp_map = {"low": 0.2, "medium": 0.45, "high": 0.8}
            temperature = temp_map.get(paraphrase_strength, 0.75)

            # panggil helper yang sudah ada di modul
            result = generate_with_gemini(prompt, max_output_tokens=max_output_tokens, temperature=temperature)
            # pastikan string
            if not isinstance(result, str):
                return str(result or "")
            return result
        except Exception as e:
            if DEBUG:
                import traceback
                traceback.print_exc()
            return "[UNAVAILABLE]"

    # Optional: jika ada kode yang memanggil generate_content pada instance
    def generate_content(self, messages, generation_config=None, safety_settings=None):
        """
        Minimal adapter: ubah messages -> prompt sederhana lalu panggil generate_with_gemini.
        Ini membuat kode lama yang langsung memanggil generate_content tetap bekerja.
        """
        try:
            # flatten messages -> ambil semua parts[].text
            pieces = []
            try:
                for m in messages:
                    # messages bisa berupa dict-like atau objects; coba akses keys aman
                    if isinstance(m, dict):
                        parts = m.get("parts", [])
                        for p in parts:
                            if isinstance(p, dict):
                                t = p.get("text")
                            else:
                                t = getattr(p, "text", None)
                            if t:
                                pieces.append(str(t))
                    else:
                        # coba atribut .parts jika object
                        parts = getattr(m, "parts", None)
                        if parts:
                            for p in parts:
                                t = getattr(p, "text", None)
                                if t:
                                    pieces.append(str(t))
            except Exception:
                pieces = [str(messages)]

            prompt = "\n\n".join(pieces) if pieces else str(messages)
            max_tokens = 320
            try:
                if generation_config is not None:
                    # try to read attribute
                    max_tokens = int(getattr(generation_config, "max_output_tokens", max_tokens))
            except Exception:
                pass

            return generate_with_gemini(prompt, max_output_tokens=max_tokens)
        except Exception:
            if DEBUG:
                import traceback
                traceback.print_exc()
            return "[UNAVAILABLE]"
