# src/uji_gemini_debug.py
import os, pathlib, json
from dotenv import load_dotenv
BASE_DIR = pathlib.Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / ".env")

import google.generativeai as genai
from google.generativeai import GenerationConfig
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
print("Using model:", MODEL)

model_obj = genai.GenerativeModel(MODEL)

source_answer = ("DASHAT adalah singkatan dari Dapur Sehat Atasi Stunting. "
    "Merupakan kegiatan pemberdayaan masyarakat dalam upaya pemenuhan gizi seimbang bagi "
    "keluarga beresiko stunting melalui sumberdaya lokal yang dipadukan dengan sumberdaya/kontribusi kemitraan lainnya."
)

prompt = (
    "TUGAS: Parafrase (tulis ulang) teks SOURCE_ANSWER berikut dalam bahasa Indonesia "
    "menggunakan kata-kata sendiri, tetap setia pada fakta, jangan menambah informasi. "
    "Buat hasil natural, sopan, 2–6 kalimat, dan tambahkan penutup singkat 'Semoga penjelasan ini membantu ya.'\n\n"
    f"SOURCE_ANSWER:\n{source_answer}\n\nParafrase:"
)

messages = [{"role":"user","parts":[{"text":prompt}]}]
gen_cfg = GenerationConfig(max_output_tokens=320, temperature=0.75, top_p=0.95)

print("Calling generate_content ...")
try:
    resp = model_obj.generate_content(messages, generation_config=gen_cfg)
    print("TYPE:", type(resp))
    # print repr
    print("repr(resp):", repr(resp)[:1000])
    # try quick text accessor safely
    try:
        print("resp.text:", resp.text)
    except Exception as e:
        print("resp.text failed:", e)
    print("resp.output:", getattr(resp, "output", None))
    # if output is list, print first candidate details
    out = getattr(resp, "output", None)
    if out:
        try:
            print("OUTPUT raw (len):", len(out))
        except:
            pass
    # print as JSON if possible
    try:
        j = json.loads(str(resp))
        print("JSON dump:", json.dumps(j, indent=2)[:2000])
    except Exception:
        pass
except Exception as e:
    print("generate_content raised:", e)
