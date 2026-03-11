# src/app.py
import os
from .preprocessing import Preprocessor
from .embedding_store import EmbeddingStore
from .retriever import Retriever
from .generator import GeminiGenerator
from .config import INDEX_PATH, EMBEDDINGS_NPY, METADATA_JSON, TOP_K

SHOW_DEBUG = False


def build_index_if_needed(df):
    emb_store = EmbeddingStore()

    if (
        os.path.exists(INDEX_PATH)
        and os.path.exists(EMBEDDINGS_NPY)
        and os.path.exists(METADATA_JSON)
    ):
        try:
            emb_store.load()
            print("Memuat index FAISS yang sudah ada")
            return emb_store
        except Exception as e:
            print("Gagal memuat index (akan rebuild):", e)

    # Jika dataset punya combined_text, pakai itu
    if "combined_text" in df.columns:
        texts = df["combined_text"].fillna("").tolist()
    else:
        # Default: gabungkan question + answer
        texts = (df["question"].fillna("") + " " + df["answer"].fillna("")).tolist()

    embeddings = emb_store.build_index(texts)

    metadata = [
        {"index": int(i), "question": row["question"], "answer": row["answer"]}
        for i, row in df.iterrows()
    ]

    emb_store.metadata = metadata
    emb_store.save(embeddings, metadata)

    print("Index dibuat dan disimpan")
    return emb_store


def main_cli():
    pre = Preprocessor()
    df = pre.preprocess()

    emb_store = build_index_if_needed(df)
    retriever = Retriever(df, emb_store)
    generator = GeminiGenerator()  # <-- pakai generator baru

    print("CLI RAG Chatbot (ketik 'exit' untuk keluar)")

    while True:
        user = input("\nAnda: ").strip()
        if user.lower() == "exit":
            break

        # Voice support
        if user.startswith("voice:"):
            audio_path = user.split(":", 1)[1].strip()
            try:
                from .voice import VoiceNote
                vn = VoiceNote()
                user = vn.transcribe(audio_path)
                print("[voice->text]:", user)
            except Exception as e:
                print("Gagal transcribe voice:", e)
                continue

        # Retrieve candidates
        candidates = retriever.retrieve(user, top_k=TOP_K)
        if not candidates:
            print("Maaf, saya tidak menemukan informasi terkait.")
            continue

        # Ambil jawaban sumber terbaik
        best_answer = candidates[0]["answer"]
        contexts = [best_answer]  # hanya kirim jawaban untuk diparafrase

        # Panggil Gemini untuk membuat parafrase
        try:
            gen_reply = generator.generate(
            user_query=user,
            contexts=contexts,
            paraphrase=True,
            paraphrase_strength="medium",
            style="formal",
            max_output_tokens=512,   # naikkan dari 240 -> 512
            )

        except Exception as e:
            print("Generator error:", e)
            gen_reply = "[UNAVAILABLE]"

        # Debug
        if SHOW_DEBUG:
            print("\n[DEBUG raw Gemini reply]:")
            print(gen_reply)
            print("\n----- END DEBUG -----\n")

        # Fallback jika terjadi error / tidak ada teks
        if (
            not isinstance(gen_reply, str)
            or gen_reply.startswith("[ERROR")
            or gen_reply.startswith("[UNAVAILABLE")
            or gen_reply.startswith("[DEBUG")
        ):
            final = best_answer  # pakai FAQ asli
        else:
            final = gen_reply.strip() or best_answer

        print("\nBot:")
        print(final)


if __name__ == "__main__":
    main_cli()
