# src/embeddings_store.py
import os
import json
from typing import List, Optional

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from .config import EMBEDDING_MODEL, EMBEDDING_DIM, INDEX_PATH, EMBEDDINGS_NPY, METADATA_JSON


class EmbeddingStore:
    """
    Helper untuk membuat, menyimpan, memuat, dan query FAISS index menggunakan
    sentence-transformers sebagai embedding model.

    Cara pakai singkat:
      store = EmbeddingStore()                     # load model
      embeddings = store.build_index(texts)        # texts: list[str]
      store.metadata = metadata                    # metadata sesuai urutan texts
      store.save(embeddings, metadata)

      # lalu di proses selanjutnya:
      store.load()
      indices, scores = store.query("pertanyaan kamu", top_k=5)
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL, dim: int = EMBEDDING_DIM, batch_size: int = 32):
        self.model_name = model_name
        self.dim = dim
        self.batch_size = int(batch_size)

        # inisialisasi model (bisa heavy -> lakukan sekali)
        try:
            self.model = SentenceTransformer(self.model_name)
        except Exception as e:
            raise RuntimeError(f"Gagal memuat model sentence-transformers '{self.model_name}': {e}")

        self.index: Optional[faiss.Index] = None
        self.metadata: List[dict] = []

    def _encode_batch(self, texts: List[str]) -> np.ndarray:
        """
        Encode list teks menjadi numpy array embeddings (float32).
        Memakai batching untuk mengurangi memory usage.
        """
        all_embs = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            emb = self.model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
            all_embs.append(emb)
        embeddings = np.vstack(all_embs).astype("float32")
        return embeddings

    def build_index(self, texts: List[str], normalize: bool = True) -> np.ndarray:
        """
        Bangun embeddings dari list teks dan buat FAISS IndexFlatIP.
        - normalize: jika True, lakukan L2 normalize sehingga Inner Product ~ Cosine.
        Mengembalikan embeddings numpy array (N x dim).
        """
        if not isinstance(texts, (list, tuple)):
            raise ValueError("Parameter 'texts' harus list[str].")

        embeddings = self._encode_batch(texts)

        # cek dimensi
        if embeddings.shape[1] != self.dim:
            # kompatibilitas: beberapa model produce dim berbeda dari EMBEDDING_DIM
            actual = embeddings.shape[1]
            raise RuntimeError(f"Dimensi embedding mismatch: expected {self.dim}, got {actual}. "
                               f"Ubah EMBEDDING_DIM di config atau pilih model lain.")

        if normalize:
            faiss.normalize_L2(embeddings)

        # build index (Inner Product) untuk cosine similarity jika sudah dinormalisasi
        index = faiss.IndexFlatIP(self.dim)
        index.add(embeddings)
        self.index = index
        return embeddings

    def build_index_from_df(self, df, text_col: str = "combined_text", normalize: bool = True) -> np.ndarray:
        """
        Convenience: build index langsung dari DataFrame kolom text_col.
        Pastikan df[text_col] ada.
        """
        if text_col not in df.columns:
            raise ValueError(f"Column '{text_col}' tidak ditemukan di DataFrame.")
        texts = df[text_col].fillna("").astype(str).tolist()
        return self.build_index(texts, normalize=normalize)

    def save(self, embeddings: np.ndarray, metadata: List[dict]):
        """
        Simpan embeddings, metadata, dan FAISS index ke disk.
        - embeddings: numpy array (N x dim)
        - metadata: list of dict dengan panjang sama dengan embeddings
        """
        if embeddings is None:
            raise ValueError("Embeddings kosong. Panggil build_index() dulu.")
        if len(metadata) != embeddings.shape[0]:
            raise ValueError("Panjang metadata harus sama dengan jumlah embeddings.")

        # simpan embeddings dan metadata
        np.save(EMBEDDINGS_NPY, embeddings)
        with open(METADATA_JSON, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        # cek index dan tulis binary
        if self.index is None:
            # jika index belum dibuat, buat index dari embeddings (tanpa normalisasi tambahan)
            idx = faiss.IndexFlatIP(self.dim)
            idx.add(embeddings)
            self.index = idx

        faiss.write_index(self.index, INDEX_PATH)

    def load(self):
        """
        Muat index FAISS, embeddings dan metadata dari disk.
        Mengembalikan tuple (embeddings, metadata).
        """
        if not os.path.exists(INDEX_PATH):
            raise FileNotFoundError(f"Index not found: {INDEX_PATH}")
        if not os.path.exists(EMBEDDINGS_NPY):
            raise FileNotFoundError(f"Embeddings file not found: {EMBEDDINGS_NPY}")
        if not os.path.exists(METADATA_JSON):
            raise FileNotFoundError(f"Metadata file not found: {METADATA_JSON}")

        self.index = faiss.read_index(INDEX_PATH)
        embeddings = np.load(EMBEDDINGS_NPY)
        with open(METADATA_JSON, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        return embeddings, self.metadata

    def query(self, text: str, top_k: int = 5) -> (List[int], List[float]):
        """
        Query index dengan teks: encode -> normalize -> search top_k.
        Mengembalikan (indices, scores).
        NOTE: index harus sudah di-load atau di-build.
        """
        if self.index is None:
            raise RuntimeError("Index belum diload. Panggil load() atau build_index() dulu.")

        q_emb = self.model.encode([text], convert_to_numpy=True)
        # pastikan float32
        q_emb = q_emb.astype("float32")
        faiss.normalize_L2(q_emb)
        D, I = self.index.search(q_emb, top_k)
        indices = I[0].tolist()
        scores = D[0].tolist()
        return indices, scores


if __name__ == "__main__":
    # Quick demo: jalankan dari root project: python -m src.embeddings_store
    print("Demo EmbeddingStore (quick test). Pastikan src.preprocessing dan config terpasang.")
    try:
        from .preprocessing import Preprocessor
    except Exception:
        # relative import fail jika dijalankan tidak sebagai package; fallback import absolute
        from src.preprocessing import Preprocessor

    pre = Preprocessor()
    df = pre.preprocess()
    texts = pre.get_combined_texts()

    print(f"Building embeddings for {len(texts)} texts using model {EMBEDDING_MODEL} ...")
    store = EmbeddingStore()
    emb = store.build_index(texts)
    meta = [{"index": i, "question": row["question"], "answer": row["answer"]} for i, row in df.iterrows()]
    store.metadata = meta
    store.save(emb, meta)
    print("Done. Files saved:", INDEX_PATH, EMBEDDINGS_NPY, METADATA_JSON)
