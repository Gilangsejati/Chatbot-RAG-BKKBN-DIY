# src/embedding_store.py
import os
import json
from typing import List, Optional, Tuple

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from .config import EMBEDDING_MODEL, EMBEDDING_DIM, INDEX_PATH, EMBEDDINGS_NPY, METADATA_JSON


class EmbeddingStore:
    """
    Helper untuk membuat, menyimpan, memuat, dan query FAISS index menggunakan
    sentence-transformers sebagai embedding model.

    Methods:
      - build_index(texts) -> np.ndarray (embeddings)
      - build_index_from_df(df, text_col="combined_text")
      - build_and_save_from_df(df, text_col="combined_text", metadata_key_cols=("question","answer","kategori"))
      - save(embeddings, metadata)
      - load() -> (embeddings, metadata)
      - query(text, top_k)
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL, dim: int = EMBEDDING_DIM, batch_size: int = 32):
        self.model_name = model_name
        self.dim = int(dim)
        self.batch_size = int(batch_size)

        # load model (may be heavy)
        try:
            self.model = SentenceTransformer(self.model_name)
        except Exception as e:
            raise RuntimeError(f"Gagal memuat model sentence-transformers '{self.model_name}': {e}")

        self.index: Optional[faiss.Index] = None
        self.metadata: List[dict] = []

    def _encode_batch(self, texts: List[str]) -> np.ndarray:
        all_embs = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i: i + self.batch_size]
            emb = self.model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
            all_embs.append(emb)
        if not all_embs:
            return np.zeros((0, self.dim), dtype="float32")
        embeddings = np.vstack(all_embs).astype("float32")
        return embeddings

    def build_index(self, texts: List[str], normalize: bool = True) -> np.ndarray:
        if not isinstance(texts, (list, tuple)):
            raise ValueError("Parameter 'texts' harus list[str].")
        embeddings = self._encode_batch(texts)
        if embeddings.shape[0] == 0:
            raise ValueError("Tidak ada teks untuk di-embed.")
        # adapt dim
        actual_dim = embeddings.shape[1]
        if actual_dim != self.dim:
            # update dim to actual (helpful bila config mismatch)
            self.dim = actual_dim

        if normalize:
            faiss.normalize_L2(embeddings)

        index = faiss.IndexFlatIP(self.dim)
        index.add(embeddings)
        self.index = index
        return embeddings

    def build_index_from_df(self, df, text_col: str = "combined_text", normalize: bool = True) -> np.ndarray:
        if text_col not in df.columns:
            raise ValueError(f"Column '{text_col}' tidak ditemukan di DataFrame.")
        texts = df[text_col].fillna("").astype(str).tolist()
        return self.build_index(texts, normalize=normalize)

    def build_and_save_from_df(self, df, question_col: str = "question", answer_col: str = "answer",
                               category_col_candidates: List[str] = ("kategori", "category"),
                               text_col: str = "combined_text", normalize: bool = True):
        """
        Build metadata list from df and create index + embeddings + save files.
        Ensures metadata contains 'index', 'question', 'answer', 'category'.
        """
        # build texts
        if text_col in df.columns:
            texts = df[text_col].fillna("").astype(str).tolist()
        else:
            # fallback combine question + answer
            qcol = question_col if question_col in df.columns else df.columns[0]
            acol = answer_col if answer_col in df.columns else (df.columns[1] if len(df.columns) > 1 else qcol)
            texts = (df[qcol].fillna("").astype(str) + " " + df[acol].fillna("").astype(str)).tolist()

        embeddings = self.build_index(texts, normalize=normalize)

        # build metadata
        meta = []
        for i, row in df.iterrows():
            q = str(row.get(question_col) or row.get("pertanyaan") or "").strip()
            a = str(row.get(answer_col) or row.get("jawaban") or "").strip()
            cat = None
            for cand in category_col_candidates:
                if cand in row.index and row.get(cand) is not None and str(row.get(cand)).strip() != "":
                    cat = str(row.get(cand)).strip()
                    break
            if not cat:
                # try infer from question if empty
                cat = "Umum"
            meta.append({
                "index": int(i),
                "question": q,
                "answer": a,
                "category": cat
            })

        # save
        self.metadata = meta
        self.save(embeddings, meta)
        return embeddings, meta

    def save(self, embeddings: np.ndarray, metadata: List[dict]):
        if embeddings is None:
            raise ValueError("Embeddings kosong. Panggil build_index() dulu.")
        if len(metadata) != embeddings.shape[0]:
            raise ValueError("Panjang metadata harus sama dengan jumlah embeddings.")

        # ensure parent dir exists
        idx_dir = os.path.dirname(INDEX_PATH) or "."
        os.makedirs(idx_dir, exist_ok=True)
        np.save(EMBEDDINGS_NPY, embeddings)
        with open(METADATA_JSON, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        if self.index is None:
            idx = faiss.IndexFlatIP(self.dim)
            idx.add(embeddings)
            self.index = idx

        faiss.write_index(self.index, INDEX_PATH)

    def load(self) -> Tuple[np.ndarray, List[dict]]:
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

    def query(self, text: str, top_k: int = 5) -> Tuple[List[int], List[float]]:
        if self.index is None:
            raise RuntimeError("Index belum diload. Panggil load() atau build_index() dulu.")
        q_emb = self.model.encode([text], convert_to_numpy=True)
        q_emb = q_emb.astype("float32")
        faiss.normalize_L2(q_emb)
        D, I = self.index.search(q_emb, top_k)
        indices = I[0].tolist()
        scores = D[0].tolist()
        return indices, scores


if __name__ == "__main__":
    # CLI helper: python -m src.embedding_store --csv data.csv
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, help="Path to CSV dataset (optional)")
    parser.add_argument("--text-col", type=str, default="combined_text", help="col for combined text or fallback to question+answer")
    parser.add_argument("--question-col", type=str, default="question", help="question column name")
    parser.add_argument("--answer-col", type=str, default="answer", help="answer column name")
    args = parser.parse_args()

    if args.csv:
        import pandas as pd
        df = pd.read_csv(args.csv)
        store = EmbeddingStore()
        print("Building index from CSV:", args.csv)
        emb, meta = store.build_and_save_from_df(df, question_col=args.question_col, answer_col=args.answer_col, text_col=args.text_col)
        print("Saved index, embeddings, metadata:", INDEX_PATH, EMBEDDINGS_NPY, METADATA_JSON)
    else:
        print("No --csv provided. Use CLI to build index from CSV.")
