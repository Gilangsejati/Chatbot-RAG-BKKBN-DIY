# src/retriever.py
from typing import List, Tuple
from rapidfuzz import fuzz, process
import numpy as np

from .embedding_store import EmbeddingStore
from .preprocessing import Preprocessor
from .config import TOP_K

# ----------------- TUNABLE PARAMETERS -----------------
SEMANTIC_WEIGHT = 0.5    # bobot untuk similarity semantic (embedding)
FUZZY_WEIGHT = 0.5       # bobot untuk similarity fuzzy (textual)
KEYWORD_BOOST_SCORE = 1.0  # kuatkan boost sehingga keyword presence menang
KEYWORD_THRESHOLD = 65     # fuzzy threshold untuk dianggap match kuat
TOKEN_OVERLAP_BOOST = 0.03 # boost kecil per token overlap
MAX_SEMANTIC_CANDIDATES = 5 * TOP_K  # ambil semantic lebih banyak untuk rerank
STOP_TOKENS = {"apa", "yang", "dimaksud", "dengan", "jelaskan", "tentang", "itu", "adalah", "apa saja"}

# lexical fallback settings (very helpful for typos)
USE_LEXICAL_FALLBACK = True
LEXICAL_FALLBACK_TOPN = 20
MIN_FUZZY_FOR_KEYWORD = 40  # jika fuzzy paling tinggi < ini dan query pendek -> lakukan fallback

# ----------------- OPTIONAL SYNONYM NORMALIZATION -----------------
SYNONYM_MAP = {
    "dasyat": "dashat",
    "dasat": "dashat",
    # tambahkan typo umum lainnya di sini
}

# ----------------- OPTIONAL RERANKER (Cross-Encoder) -----------------
USE_RERANKER = False
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
try:
    if USE_RERANKER:
        from sentence_transformers import CrossEncoder
        reranker = CrossEncoder(RERANK_MODEL_NAME)
    else:
        reranker = None
except Exception:
    reranker = None


class Retriever:
    def __init__(self, df, emb_store: EmbeddingStore):
        self.df = df
        self.emb_store = emb_store

    @staticmethod
    def multi_fuzzy(query: str, candidate: str) -> float:
        q = (query or "").lower()
        c = (candidate or "").lower()
        try:
            s1 = fuzz.partial_ratio(q, c)
            s2 = fuzz.token_sort_ratio(q, c)
            s3 = fuzz.token_set_ratio(q, c)
            return float((s1 + s2 + s3) / 3.0)
        except Exception:
            return 0.0

    @staticmethod
    def normalize(arr: List[float]) -> np.ndarray:
        a = np.array(arr, dtype=float)
        if a.size == 0:
            return a
        mn, mx = float(a.min()), float(a.max())
        if mx - mn <= 1e-12:
            return np.zeros_like(a)
        return (a - mn) / (mx - mn)

    def _apply_synonyms(self, text: str) -> str:
        if not SYNONYM_MAP:
            return text
        t = text
        for k, v in SYNONYM_MAP.items():
            if k in t:
                t = t.replace(k, v)
        return t

    def _keyword_presence_boosts(self, q_tokens: List[str], candidate_texts: List[str]) -> List[float]:
        boosts = []
        q_set = set([t for t in q_tokens if t and t not in STOP_TOKENS])
        for txt in candidate_texts:
            txt_set = set((txt or "").lower().split())
            if len(q_set.intersection(txt_set)) > 0:
                boosts.append(float(KEYWORD_BOOST_SCORE))
            else:
                boosts.append(0.0)
        return boosts

    def retrieve(self, user_query: str, top_k: int = TOP_K) -> List[dict]:
        # 1) clean + synonyms
        q_clean = Preprocessor.text_cleanup(user_query)
        q_clean = self._apply_synonyms(q_clean)
        top_k = int(top_k)

        # 2) semantic retrieval
        try:
            sem_idx, sem_scores = self.emb_store.query(q_clean, MAX_SEMANTIC_CANDIDATES)
        except Exception:
            sem_idx, sem_scores = [], []

        base_candidate_indices = list(sem_idx) if len(sem_idx) > 0 else list(range(len(self.df)))
        base_candidate_texts = [
            (self.df.iloc[i].get("question_clean", "") or self.df.iloc[i].get("question", "")).strip()
            for i in base_candidate_indices
        ]

        # keyword tokens (non stop)
        q_tokens = [t for t in q_clean.split() if t and t not in STOP_TOKENS]

        # 3) keyword-first filter (generic)
        filtered_indices = []
        filtered_texts = []
        if len(q_tokens) > 0:
            for i, txt in zip(base_candidate_indices, base_candidate_texts):
                for t in q_tokens:
                    if t in txt:
                        filtered_indices.append(i)
                        filtered_texts.append(txt)
                        break

        if len(filtered_indices) > 0:
            candidate_indices = filtered_indices
            candidate_texts = filtered_texts
        else:
            candidate_indices = base_candidate_indices
            candidate_texts = base_candidate_texts

        # 4) compute multi-fuzzy for candidates
        fuzzy_raw = [self.multi_fuzzy(q_clean, ct) for ct in candidate_texts]

        # 5) lexical fallback if needed (helps heavy-typo & very short queries)
        if USE_LEXICAL_FALLBACK and (len(candidate_indices) == 0 or (len(fuzzy_raw) > 0 and max(fuzzy_raw) < MIN_FUZZY_FOR_KEYWORD and len(q_tokens) <= 2)):
            corpus = (self.df["question_clean"].fillna("") + " " + self.df["answer"].fillna("")).tolist()
            hits = process.extract(q_clean, corpus, scorer=fuzz.token_sort_ratio, limit=LEXICAL_FALLBACK_TOPN)
            candidate_indices = [h[2] for h in hits]
            candidate_texts = [self.df.iloc[i].get("question_clean", "") or self.df.iloc[i].get("question", "") for i in candidate_indices]
            fuzzy_raw = [float(h[1]) for h in hits]
            sem_scores_aligned = [0.0] * len(candidate_indices)
        else:
            # align semantic scores
            if len(sem_scores) > 0:
                sem_map = {int(idx): float(score) for idx, score in zip(base_candidate_indices, list(sem_scores))}
                sem_scores_aligned = [sem_map.get(idx, 0.0) for idx in candidate_indices]
            else:
                sem_scores_aligned = [0.0] * len(candidate_indices)

        # 6) normalize + combine
        sem_norm = self.normalize(sem_scores_aligned)
        fuzzy_norm = self.normalize(fuzzy_raw)
        if len(sem_norm) < len(fuzzy_norm):
            sem_norm = np.pad(sem_norm, (0, len(fuzzy_norm) - len(sem_norm)), constant_values=0.0)

        combined = SEMANTIC_WEIGHT * np.array(sem_norm) + FUZZY_WEIGHT * np.array(fuzzy_norm)

        # 7) boosts
        keyword_boosts = self._keyword_presence_boosts(q_tokens, candidate_texts)
        if len(keyword_boosts) == len(combined):
            combined = combined + np.array(keyword_boosts)

        qset = set(q_tokens)
        for pos, ct in enumerate(candidate_texts):
            overlap = len(qset.intersection(set((ct or "").split())))
            if overlap > 0:
                combined[pos] += TOKEN_OVERLAP_BOOST * overlap

        # 8) ranking
        combined_list = [(int(idx), float(combined[pos])) for pos, idx in enumerate(candidate_indices)]
        combined_list = sorted(combined_list, key=lambda x: x[1], reverse=True)

        # 9) optional rerank
        if reranker is not None and len(combined_list) > 0:
            topn = min(15, len(combined_list))
            top_candidates = combined_list[:topn]
            pairs = [[q_clean, str(self.df.iloc[idx]["answer"])] for idx, _ in top_candidates]
            try:
                rerank_scores = reranker.predict(pairs)
                for i, (idx, _) in enumerate(top_candidates):
                    combined_list[i] = (idx, float(rerank_scores[i]))
                combined_list = sorted(combined_list, key=lambda x: x[1], reverse=True)
            except Exception:
                pass

        # 10) select unique top_k
        seen = set()
        results: List[dict] = []
        for idx, score in combined_list:
            if idx in seen:
                continue
            seen.add(idx)
            row = self.df.iloc[idx]
            results.append({
                "index": int(idx),
                "question": row.get("question"),
                "answer": row.get("answer"),
                "score": float(score)
            })
            if len(results) >= top_k:
                break

        return results


# --------------------- QUICK DEBUG RUN ---------------------
if __name__ == "__main__":
    # jalankan dari root project: python -m src.retriever
    try:
        from .preprocessing import Preprocessor  # relative ok when -m src.retriever
        from .embedding_store import EmbeddingStore
    except Exception:
        from src.preprocessing import Preprocessor
        from src.embedding_store import EmbeddingStore

    pre = Preprocessor()
    df = pre.preprocess()

    store = EmbeddingStore()
    try:
        store.load()
    except Exception:
        print("Index tidak ditemukan — membangun index sekarang (dari combined_text)...")
        emb = store.build_index(pre.get_combined_texts())
        meta = [{"index": i, "question": row["question"], "answer": row["answer"]} for i, row in df.iterrows()]
        store.metadata = meta
        store.save(emb, meta)

    retriever = Retriever(df, store)

    queries = [
        "apa yang dimaksud dengan dashat",
        "jelaskan mengenai dashat",
        "apa yang dimaksud pencatatan dan pelaporan Kampung Keluarga Berkualitas",
        "apa itu keluarga"
    ]

    for q in queries:
        print("\n=== Query:", q)
        res = retriever.retrieve(q, top_k=5)
        for r in res:
            print(f"- idx={r['index']} score={r['score']:.4f} Q: {r['question']}")
