# src/retriever.py
from typing import List
try:
    from rapidfuzz import fuzz, process
except Exception:
    # graceful fallback if rapidfuzz is not installed
    fuzz = None
    process = None

import numpy as np

from .embedding_store import EmbeddingStore
from .preprocessing import Preprocessor

# safe import TOP_K from config (fall back to 5)
try:
    from .config import TOP_K
except Exception:
    TOP_K = 5

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
        """
        df: pandas DataFrame produced by Preprocessor.preprocess()
        emb_store: instance of EmbeddingStore (must implement query(text, topn) -> (indices, scores))
        """
        self.df = df
        self.emb_store = emb_store

    @staticmethod
    def multi_fuzzy(query: str, candidate: str) -> float:
        q = (query or "").lower()
        c = (candidate or "").lower()
        try:
            if fuzz is None:
                return 0.0
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
        q_clean = Preprocessor.text_cleanup(user_query)
        q_clean = self._apply_synonyms(q_clean)
        top_k = int(top_k)

        # semantic retrieval (if emb_store available)
        try:
            sem_idx, sem_scores = self.emb_store.query(q_clean, MAX_SEMANTIC_CANDIDATES)
            sem_idx = list(sem_idx) if sem_idx is not None else []
            sem_scores = list(sem_scores) if sem_scores is not None else []
        except Exception:
            sem_idx, sem_scores = [], []

        base_candidate_indices = list(sem_idx) if len(sem_idx) > 0 else list(range(len(self.df)))
        base_candidate_texts = [
            (self.df.iloc[i].get("question_clean", "") or self.df.iloc[i].get("question", "")).strip()
            for i in base_candidate_indices
        ]

        # keyword tokens (non stop)
        q_tokens = [t for t in q_clean.split() if t and t not in STOP_TOKENS]

        # keyword-first filter (generic)
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

        # compute multi-fuzzy for candidates
        fuzzy_raw = [self.multi_fuzzy(q_clean, ct) for ct in candidate_texts]

        # lexical fallback if needed
        if USE_LEXICAL_FALLBACK and (len(candidate_indices) == 0 or (len(fuzzy_raw) > 0 and max(fuzzy_raw) < MIN_FUZZY_FOR_KEYWORD and len(q_tokens) <= 2)):
            # only use process if available
            if process is not None and fuzz is not None:
                corpus = (self.df["question_clean"].fillna("") + " " + self.df["answer"].fillna("")).tolist()
                hits = process.extract(q_clean, corpus, scorer=fuzz.token_sort_ratio, limit=LEXICAL_FALLBACK_TOPN)
                candidate_indices = [h[2] for h in hits]
                candidate_texts = [self.df.iloc[i].get("question_clean", "") or self.df.iloc[i].get("question", "") for i in candidate_indices]
                fuzzy_raw = [float(h[1]) for h in hits]
                sem_scores_aligned = [0.0] * len(candidate_indices)
            else:
                # fallback to global candidates (no lexical available)
                candidate_indices = base_candidate_indices
                candidate_texts = base_candidate_texts
                fuzzy_raw = [0.0] * len(candidate_texts)
                sem_scores_aligned = [0.0] * len(candidate_indices)
        else:
            # align semantic scores
            if len(sem_scores) > 0 and len(base_candidate_indices) == len(sem_scores):
                sem_map = {int(idx): float(score) for idx, score in zip(base_candidate_indices, list(sem_scores))}
                sem_scores_aligned = [sem_map.get(idx, 0.0) for idx in candidate_indices]
            else:
                sem_scores_aligned = [0.0] * len(candidate_indices)

        # normalize + combine
        sem_norm = self.normalize(sem_scores_aligned)
        fuzzy_norm = self.normalize(fuzzy_raw)
        if len(sem_norm) < len(fuzzy_norm):
            sem_norm = np.pad(sem_norm, (0, len(fuzzy_norm) - len(sem_norm)), constant_values=0.0)

        combined = SEMANTIC_WEIGHT * np.array(sem_norm) + FUZZY_WEIGHT * np.array(fuzzy_norm)

        # boosts
        keyword_boosts = self._keyword_presence_boosts(q_tokens, candidate_texts)
        if len(keyword_boosts) == len(combined):
            combined = combined + np.array(keyword_boosts)

        qset = set(q_tokens)
        for pos, ct in enumerate(candidate_texts):
            overlap = len(qset.intersection(set((ct or "").split())))
            if overlap > 0:
                combined[pos] += TOKEN_OVERLAP_BOOST * overlap

        # ranking
        combined_list = [(int(idx), float(combined[pos])) for pos, idx in enumerate(candidate_indices)]
        combined_list = sorted(combined_list, key=lambda x: x[1], reverse=True)

        # optional rerank
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

        # select unique top_k
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

    def retrieve_by_category(self, category: str, user_query: str, k: int = TOP_K) -> List[dict]:
        """
        Retrieve within a specific category.
        Strategy:
         - Use emb_store.query to get semantic candidates globally, then filter to the category subset.
         - If no semantic candidates inside category, run fuzzy matching restricted to the category subset.
        """
        if category is None or str(category).strip() == "":
            return self.retrieve(user_query, top_k=k)

        cat = str(category).strip().lower()

        # find category column
        if "kategori" in self.df.columns:
            mask = self.df["kategori"].fillna("").astype(str).str.strip().str.lower() == cat
        elif "category" in self.df.columns:
            mask = self.df["category"].fillna("").astype(str).str.strip().str.lower() == cat
        else:
            # no category column -> fallback to global retrieve
            return self.retrieve(user_query, top_k=k)

        available_df = self.df[mask]
        if available_df.empty:
            return []

        available_indices = list(available_df.index.astype(int).tolist())

        # clean query
        q_clean = Preprocessor.text_cleanup(user_query)
        q_clean = self._apply_synonyms(q_clean)

        # 1) Get semantic candidates globally, then keep only those in category
        sem_idx, sem_scores = [], []
        try:
            sem_idx_raw, sem_scores_raw = self.emb_store.query(q_clean, MAX_SEMANTIC_CANDIDATES)
            sem_idx = list(sem_idx_raw) if sem_idx_raw is not None else []
            sem_scores = list(sem_scores_raw) if sem_scores_raw is not None else []
        except Exception:
            sem_idx, sem_scores = [], []

        sem_filtered = []
        sem_scores_filtered = []
        if sem_idx:
            for idx, sc in zip(sem_idx, sem_scores):
                if int(idx) in available_indices:
                    sem_filtered.append(int(idx))
                    sem_scores_filtered.append(float(sc))

        # if we have semantic candidates within category, rerank + return top-k
        if sem_filtered:
            candidate_indices = sem_filtered
            sem_scores_aligned = sem_scores_filtered
            candidate_texts = [
                (self.df.loc[i].get("question_clean", "") or self.df.loc[i].get("question", "")).strip()
                for i in candidate_indices
            ]
            # compute fuzzy for these candidates
            fuzzy_raw = [self.multi_fuzzy(q_clean, ct) for ct in candidate_texts]

            # normalize + combine
            sem_norm = self.normalize(sem_scores_aligned)
            fuzzy_norm = self.normalize(fuzzy_raw)
            if len(sem_norm) < len(fuzzy_norm):
                sem_norm = np.pad(sem_norm, (0, len(fuzzy_norm) - len(sem_norm)), constant_values=0.0)
            combined = SEMANTIC_WEIGHT * np.array(sem_norm) + FUZZY_WEIGHT * np.array(fuzzy_norm)

            # boosts
            q_tokens = [t for t in q_clean.split() if t and t not in STOP_TOKENS]
            keyword_boosts = self._keyword_presence_boosts(q_tokens, candidate_texts)
            if len(keyword_boosts) == len(combined):
                combined = combined + np.array(keyword_boosts)
            qset = set(q_tokens)
            for pos, ct in enumerate(candidate_texts):
                overlap = len(qset.intersection(set((ct or "").split())))
                if overlap > 0:
                    combined[pos] += TOKEN_OVERLAP_BOOST * overlap

            combined_list = [(int(idx), float(combined[pos])) for pos, idx in enumerate(candidate_indices)]
            combined_list = sorted(combined_list, key=lambda x: x[1], reverse=True)

            results = []
            seen = set()
            for idx, score in combined_list:
                if idx in seen:
                    continue
                seen.add(idx)
                row = self.df.loc[idx]
                results.append({"index": int(idx), "question": row.get("question"), "answer": row.get("answer"), "score": float(score)})
                if len(results) >= k:
                    break
            return results

        # 2) No semantic candidate in category -> perform fuzzy search on subset questions
        subset_questions = (available_df["question_clean"].fillna("") + " " + available_df["answer"].fillna("")).tolist()

        # if rapidfuzz not available, fallback to returning top-k items from category
        if process is None or fuzz is None:
            out = []
            for i in available_indices[:k]:
                row = self.df.loc[i]
                out.append({"index": int(i), "question": row.get("question"), "answer": row.get("answer"), "score": 0.0})
            return out

        hits = process.extract(q_clean, subset_questions, scorer=fuzz.token_sort_ratio, limit=min(LEXICAL_FALLBACK_TOPN, len(subset_questions)))
        if not hits:
            out = []
            for i in available_indices[:k]:
                row = self.df.loc[i]
                out.append({"index": int(i), "question": row.get("question"), "answer": row.get("answer"), "score": 0.0})
            return out

        # map hit positions to actual df indices
        subset_pos_to_idx = {pos: available_indices[pos] for pos in range(len(available_indices))}
        candidate_indices = []
        fuzzy_raw = []
        candidate_texts = []
        for h in hits:
            # hits entries are typically (matched_text, score, pos_in_subset)
            if len(h) >= 3:
                pos = int(h[2])
            else:
                # fallback: find position by string match
                try:
                    pos = subset_questions.index(h[0])
                except ValueError:
                    continue
            if pos not in subset_pos_to_idx:
                continue
            real_idx = subset_pos_to_idx[pos]
            candidate_indices.append(real_idx)
            candidate_texts.append((self.df.loc[real_idx].get("question_clean", "") or self.df.loc[real_idx].get("question", "")).strip())
            fuzzy_raw.append(float(h[1]) if len(h) >= 2 else 0.0)

        # align sem scores to zeros (no semantic)
        sem_scores_aligned = [0.0] * len(candidate_indices)

        # normalize + combine
        sem_norm = self.normalize(sem_scores_aligned)
        fuzzy_norm = self.normalize(fuzzy_raw)
        if len(sem_norm) < len(fuzzy_norm):
            sem_norm = np.pad(sem_norm, (0, len(fuzzy_norm) - len(sem_norm)), constant_values=0.0)
        combined = SEMANTIC_WEIGHT * np.array(sem_norm) + FUZZY_WEIGHT * np.array(fuzzy_norm)

        # boosts
        q_tokens = [t for t in q_clean.split() if t and t not in STOP_TOKENS]
        keyword_boosts = self._keyword_presence_boosts(q_tokens, candidate_texts)
        if len(keyword_boosts) == len(combined):
            combined = combined + np.array(keyword_boosts)
        qset = set(q_tokens)
        for pos, ct in enumerate(candidate_texts):
            overlap = len(qset.intersection(set((ct or "").split())))
            if overlap > 0:
                combined[pos] += TOKEN_OVERLAP_BOOST * overlap

        combined_list = [(int(idx), float(combined[pos])) for pos, idx in enumerate(candidate_indices)]
        combined_list = sorted(combined_list, key=lambda x: x[1], reverse=True)

        results = []
        seen = set()
        for idx, score in combined_list:
            if idx in seen:
                continue
            seen.add(idx)
            row = self.df.loc[idx]
            results.append({"index": int(idx), "question": row.get("question"), "answer": row.get("answer"), "score": float(score)})
            if len(results) >= k:
                break
        return results
