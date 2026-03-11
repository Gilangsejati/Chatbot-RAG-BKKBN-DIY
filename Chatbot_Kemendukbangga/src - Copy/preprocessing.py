# src/preprocessing.py
import re
import unicodedata
from typing import List, Tuple, Optional

import pandas as pd

from .config import DATA_PATH


class Preprocessor:
    """
    Loader + text preprocessing untuk dataset FAQ.
    - Membaca CSV dari DATA_PATH
    - Mendeteksi dan mapping kolom (PERTANYAAN/JAWABAN / question/answer)
    - Membersihkan teks (lowercase, remove punctuation, normalize unicode)
    - Menyediakan helper untuk membuat teks gabungan (question + answer) untuk embedding
    """

    def __init__(self, path: str = DATA_PATH):
        self.path = path
        self.df: Optional[pd.DataFrame] = None

    def _read_csv_flex(self) -> pd.DataFrame:
        """
        Baca CSV dengan percobaan delimiter umum. Kembalikan DataFrame.
        """
        # coba default comma dulu
        try:
            df = pd.read_csv(self.path)
            # jika hanya 1 kolom dan dipisah oleh ';' kemungkinan delimiter lain
            if df.shape[1] == 1:
                # coba ; sebagai delimiter
                df = pd.read_csv(self.path, sep=";")
        except Exception:
            # fallback: pakai sep=';'
            df = pd.read_csv(self.path, sep=";")
        return df

    def load_csv(self) -> pd.DataFrame:
        """
        Baca CSV, normalisasi nama kolom, dan mapping otomatis ke 'question' & 'answer'.
        Akan melempar ValueError jika kolom tidak ditemukan.
        """
        df = self._read_csv_flex()

        # normalisasi nama kolom (strip + lower)
        orig_cols = list(df.columns)
        cols_normalized = [c.strip().lower() for c in orig_cols]
        col_map = {orig: norm for orig, norm in zip(orig_cols, cols_normalized)}
        df = df.rename(columns=col_map)

        # mapping otomatis: cari kolom yang mengandung kata 'pertanyaan' / 'question'
        mapping = {}
        for col in df.columns:
            if "pertanyaan" in col or "question" in col:
                mapping[col] = "question"
            elif "jawaban" in col or "jawab" in col or "answer" in col:
                mapping[col] = "answer"
            # optional: keep kategori / referensi if needed later
            elif "kategori" in col:
                mapping[col] = "kategori"
            elif "referensi" in col or "reference" in col:
                mapping[col] = "referensi"

        # rename according to mapping
        df = df.rename(columns=mapping)

        # validasi: must have question & answer
        if "question" not in df.columns or "answer" not in df.columns:
            raise ValueError(
                "Kolom tidak ditemukan! Kolom yang tersedia: "
                f"{list(df.columns)}. Pastikan ada kolom PERTANYAAN dan JAWABAN (atau question/answer)."
            )

        # select only needed columns and drop rows with missing question or answer
        df = df[["question", "answer"]].copy()
        df = df.dropna(subset=["question"]).reset_index(drop=True)
        # keep answer even if empty string; convert to string
        df["answer"] = df["answer"].fillna("").astype(str)

        self.df = df
        return self.df

    @staticmethod
    def text_cleanup(text: str) -> str:
        """
        Bersihkan teks:
         - pastikan string
         - unicode normalize
         - lowercase
         - hilangkan tanda baca (tetap simpan angka & huruf)
         - collapse spasi
        """
        if text is None:
            return ""
        if not isinstance(text, str):
            text = str(text)

        # unicode normalize (menghilangkan accent)
        text = unicodedata.normalize("NFKD", text)

        # ubah ke lower dan strip
        text = text.strip().lower()

        # replace punctuation with space (keep alphanumeric & underscore & whitespace)
        # \w == [a-zA-Z0-9_], kita ingin juga mempertahankan underscore jika ada
        text = re.sub(r"[^\w\s]", " ", text)

        # collapse multiple whitespace menjadi single space
        text = " ".join(text.split())

        return text

    def preprocess(self) -> pd.DataFrame:
        """
        Jalankan full preprocessing: load, buat question_clean & answer_clean,
        serta combined_text (question_clean + ' ' + answer_clean).
        """
        if self.df is None:
            self.load_csv()

        # apply cleanup
        self.df["question_clean"] = self.df["question"].apply(self.text_cleanup)
        self.df["answer_clean"] = self.df["answer"].apply(self.text_cleanup)

        # combined text untuk embedding (gabungan question + answer)
        self.df["combined_text"] = (self.df["question_clean"].fillna("") + " " + self.df["answer_clean"].fillna("")).str.strip()

        return self.df

    def get_combined_texts(self) -> List[str]:
        """
        Helper: kembalikan list teks gabungan yang siap di-embed.
        Pastikan sudah memanggil preprocess() dulu.
        """
        if self.df is None:
            self.preprocess()
        return self.df["combined_text"].tolist()

    def get_questions_answers(self) -> List[Tuple[str, str]]:
        """
        Helper: kembalikan list tuple (question, answer).
        """
        if self.df is None:
            self.preprocess()
        return list(self.df[["question", "answer"]].itertuples(index=False, name=None))


if __name__ == "__main__":
    # quick test: jalankan dari root project: python -m src.preprocessing
    print("Running quick preprocessing test...")
    pre = Preprocessor()
    df = pre.preprocess()
    print("Jumlah baris:", len(df))
    print("Kolom:", df.columns.tolist())
    print("\nContoh 5 baris:")
    print(df.head(5).to_string(index=False))
    print("\nContoh combined_text (5):")
    for i, txt in enumerate(pre.get_combined_texts()[:5]):
        print(f"{i+1}. {txt[:200]}{'...' if len(txt)>200 else ''}")
