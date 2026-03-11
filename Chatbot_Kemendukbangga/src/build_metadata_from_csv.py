# src/build_metadata_from_csv.py
import csv
import json
from pathlib import Path

# ROOT project = folder di atas /src
ROOT = Path(__file__).resolve().parents[1]

# CSV berada di ROOT/data/Data_Training_bkkbn.csv
CSV_PATH = ROOT / "data" / "Data_Training_bkkbn.csv"
OUT_PATH = ROOT / "metadata.json"

if not CSV_PATH.exists():
    raise SystemExit(f"CSV file not found: {CSV_PATH}")

print("Using CSV:", CSV_PATH)

# read small sample to detect delimiter
sample = CSV_PATH.read_text(encoding="utf-8", errors="ignore")[:8192]
from csv import Sniffer
dialect = None
for d in [",", ";", "\t", "|"]:
    try:
        dialect = Sniffer().sniff(sample, delimiters=d)
        break
    except Exception:
        dialect = None

delimiter = dialect.delimiter if dialect else ","
print("Detected delimiter:", repr(delimiter))

# read CSV rows
with open(CSV_PATH, "r", encoding="utf-8", errors="ignore", newline="") as f:
    reader = csv.reader(f, delimiter=delimiter)
    rows = list(reader)

if not rows:
    raise SystemExit("CSV appears empty")

header = rows[0]
if len(header) == 1 and "," in header[0] and delimiter != ",":
    header = [c.strip() for c in header[0].split(",")]

# map columns heuristically
col_map = {}
for idx, h in enumerate(header):
    h_lower = (h or "").strip().lower()
    if "kategori" in h_lower or "category" in h_lower:
        col_map["category"] = idx
    elif "pertanyaan" in h_lower or "question" in h_lower:
        col_map["question"] = idx
    elif "jawaban" in h_lower or "answer" in h_lower:
        col_map["answer"] = idx

print("Detected columns mapping:", col_map)
if "question" not in col_map:
    raise SystemExit("Cannot detect question column.")

meta = []
idx_counter = 0
for row in rows[1:]:
    def safe_get(i):
        return row[i].strip() if i is not None and i < len(row) and row[i] else ""

    q = safe_get(col_map.get("question"))
    if not q:
        continue

    a = safe_get(col_map.get("answer")) if "answer" in col_map else ""
    cat = safe_get(col_map.get("category")) if "category" in col_map else ""

    cat = cat.strip() or "Umum"

    meta.append({
        "index": idx_counter,
        "question": q,
        "answer": a,
        "category": cat
    })
    idx_counter += 1

with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print(f"metadata.json created with {len(meta)} items at:")
print(" →", OUT_PATH)
