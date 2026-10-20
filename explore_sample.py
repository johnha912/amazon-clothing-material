"""Material coverage stats for the first N listings of the Amazon Reviews 2023 clothing metadata.

Get the sample:  curl -sL https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/meta_categories/meta_Clothing_Shoes_and_Jewelry.jsonl | head -n 10000 > data/meta_head.jsonl
Run:             python explore_sample.py data/meta_head.jsonl
"""
import collections, json, sys

from pipeline import DETAIL_KEYS, dominant

rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8")]
n = len(rows)
detail = sum(any(k in (r.get("details") or {}) for k in DETAIL_KEYS) for r in rows)
labels, with_pct = collections.Counter(), 0
for r in rows:
    d = r.get("details") or {}
    text = " ".join([*(str(d[k]) for k in DETAIL_KEYS if k in d), *r.get("features", [])])
    m, pct = dominant(text)
    if m:
        labels[m] += 1
        with_pct += pct
labeled = sum(labels.values())
print(f"n={n} detail_field={detail} ({detail/n:.1%}) any_material={labeled} ({labeled/n:.1%}) "
      f"with_percent={with_pct} ({with_pct/n:.1%}) price={sum(r.get('price') is not None for r in rows)/n:.1%} "
      f"parents={len({r['parent_asin'] for r in rows})}")
for m, c in labels.most_common(12):
    print(f"  {m:16s} {c:5d} {c/labeled:.1%}")

