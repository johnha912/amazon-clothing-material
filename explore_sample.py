"""Material coverage stats for the first N listings of the Amazon Reviews 2023 clothing metadata.

Get the sample:  curl -sL https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/meta_categories/meta_Clothing_Shoes_and_Jewelry.jsonl | head -n 10000 > data/meta_head.jsonl
Run:             python explore_sample.py data/meta_head.jsonl
"""
import collections, json, re, sys

MATERIALS = ["cotton", "polyester", "spandex", "nylon", "rayon", "leather", "wool", "silk",
             "linen", "acrylic", "denim", "rubber", "polyurethane", "viscose", "modal", "elastane",
             "sterling silver", "stainless steel", "gold", "alloy", "metal", "canvas", "suede", "fleece"]
ALIAS = {"elastane": "spandex", "viscose": "rayon", "modal": "rayon", "suede": "leather",
         "faux leather": "leather", "pu": "polyurethane"}
DETAIL_KEYS = ["Material", "Fabric Type", "Outer Material", "Sole Material", "Metal Type", "Material Type"]
PCT = re.compile(r"(\d{1,3})\s*%\s*([a-z][a-z ]+)")

def dominant(text):
    """Highest-percentage known material in the text, else the first known material mentioned."""
    text = text.lower()
    best = None
    for pct, name in PCT.findall(text):
        for m in MATERIALS:
            if name.strip().startswith(m) and (best is None or int(pct) > best[0]):
                best = (int(pct), ALIAS.get(m, m))
    if best:
        return best[1], True
    hits = [(text.find(m), m) for m in MATERIALS if re.search(rf"\b{m}\b", text)]
    return (ALIAS.get(min(hits)[1], min(hits)[1]), False) if hits else (None, False)

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

assert dominant("90% Polyester, 10% Spandex") == ("polyester", True)
assert dominant("Imported, 100% cotton") == ("cotton", True)
assert dominant("Genuine leather upper") == ("leather", False)
assert dominant("Zipper closure") == (None, False)
