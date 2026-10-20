"""Shared pandas method chain: raw clothing metadata JSONL -> one tidy row per product.

    from pipeline import load
    df = load("data/meta_head.jsonl")

Every step takes a DataFrame and returns a new one, so steps chain with .pipe().
Add a step as a function here and one .pipe() line in load(); don't edit the frame in place.
Self-check: python pipeline.py data/meta_head.jsonl
"""
import re
import sys

import pandas as pd

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


def read_raw(path, nrows=None):
    return pd.read_json(path, lines=True, nrows=nrows)


def tidy(df):
    """Keep the columns we model on; join the free text into one string."""
    details = df.details.map(lambda d: d if isinstance(d, dict) else {})
    return (df
            .assign(department=details.map(lambda d: d.get("Department")).str.lower(),
                    composition=[" ".join([*(str(d[k]) for k in DETAIL_KEYS if k in d), *f])
                                 for d, f in zip(details, df.features)],
                    text=lambda d: d.title.fillna("") + " " + d.features.str.join(" ")
                                   + " " + d.description.str.join(" "))
            [["parent_asin", "title", "text", "composition", "department", "price",
              "average_rating", "rating_number"]])


def add_material(df, top=10):
    """Label = dominant material; materials outside the `top` most common become "other"."""
    labels = df.composition.map(dominant)
    return (df
            .assign(material=labels.str[0], has_pct=labels.str[1])
            .assign(material=lambda d: d.material.where(
                d.material.isna() | d.material.isin(d.material.value_counts().index[:top]), "other"))
            # README: label strings must not leak into model input.
            # ponytail: masks only "NN% fiber" spans; bare material words still leak via the fallback label
            .assign(text=lambda d: d.text.str.lower().str.replace(PCT, " ", regex=True))
            .drop(columns="composition"))


def load(path, nrows=None):
    return (read_raw(path, nrows)
            .pipe(tidy)
            .pipe(add_material))


if __name__ == "__main__":
    assert dominant("90% Polyester, 10% Spandex") == ("polyester", True)
    assert dominant("Imported, 100% cotton") == ("cotton", True)
    assert dominant("Genuine leather upper") == ("leather", False)
    assert dominant("Zipper closure") == (None, False)
    if len(sys.argv) > 1:
        df = load(sys.argv[1])
        assert not df.text.str.contains(r"\d+\s*%\s*polyester").any()
        df.info()
        print(df.material.value_counts(dropna=False))
