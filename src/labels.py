"""The labeling rule for the dominant material. This is the only place it lives (see AGENTS.md).

dominant() is moved here unchanged from explore_sample.py.
"""
import re

MATERIALS = ["cotton", "polyester", "spandex", "nylon", "rayon", "leather", "wool", "silk",
             "linen", "acrylic", "denim", "rubber", "polyurethane", "viscose", "modal", "elastane",
             "sterling silver", "stainless steel", "gold", "alloy", "metal", "canvas", "suede", "fleece"]
ALIAS = {"elastane": "spandex", "viscose": "rayon", "modal": "rayon", "suede": "leather",
         "faux leather": "leather", "pu": "polyurethane"}
DETAIL_KEYS = ["Material", "Fabric Type", "Outer Material", "Sole Material", "Metal Type", "Material Type"]
PCT = re.compile(r"(\d{1,3})\s*%\s*([a-z][a-z ]+)")

# Rare materials merged into about ten classes. Confirm this list with Nguyen and Yanglin
# before the 250k sample, because every model trains on these names.
CLASS_OF = {
    "polyester": "polyester", "cotton": "cotton", "leather": "leather", "rubber": "rubber",
    "spandex": "spandex", "nylon": "nylon", "rayon": "rayon", "wool": "wool",
    "sterling silver": "metal", "stainless steel": "metal", "gold": "metal",
    "alloy": "metal", "metal": "metal",
}
CLASSES = sorted(set(CLASS_OF.values())) + ["other"]

_names = sorted({*MATERIALS, *ALIAS}, key=len, reverse=True)
COMPOSITION = re.compile(r"\b\d{1,3}\s*%\s*(?:" + "|".join(map(re.escape, _names)) + r")\b", re.IGNORECASE)


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


def to_class(material):
    """Map a material name to one of CLASSES. None stays None."""
    if material is None:
        return None
    return CLASS_OF.get(material, "other")


def label(text):
    """Class label from a percentage composition only. No composition means no label (None)."""
    material, has_pct = dominant(text)
    return to_class(material) if has_pct else None


def strip_composition(text):
    """Remove composition strings such as '90% Polyester' so the model can't read the answer."""
    return re.sub(r"\s{2,}", " ", COMPOSITION.sub(" ", text)).strip()
