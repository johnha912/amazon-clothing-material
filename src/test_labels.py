"""Tests for src/labels.py. Run from the repo root: python -m src.test_labels"""
from src.labels import CLASSES, dominant, label, strip_composition, to_class


def test_dominant_original_cases():
    assert dominant("90% Polyester, 10% Spandex") == ("polyester", True)
    assert dominant("Imported, 100% cotton") == ("cotton", True)
    assert dominant("Genuine leather upper") == ("leather", False)
    assert dominant("Zipper closure") == (None, False)


def test_dominant_tie_first_listed_wins():
    assert dominant("50% Cotton, 50% Polyester") == ("cotton", True)


def test_dominant_aliases():
    assert dominant("95% Viscose, 5% Elastane") == ("rayon", True)


def test_label_needs_a_percentage():
    assert label("90% Polyester, 10% Spandex") == "polyester"
    assert label("Genuine leather upper") is None


def test_rare_materials_become_other_or_metal():
    assert to_class("linen") == "other"
    assert to_class("gold") == "metal"
    assert to_class(None) is None
    assert "other" in CLASSES


def test_strip_composition_keeps_the_rest():
    out = strip_composition("Soft tee. 90% Polyester, 10% Spandex. Machine wash")
    assert "%" not in out
    assert "Soft tee" in out and "Machine wash" in out


def test_no_leakage_after_stripping():
    raw = "Shell: 60% Cotton, 40% Polyester. Relaxed fit"
    assert label(raw) == "cotton"
    assert label(strip_composition(raw)) is None


if __name__ == "__main__":
    tests = [f for name, f in dict(globals()).items() if name.startswith("test_")]
    for t in tests:
        t()
    print(f"{len(tests)} tests passed")
