from __future__ import annotations

import ast
import html
import json
import re
import unicodedata
from collections.abc import Iterable
from typing import Any

from matcher.models import ProductRecord


HTML_RE = re.compile(r"<[^>]+>")
SPACE_RE = re.compile(r"\s+")
NON_WORD_RE = re.compile(r"[^a-z0-9.%+&/' -]+")
COUNT_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s*(count|ct|pieces?|pcs?)\b", re.I)
PACK_RE = re.compile(r"\b(\d+)\s*(?:pack|pk)\b", re.I)
MULTIPACK_RE = re.compile(
    r"\b(\d+)\s*[x×]\s*(\d+(?:\.\d+)?)\s*(fl\.?\s*oz|oz|lb|g|kg|ml|l)\b",
    re.I,
)
SIZE_RE = re.compile(
    r"\b(\d+(?:\.\d+)?)\s*"
    r"(fl\.?\s*oz|fluid ounces?|ounces?|oz|pounds?|lbs?|lb|grams?|g|"
    r"kilograms?|kg|milliliters?|ml|liters?|litres?|l)\b",
    re.I,
)

UNIT_ALIASES = {
    "ounce": "oz",
    "ounces": "oz",
    "oz": "oz",
    "pound": "lb",
    "pounds": "lb",
    "lbs": "lb",
    "lb": "lb",
    "gram": "g",
    "grams": "g",
    "g": "g",
    "kilogram": "kg",
    "kilograms": "kg",
    "kg": "kg",
    "fl oz": "fl_oz",
    "fl. oz": "fl_oz",
    "fluid ounce": "fl_oz",
    "fluid ounces": "fl_oz",
    "milliliter": "ml",
    "milliliters": "ml",
    "ml": "ml",
    "liter": "l",
    "liters": "l",
    "litre": "l",
    "litres": "l",
    "l": "l",
}

def clean_text(value: Any) -> str:
    """Return lowercase, punctuation-normalized product text."""
    if value is None:
        return ""
    text = html.unescape(str(value))
    text = HTML_RE.sub(" ", text)
    text = unicodedata.normalize("NFKC", text).lower()
    text = text.replace("®", " ").replace("™", " ")
    text = NON_WORD_RE.sub(" ", text)
    return SPACE_RE.sub(" ", text).strip(" -|,")


def parse_mapping(value: Any) -> dict[str, Any]:
    """Parse a JSON-like CSV field, returning an empty dict when unavailable."""
    if not value:
        return {}
    if isinstance(value, dict):
        return value
    text = str(value).strip()
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
            if isinstance(parsed, dict):
                return parsed
        except (ValueError, SyntaxError, TypeError, json.JSONDecodeError):
            pass
    return {}


def _canonical_size(value: float, unit: str) -> tuple[float, str]:
    unit = UNIT_ALIASES.get(clean_text(unit), clean_text(unit).replace(" ", "_"))
    if unit == "lb":
        return value * 16, "oz"
    if unit == "kg":
        return value * 1000, "g"
    if unit == "l":
        return value * 1000, "ml"
    return value, unit


def extract_quantity(
    name: str, sizing: dict[str, Any]
) -> tuple[str | None, float | None]:
    """Extract normalized quantity unit and total quantity."""
    texts = [name, str(sizing.get("size_user_friendly") or "")]
    for text in texts:
        if not text:
            continue

        multipack = MULTIPACK_RE.search(text)
        if multipack:
            count = int(multipack.group(1))
            value, unit = _canonical_size(
                float(multipack.group(2)), multipack.group(3)
            )
            return unit, value * count

        size_match = SIZE_RE.search(text)
        count_match = COUNT_RE.search(text)
        pack_match = PACK_RE.search(text)
        pack_count = int(pack_match.group(1)) if pack_match else None
        if pack_count is None and count_match:
            label = count_match.group(2).lower()
            if label not in {"piece", "pieces", "pcs"}:
                pack_count = int(float(count_match.group(1)))

        if size_match:
            value, unit = _canonical_size(
                float(size_match.group(1)), size_match.group(2)
            )
            return unit, value * (pack_count or 1)

        if pack_count is not None:
            return "count", float(pack_count)

    return None, None


def core_product_name(name: str, brand: str | None) -> str:
    """Remove brand, size, count, and pack information from a name."""
    text = clean_text(name)
    text = MULTIPACK_RE.sub(" ", text)
    text = SIZE_RE.sub(" ", text)
    text = COUNT_RE.sub(" ", text)
    text = PACK_RE.sub(" ", text)
    if brand and (text == brand or text.startswith(brand + " ")):
        text = text[len(brand) :]
    return SPACE_RE.sub(" ", text).strip(" ,-|")


def normalize_product(row: dict[str, Any]) -> ProductRecord:
    """Convert one CSV row into the normalized representation used for matching."""
    raw_name = str(row.get("name") or "").strip()
    item_info = parse_mapping(row.get("item_info"))
    sizing = parse_mapping(row.get("sizing_comp"))

    brand = clean_text(row.get("brand_raw")) or None
    normalized_name = clean_text(raw_name)
    product_type = core_product_name(raw_name, brand)

    category_path = tuple(
        category
        for key in ("category_0", "category_1", "category_2", "category_3")
        if (category := clean_text(item_info.get(key)))
    )
    if not category_path:
        category_path = tuple(
            category
            for value in (
                row.get("department"),
                row.get("category"),
                row.get("subcategory"),
            )
            if (category := clean_text(value))
        )

    quantity_unit, total_quantity = extract_quantity(raw_name, sizing)
    retrieval_text = " | ".join(
        part
        for part in (
            normalized_name,
            f"product {product_type}",
            f"brand {brand}" if brand else "",
            f"category {' '.join(category_path)}" if category_path else "",
        )
        if part
    )

    return ProductRecord(
        item_id=str(row.get("item_id") or "").strip(),
        raw_name=raw_name,
        normalized_name=normalized_name,
        brand=brand,
        product_type=product_type,
        category_path=category_path,
        quantity_unit=quantity_unit,
        total_quantity=total_quantity,
        retrieval_text=retrieval_text,
    )


def enrich_missing_brands(record_sets: Iterable[list[ProductRecord]]) -> None:
    """Fill missing brands from exact brand prefixes observed in the input data."""
    records = [record for record_set in record_sets for record in record_set]
    brands = {
        record.brand
        for record in records
        if record.brand and len(record.brand) >= 3
    }
    brands_by_first_word: dict[str, list[str]] = {}
    for brand in brands:
        brands_by_first_word.setdefault(brand.split()[0], []).append(brand)
    for choices in brands_by_first_word.values():
        choices.sort(key=len, reverse=True)

    for record in records:
        if record.brand or not record.normalized_name:
            continue
        first_word = record.normalized_name.split()[0]
        for brand in brands_by_first_word.get(first_word, ()):
            if record.normalized_name == brand or record.normalized_name.startswith(
                brand + " "
            ):
                record.brand = brand
                record.product_type = core_product_name(record.raw_name, brand)
                record.retrieval_text += f" | inferred brand {brand}"
                break
