from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ProductRecord:
    item_id: str
    raw_name: str
    normalized_name: str
    brand: str | None
    product_type: str
    category_path: tuple[str, ...]
    quantity_unit: str | None
    total_quantity: float | None
    retrieval_text: str


@dataclass(slots=True)
class Candidate:
    b_index: int
    exact_name: bool = False
    lexical_score: float = 0.0
    vector_score: float = 0.0
