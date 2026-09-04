from matcher.normalize import (
    clean_text,
    enrich_missing_brands,
    extract_quantity,
    normalize_product,
)


def test_clean_text_handles_html_and_unicode() -> None:
    assert clean_text("<b>Garbanzo Beans®</b>") == "garbanzo beans"


def test_extract_quantity_normalizes_pounds() -> None:
    assert extract_quantity("Beans 1 lb", {}) == ("oz", 16)


def test_extract_quantity_handles_multipack() -> None:
    assert extract_quantity("Juice 2 x 8 fl oz", {}) == ("fl_oz", 16)


def test_brand_prefix_is_removed_from_product_type() -> None:
    record = normalize_product(
        {
            "item_id": "1",
            "name": "Great Value Organic Garbanzo Beans, 15 oz",
            "brand_raw": "Great Value",
            "is_private_label": True,
            "item_info": '{"category_0":"Food","category_1":"Beans"}',
            "sizing_comp": "{}",
        }
    )
    assert record.brand == "great value"
    assert record.product_type == "organic garbanzo beans"
    assert record.total_quantity == 15


def test_missing_brands_are_inferred_from_observed_brand_prefixes() -> None:
    known_a = normalize_product({"item_id": "1", "name": "Acme Beans", "brand_raw": "Acme"})
    missing_a = normalize_product({"item_id": "2", "name": "Acme Rice"})
    known_b = normalize_product({"item_id": "3", "name": "Acme Pasta", "brand_raw": "Acme"})

    enrich_missing_brands(([known_a, missing_a], [known_b]))

    assert missing_a.brand == "acme"
    assert missing_a.product_type == "rice"
